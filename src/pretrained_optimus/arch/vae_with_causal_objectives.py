import math
import numpy as np
import torch
import torch.nn as nn

from .utils import log_sum_exp

from . import BertForLatentConnector, GPT2Encoder


class VAEClasBak(nn.Module):
    """VAE with normal prior"""
    def __init__(self,
                 encoder,
                 decoder,
                 classifier,
                 tokenizer_encoder,
                 tokenizer_decoder,
                 tokenizer_classifier,
                 args): #
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.classifier = classifier

        #
        self.num_c_labels = 2
        self.c_layer = nn.Linear(args.latent_size - args.attribute_dim, self.num_c_labels, bias=False)

        # For inference-time control
        self.num_a_labels = self.classifier.num_labels
        #self.a_layer_given_z = nn.Linear(args.latent_size - args.attribute_dim, self.num_a_labels, bias=False)
        #self.a_layer_given_z = nn.Sequential(
        #    nn.Linear(args.latent_size - args.attribute_dim, args.latent_size - args.attribute_dim),
        #    nn.ReLU(),
        #    nn.Linear(args.latent_size - args.attribute_dim, args.latent_size - args.attribute_dim),
        #    nn.ReLU(),
        #    nn.Linear(args.latent_size - args.attribute_dim, self.num_a_labels)
        #)
        #mid_size = (args.latent_size - args.attribute_dim) * 2
        mid_size = (args.latent_size - args.attribute_dim)
        self.a_layer_given_z = nn.Sequential(
            nn.Linear(args.latent_size - args.attribute_dim, mid_size),
            nn.ReLU(),
            nn.Linear(mid_size, self.num_a_labels)
        )

        self.args = args
        self.nz = args.latent_size
        self.nattr = args.attribute_dim
        print('latent size: ', self.nz)
        print('attribute dim: ', self.nattr)


        # For simulating z posterior
        # From cara.py
        self.latent_generator = nn.Linear(self.nz - self.nattr, self.nz - self.nattr)
        self.latent_discriminator = nn.Linear(self.nz - self.nattr, 1)
        # ======

        self.bos_token_id = tokenizer_decoder.convert_tokens_to_ids([tokenizer_decoder.bos_token])[0]
        self.eos_token_id = tokenizer_decoder.convert_tokens_to_ids([tokenizer_decoder.eos_token])[0]
        self.pad_token_id = tokenizer_decoder.convert_tokens_to_ids([tokenizer_decoder.pad_token])[0]
        assert tokenizer_decoder.bos_token == tokenizer_classifier.bos_token
        assert tokenizer_decoder.eos_token == tokenizer_classifier.eos_token
        assert tokenizer_decoder.pad_token == tokenizer_classifier.pad_token

        # TODO(hzt): from attribute label to hidden vector
        self.attr_layer = nn.Linear(1, self.nattr, bias=False)

        # connector: from Bert hidden units to the latent space
        # self.linear = nn.Linear(args.nz, 2 * args.nz, bias=False)

        # Standard Normal prior
        #loc = torch.zeros(self.nz, device=args.device)
        #scale = torch.ones(self.nz, device=args.device)
        loc = torch.zeros(self.nz - self.nattr, device=args.device)  # TODO(hzt)
        scale = torch.ones(self.nz - self.nattr, device=args.device)  # TODO(hzt)
        self.prior = torch.distributions.normal.Normal(loc, scale)


    def connect(self, bert_fea, nsamples=1):
        """
        Returns: Tensor1, Tensor2
            Tensor1: the tensor latent z with shape [batch, nsamples, nz]
            Tensor2: the tenor of KL for each x with shape [batch]
        """

        # (batch_size, nz)

        mean, logvar = self.encoder.linear(bert_fea).chunk(2, -1)
        # mean, logvar = mean.squeeze(0), logvar.squeeze(0)

        mean, logvar = mean[:, :-self.nattr], logvar[:, :-self.nattr]  # TODO(hzt)

        # (batch, nsamples, nz)
        z = self.reparameterize(mean, logvar, nsamples)
        KL = 0.5 * (mean.pow(2) + logvar.exp() - logvar - 1).sum(dim=1)

        return z, KL

    def connect_deterministic(self, bert_fea, nsamples=1):
        """
        Returns: Tensor1, Tensor2
            Tensor1: the tensor latent z with shape [batch, nsamples, nz]
            Tensor2: the tenor of KL for each x with shape [batch]
        """

        # (batch_size, nz)

        mean, logvar = self.encoder.linear(bert_fea).chunk(2, -1)
        # mean, logvar = mean.squeeze(0), logvar.squeeze(0)

        mean, logvar = mean[:, :-self.nattr], logvar[:, :-self.nattr]  # TODO(hzt)

        logvar.fill_(.0)
        # (batch, nsamples, nz)
        #z = self.reparameterize(mean, logvar, nsamples)  # TODO(hzt)
        z = mean  # TODO(hzt)
        KL = 0.5 * (mean.pow(2) + logvar.exp() - logvar - 1).sum(dim=1)

        return z, KL


    def reparameterize(self, mu, logvar, nsamples=1):
        """sample from posterior Gaussian family
        Args:
            mu: Tensor
                Mean of gaussian distribution with shape (batch, nz)
            logvar: Tensor
                logvar of gaussian distibution with shape (batch, nz)
        Returns: Tensor
            Sampled z with shape (batch, nsamples, nz)
        """
        batch_size, nz = mu.size()
        std = logvar.mul(0.5).exp()

        mu_expd = mu.unsqueeze(1).expand(batch_size, nsamples, nz)
        std_expd = std.unsqueeze(1).expand(batch_size, nsamples, nz)

        eps = torch.zeros_like(std_expd).normal_()

        return mu_expd + torch.mul(eps, std_expd)


    def _get_reverse_attributes(self, a, multi=False):
        if multi:
            # 0 -> 2
            # 1 -> 3
            # 2 -> 0
            # 3 -> 1
            return (a + 2) % 4
        else:
            return 1 - a


    def _get_sentiment_attributes(self, a, multi=False):
        if multi:
            # 0 -> 0
            # 1 -> 0
            # 2 -> 1
            # 3 -> 1
            return (a > 1).long()
        else:
            return a


    def _get_category_attributes(self, a, multi=False):
        if multi:
            # 0 -> 0
            # 1 -> 1
            # 2 -> 0
            # 3 -> 1
            return (a % 2).long()
        else:
            return a


    def _add_attributes(self, z, a, concat=True):
        """
        z.shape == [batch_size, dim] or [batch_size, nsamples, dim]
        a.shape == [batch_size]
        """
        batch_size = a.shape[0]

        a_fea = self.attr_layer(a.unsqueeze(-1).float())  # [batch_size, self.nattr]

        if z.dim() == 3:
            a_fea = a_fea.view([batch_size, 1, -1]).repeat(1, z.shape[1], 1)

        #if z.dim() == 2:
        #    a = a.unsqueeze(-1)
        #elif z.dim() == 3:
        #    a = a.view([batch_size,1,1]).repeat(1,z.shape[1],1)
        #else:
        #    raise NotImplementedError

        if concat:
            z_new = torch.cat( (z, a_fea), -1)
        else:
            z_new = torch.cat( (z[:,:-self.nattr], a_fea), -1)
        return z_new

    def _get_z_part(self, z):
        if z.dim() == 3:
            return z[:, :, :-self.nattr]
        else:
            return z[:, :-self.nattr]

    def _normalize_z(self, z):
        assert z.dim() == 2
        z_min = torch.min(z, -1, keepdim=True).values.detach()
        z_ = z - z_min
        z_max = torch.max(z_, -1, keepdim=True).values.detach()
        return z_ / (z_max + 1e-10)

    def _append_a(self, x, a):
        """
            '0 . ': [657, 764]
            '1 . ': [352, 764]

            a.shape = [batch_size]
        """
        batch_size = x.shape[0]
        a0 = torch.from_numpy(np.repeat([[657, 764]], batch_size, axis=0)).to('cuda')
        a1 = torch.from_numpy(np.repeat([[352, 764]], batch_size, axis=0)).to('cuda')
        a_ = a.unsqueeze(1)
        prefix = a0 * (1-a_) + a1 * a_
        return torch.cat( (prefix, x), 1 )

    def _append_a_embeds(self, x_embeds, a, wte):
        """
        wte: nn.Embedding
        """
        batch_size = x_embeds.shape[0]
        a0 = torch.from_numpy(np.repeat([[657, 764]], batch_size, axis=0)).to('cuda')
        a1 = torch.from_numpy(np.repeat([[352, 764]], batch_size, axis=0)).to('cuda')
        a_ = a.unsqueeze(1)
        prefix = a0 * (1-a_) + a1 * a_
        prefix_embeds = wte(prefix)
        return torch.cat( (prefix_embeds, x_embeds), 1 )

    def _c_predict(self, fea, labels=None):
        #print('labels: ')
        #print(labels)
        logits = self.c_layer(fea)
        loss = None
        if labels is not None:
            label_mask = (labels != -1).long()  # -1 means no label available
            labels_ = labels * label_mask
            loss_fn = torch.nn.CrossEntropyLoss(reduction='none')
            loss = loss_fn(logits.view(-1, self.num_c_labels), labels_.view(-1))
            loss = (loss * label_mask.float()).sum() / (label_mask.sum() + 1e-6)
        return loss, logits

    def _a_predict(self, fea, labels=None):
        """
        Predicts attribute given z
        """
        logits = self.a_layer_given_z(fea)
        loss = None
        if labels is not None:
            loss_fn = torch.nn.CrossEntropyLoss()
            loss = loss_fn(logits.view(-1, self.num_a_labels), labels.view(-1))
        return loss, logits


    def forward(self,
                inputs,
                labels,
                attributes=None,
                attributes_2=None,
                lambda_recon=1.0,
                lambda_clas=0.0,
                lambda_reg_z=0.0,
                lambda_c_loss=0.0,
                lambda_reg_z_c=0.0,
                temperature=1.0,
                max_length=20,
                use_gumbel=True,
                tokenizer_decoder=None,
                tokenizer_encoder=None,
                hard=False,
                cond_a=False,
                cond_c=False,
                train_a_layer=False,
                train_gan=False):
        """
        attributes: shape = [batch_size]
        """

        reconstrution_mask=(labels != 50257).float() # 50257 is the padding token for GPT2
        sent_length = torch.sum(reconstrution_mask, dim=1)

        #for b in range(inputs.shape[0]):
        #    text_x1_data = tokenizer_decoder.decode(inputs[b].tolist(), clean_up_tokenization_spaces=False)
        #    text_x1_data = text_x1_data.split('<EOS>')[0].strip()
        #    print('%d\t%s' % (attributes_2.tolist()[b], text_x1_data))


        if isinstance(self.encoder, BertForLatentConnector):
            attention_mask=(inputs > 0).float()
            outputs = self.encoder(inputs, attention_mask)
            pooled_hidden_fea = outputs[1]  # model outputs are always tuple in pytorch-transformers (see doc)
        elif isinstance(self.encoder, GPT2Encoder):
            encoder_inputs = inputs
            if self.args.multi_attribute:
                if cond_a and cond_c:
                    raise NotImplementedError
                elif cond_a:
                    encoder_input = self._append_a(inputs, self._get_sentiment_attributes(attributes, self.args.multi_attribute))
                elif cond_c:
                    encoder_input = self._append_a(inputs, self._get_category_attributes(attributes, self.args.multi_attribute))
            else:
                if cond_c:
                    raise NotImplementedError
                if cond_a:
                    encoder_input = model_vae._append_a(x0, attributes)
            outputs = self.encoder(
                encoder_inputs,
                #attention_mask=attention_mask,
                end_token_id_or_embeds=self.eos_token_id)
            assert self.eos_token_id == 50259
            pooled_hidden_fea = outputs

        #max_length = int(sent_length.max().item())  # TODO(hzt)
        #max_length = min(28, int(sent_length.max().item()))
        #reconstrution_mask = reconstrution_mask[:, :max_length]
        #sent_length = torch.sum(reconstrution_mask, dim=1)
        #labels =

        #if max_length > 25:
        #    print('max_length: ', max_length, '\n')
        #    for i in range(sent_length.shape[0]):
        #        if sent_length[i].item() > 25:
        #            print(sent_length[i], flush=True)
        #            print(tokenizer_decoder.decode(labels[i].cpu().numpy()), flush=True)

        # print('fb_mode: ', self.args.fb_mode)

        if self.args.fb_mode==0:
            # Connect hidden feature to the latent space
            latent_z, loss_kl = self.connect(pooled_hidden_fea)
            latent_z = latent_z.squeeze(1)  # shape = [batch_size, nz - nattr]

            # TODO(hzt): add attributes
            #latent_z_rev = self._add_attributes(latent_z, self._get_reverse_attributes(attributes, self.args.multi_attribute))
            #latent_z = self._add_attributes(latent_z, attributes)
            latent_z_rev = self._add_attributes(latent_z, 1 - self._get_sentiment_attributes(attributes, self.args.multi_attribute))
            latent_z = self._add_attributes(latent_z, self._get_sentiment_attributes(attributes, self.args.multi_attribute))

            # Decoding
            outputs = self.decoder(input_ids=labels, past=latent_z, labels=labels, label_ignore=self.pad_token_id)
            loss_rec = outputs[0]  # model outputs are always tuple in pytorch-transformers (see doc)

        elif self.args.fb_mode==1:
            # Connect hidden feature to the latent space
            mu, logvar = self.encoder.linear(pooled_hidden_fea).chunk(2, -1)

            mu, logvar = mu[:, :-self.nattr], logvar[:, :-self.nattr]  # TODO(hzt)

            latent_z = self.reparameterize(mu, logvar, nsamples=1)
            latent_z = latent_z.squeeze(1)

            #latent_z_rev = self._add_attributes(latent_z, self._get_reverse_attributes(attributes, self.args.multi_attribute))
            #latent_z = self._add_attributes(latent_z, attributes)  # TODO(hzt)
            latent_z_rev = self._add_attributes(latent_z, 1 - self._get_sentiment_attributes(attributes, self.args.multi_attribute))
            latent_z = self._add_attributes(latent_z, self._get_sentiment_attributes(attributes, self.args.multi_attribute))

            loss_kl = 0.5 * (mu.pow(2) + logvar.exp() - logvar - 1)
            kl_mask = (loss_kl > self.args.dim_target_kl).float()
            loss_kl = (kl_mask * loss_kl).sum(dim=1)

            # past = self.decoder.linear(latent_z)
            # Decoding
            outputs = self.decoder(input_ids=labels, past=latent_z, labels=labels, label_ignore=self.pad_token_id)
            loss_rec = outputs[0]  # model outputs are always tuple in pytorch-transformers (see doc)

        elif self.args.fb_mode==2:
            # Connect hidden feature to the latent space
            latent_z, loss_kl = self.connect_deterministic(pooled_hidden_fea)
            latent_z = latent_z.squeeze(1)

            #latent_z_rev = self._add_attributes(latent_z, self._get_reverse_attributes(attributes, self.args.multi_attribute))
            #latent_z = self._add_attributes(latent_z, attributes)
            latent_z_rev = self._add_attributes(latent_z, 1 - self._get_sentiment_attributes(attributes, self.args.multi_attribute))
            latent_z = self._add_attributes(latent_z, self._get_sentiment_attributes(attributes, self.args.multi_attribute))

            # past = self.decoder.linear(latent_z)
            # Decoding
            outputs = self.decoder(input_ids=labels, past=latent_z, labels=labels, label_ignore=self.pad_token_id)
            loss_rec = outputs[0]  # model outputs are always tuple in pytorch-transformers (see doc)


        z_part = self._get_z_part(latent_z)


        if train_a_layer:
            #loss_a, logits_a = self._a_predict(z_part.detach(), labels=attributes)
            loss_a, logits_a = self._a_predict(z_part.detach(), labels=self._get_sentiment_attributes(attributes, self.args.multi_attribute))
            return loss_a, logits_a


        if train_gan:
            batch_size = inputs.shape[0]
            ones_label = torch.ones([batch_size]).to(dtype=torch.float32, device=inputs.device)
            zeros_label = torch.zeros([batch_size]).to(dtype=torch.float32, device=inputs.device)

            # Generate z
            random_noise = torch.nn.init.normal_(torch.empty(inputs.size(0), self.nz - self.nattr)).to(device=inputs.device, dtype=torch.float32)
            gen_z = self.latent_generator(random_noise)  # (B, nz)

            prob_encode_z_dis = self.latent_discriminator(z_part).squeeze(1).float()  # (B)
            prob_gen_z_dis = self.latent_discriminator(gen_z).squeeze(1).float()  # (B)

            # Train latent discriminator
            loss_lsd = torch.nn.BCEWithLogitsLoss()(prob_gen_z_dis, zeros_label) + torch.nn.BCEWithLogitsLoss()(prob_encode_z_dis, ones_label)
            acc_encode_z_dis = ((prob_encode_z_dis >= 0).float() == ones_label).float().mean()
            acc_gen_z_dis = ((prob_gen_z_dis >= 0).float() == zeros_label).float().mean()
            # Train sampler adversarially
            loss_lsg = torch.nn.BCEWithLogitsLoss()(prob_gen_z_dis, ones_label)

            loss = loss_lsd + loss_lsg

            return loss, loss_lsd, loss_lsg, acc_encode_z_dis, acc_gen_z_dis


        # ===================================
        ## TODO(hzt): classifier loss
        # ===================================
        loss_clas = torch.zeros_like(loss_rec)

        #reconstrution_mask_clas = (labels[:,1:] != 50257).float()  # 50257 is the padding token for GPT2
        #sent_length_clas = torch.sum(reconstrution_mask_clas, dim=1)
        #max_length_clas = int(sent_length_clas.max().item())  # TODO(hzt)

        reconstrution_mask_clas = (labels[:,1:] != 50257).float() * (labels[:,1:] != 50259).float() # 50257 is the padding token for GPT2
        sent_length_clas = torch.sum(reconstrution_mask_clas, dim=1)
        max_length_clas = int(sent_length_clas.max().item()) + 1  # TODO(hzt): + 1 for shape matching

        if lambda_clas > 0:
            logits, gumbel_weights, sample_mask = self.gumbel_sequence_sample(
                latent_z_rev,
                temperature,
                max_length_clas,
                use_gumbel=use_gumbel,
                hard=hard,
                tokenizer=tokenizer_decoder)


            #sample_mask = reconstrution_mask  # TODO(hzt)
            sample_mask = reconstrution_mask_clas  # TODO(hzt)
            gumbel_weights = gumbel_weights[:,:-1]
            logits = logits[:,:-1]


            ### Debug ================

            #gumbel_weights_max, gumbel_weights_maxid = torch.max(gumbel_weights, dim=-1)

            #print('gumbel_weights_max: ')
            #print(gumbel_weights_max)
            #print('gumbel_weights_maxid: ')
            #print(gumbel_weights_maxid)
            #print('sample_mask: ', sample_mask.shape)
            #print(sample_mask)
            ##print('labels: ', labels.shape)
            ##print(labels)
            ##print('sent_length_clas: ')
            ##print(sent_length_clas)

            #for b in range(8):
            #    text_x1_data = tokenizer_decoder.decode(gumbel_weights_maxid[b].tolist(), clean_up_tokenization_spaces=False)
            #    text_x1_data = text_x1_data.split('<EOS>')[0].strip()
            #    print(text_x1_data)

            #print('-' * 40)
            #for b in range(8):
            #    text_x1_data = tokenizer_decoder.decode(labels[b].tolist(), clean_up_tokenization_spaces=False)
            #    text_x1_data = text_x1_data.split('<EOS>')[0].strip()
            #    print(text_x1_data)

            ##print('=' * 40)

            ###exit()

            ### =====================


            inputs_embeds = torch.matmul(gumbel_weights, self.classifier.transformer.wte.weight)
            inputs_embeds = inputs_embeds * sample_mask.unsqueeze(-1)

            outputs = self.classifier(
                #input_ids=gumbel_weights,  # TODO: for debugging
                #end_token_id_or_embeds=self.eos_token_id,
                input_ids=None,
                inputs_embeds=inputs_embeds,
                end_token_id_or_embeds=torch.zeros_like(inputs_embeds[0][-1]),
                labels=1-self._get_sentiment_attributes(attributes, self.args.multi_attribute),
                attention_mask=sample_mask)

            tmp_output, pooled_fea = outputs
            loss_clas, logits_clas = tmp_output


            #sample_length = torch.sum(sample_mask, dim=1)
            #print('sent length: %d; sample length: %d ##|' % (int(torch.mean(sent_length.float()).item()), int(torch.mean(sample_length.float()).item())))


            #print('sent_length: ', sent_length.shape)
            #for b in range(inputs.shape[0]):
            #    print(sent_length[b].item())
            #    #text_x0 = tokenizer_encoder.decode(inputs[b,:sent_length[b].item()].tolist(), clean_up_tokenization_spaces=True)[0]
            #    text_x0 = tokenizer_encoder.decode(inputs[b,:].tolist(), clean_up_tokenization_spaces=True)[0]

            #    text_x1 = tokenizer_decoder.decode(gumbel_weights[b,:].tolist(), clean_up_tokenization_spaces=True)
            #    #text_x1 = text_x1.split()[1:-1]
            #    #text_x1 = text_x1.split()[:-1]
            #    #text_x1 = ' '.join(text_x1)
            #    text_x1 = text_x1.split('<EOS>')[0].strip()
            #    print(text_x0  +  '\n' + text_x1)

            #preds = logits_clas.detach().cpu().numpy()
            #preds = np.argmax(preds, axis=1)
            #out_label_ids = attributes.detach().cpu().numpy()
            #print(out_label_ids)
            #print(preds)
            #print(torch.softmax(logits_clas, dim=-1).detach().cpu().numpy()[:,1])
            #result = (preds == out_label_ids).mean()
            #print(result)


        if lambda_reg_z > 0 or lambda_reg_z_c > 0:
            assert isinstance(self.encoder, GPT2Encoder)
            if cond_a:
                pooled_hidden_fea_rev = self.encoder(
                    input_ids=None,
                    inputs_embeds=self._append_a_embeds(inputs_embeds,
                                                        1-self._get_sentiment_attributes(attributes, self.args.multi_attribute),
                                                        self.encoder.transformer.wte),
                    end_token_id_or_embeds=torch.zeros_like(inputs_embeds[0][-1]))
            else:
                pooled_hidden_fea_rev = self.encoder(
                    input_ids=None,
                    inputs_embeds=inputs_embeds,
                    end_token_id_or_embeds=torch.zeros_like(inputs_embeds[0][-1]))
            latent_z_rev, _ = self.connect_deterministic(pooled_hidden_fea_rev)
            latent_z_rev = latent_z_rev.squeeze(1)

        # ===================================
        ## TODO(hzt): reg z loss
        # ===================================
        loss_reg_z = torch.zeros_like(loss_rec)
        if lambda_reg_z > 0:
            #z_rev_normalized = self._normalize_z(latent_z_rev)  # TODO(hzt): ?
            z_rev_normalized = latent_z_rev
            z_rev_normalized = torch.sigmoid(z_rev_normalized)

            z_part_normalized = self._normalize_z(z_part)

            loss_fn = torch.nn.BCELoss(reduction='mean')
            loss_reg_z = loss_fn(input=z_rev_normalized, target=z_part_normalized.detach())


        # ===================================
        ## TODO(hzt): c loss
        # ===================================
        loss_c = torch.zeros_like(loss_rec)
        if lambda_c_loss > 0:
            loss_c, _ = self._c_predict(z_part, labels=attributes_2)

        loss_reg_z_c = torch.zeros_like(loss_rec)
        if lambda_reg_z_c > 0:
            loss_reg_z_c, _ = self._c_predict(latent_z_rev, labels=attributes_2)

        if self.args.length_weighted_loss:
            loss_rec = loss_rec / sent_length
            loss = loss_rec + self.args.beta * loss_kl
        else:
            loss = loss_rec + self.args.beta * loss_kl

        loss = lambda_recon * loss + lambda_clas * loss_clas + lambda_reg_z * loss_reg_z + lambda_c_loss * loss_c + lambda_reg_z_c * loss_reg_z_c

        return loss_rec, loss_kl, loss_clas, loss_reg_z, loss_c, loss_reg_z_c, loss, max_length_clas


    def encoder_sample(self, bert_fea, nsamples):
        """sampling from the encoder
        Returns: Tensor1
            Tensor1: the tensor latent z with shape [batch, nsamples, nz]
        """

        # (batch_size, nz)

        mu, logvar = self.encoder.linear(bert_fea).chunk(2, -1)
        mu, logvar = mu.squeeze(0), logvar.squeeze(0)

        # (batch, nsamples, nz)
        z = self.reparameterize(mu, logvar, nsamples)

        return z, (mu, logvar)


    def encode_stats(self, x):
        """
        Returns: Tensor1, Tensor2
            Tensor1: the mean of latent z with shape [batch, nz]
            Tensor2: the logvar of latent z with shape [batch, nz]
        """

        return self.encoder.encode_stats(x)


    ### TODO(hzt): Gumbel Softmax decoding ###########################

    def sample_gumbel(self, shape, eps=1e-20):
        U = torch.rand(shape)
        return -torch.log(-torch.log(U + eps) + eps)

    def gumbel_softmax_sample(self, logits, temperature, use_gumbel=True):
        if use_gumbel:
            y = logits + self.sample_gumbel(logits.size()).to(logits.device)
        else:
            y = logits
        #return torch.softmax(y / temperature, dim=-1)  # TODO(hzt)
        return torch.softmax(y / temperature, dim=-1), torch.softmax(y / 0.00001, dim=-1)

    def gumbel_softmax(self, logits, temperature, hard=False, use_gumbel=True):
        """
        ST-gumple-softmax
        input: [*, n_class]
        return: flatten --> [*, n_class] an one-hot vector
        """
        y, yy = self.gumbel_softmax_sample(logits, temperature, use_gumbel=use_gumbel)

        if hard:  # TODO(hzt)
            y_argmax = torch.argmax(y, dim=-1)
            y_hard = torch.nn.functional.one_hot(y_argmax, num_classes=logits.shape[-1])
            y = y_hard - y.detach() + y

        #return y, yy  # TODO(hzt)

        #y = torch.nn.functional.gumbel_softmax(logits, tau=temperature, hard=hard)
        return y

        # if not hard:
        #     return y
        #     # return y.view(-1, latent_dim * categorical_dim)

        # shape = y.size()
        # _, ind = y.max(dim=-1)
        # y_hard = torch.zeros_like(y).view(-1, shape[-1])
        # y_hard.scatter_(1, ind.view(-1, 1), 1)
        # y_hard = y_hard.view(*shape)
        # # Set gradients w.r.t. y_hard gradients w.r.t. y
        # y_hard = (y_hard - y).detach() + y
        # return y_hard.view(-1, latent_dim * categorical_dim)


    def gumbel_sequence_sample(self, z, temperature, max_length, use_gumbel=True, hard=False, tokenizer=None):
        '''
        Returns:
            logits: the source logits of each token [B x seq_len x vsize]
            embeds: the representations of each token [B x  seq_len x hidden_dim]
        '''
        eos_id = self.eos_token_id
        cur_len = 0
        past = z
        input_ids = torch.ones(z.size(0), 1, device=z.device, dtype=torch.long) * self.bos_token_id
        input_emb = self.decoder.transformer.wte(input_ids)

        #sample_mask = torch.ones(input_ids.size(0), max_length, device=input_ids.device).type_as(input_ids)
        sample_mask = torch.ones(input_ids.size(0), max_length+1, device=input_ids.device).type_as(input_ids)
        gumbel_weights = []
        logits = []
        #argmax_id = []
        #gumbel_id = []

        while cur_len <= max_length:
            if input_emb is not None:
                #input_emb = input_emb.unsqueeze(1)  # TODO(hzt)
                #print('input_emb: ', input_emb.shape)
                outputs = self.decoder(input_ids=None, inputs_embeds=input_emb, past=past)
            else:
                outputs = self.decoder(input_ids, past=past)

            # TODO(hzt): for debugging
            #outputs = self.decoder(input_ids, past=past)
            #next_token_logits = outputs[0][:, -1, :]
            #next_token = torch.argmax(next_token_logits, dim=-1)
            #input_ids = torch.cat((input_ids, next_token.unsqueeze(1)), dim=1)
            #not_eos = 1 - (next_token == eos_id).type_as(sample_mask)
            #sample_mask[:,cur_len:] = sample_mask[:, cur_len:] * not_eos.unsqueeze(-1)

            next_token_logits = outputs[0][:, -1, :]
            #past = outputs[1]  # TODO(hzt)

            #g_weights, g_weights_hard = self.gumbel_softmax(next_token_logits, temperature, hard=hard, use_gumbel=use_gumbel)
            g_weights = self.gumbel_softmax(next_token_logits, temperature, hard=hard, use_gumbel=use_gumbel)

            #argmax_id.append(next_token_logits[0].argmax().item())
            #gumbel_id.append(g_weights[0].argmax().item())

            ## TODO(hzt) ====================
            #next_token = torch.argmax(next_token_logits, dim=-1)
            #input_ids = torch.cat((input_ids, next_token.unsqueeze(1)), dim=1)
            #input_emb_temp = self.decoder.transformer.wte(input_ids)
            #print('input_emb_temp')
            #print(input_emb_temp.shape)
            #print(input_emb_temp[0,:20,:5])


            input_emb_ = torch.matmul(g_weights, self.decoder.transformer.wte.weight)  # TODO(hzt)
            #input_emb_ = torch.matmul(g_weights_hard, self.decoder.transformer.wte.weight)

            input_emb_ = input_emb_.unsqueeze(1)
            input_emb = input_emb_ if input_emb is None else torch.cat((input_emb, input_emb_), dim=1)


            #print('input_emb')
            #print(input_emb.shape)
            #print(input_emb[0,:20,:5])


            # if the input_emb is <|endoftext|>
            eos_probs = g_weights[:,eos_id].detach()
            not_eos = (eos_probs < 0.5).type_as(sample_mask)
            #sample_mask[:,cur_len+1:] = sample_mask[:, cur_len+1:] * not_eos.unsqueeze(-1)  # TODO(hzt)
            sample_mask[:,cur_len:] = sample_mask[:, cur_len:] * not_eos.unsqueeze(-1)

            gumbel_weights.append(g_weights)
            logits.append(next_token_logits)
            cur_len += 1

        # logits = logits[1:]     # remove the fist logits for <|endoftext|>  # TODO(hzt)
        # gumbel_weights = gumbel_weights[:-1]

        logits = torch.stack(logits, 1)
        gumbel_weights = torch.stack(gumbel_weights, 1)

        # TODO(hzt): for debugging
        # gumbel_weights = input_ids[:, 1:]

        #print('logits: ', logits.shape)
        #print('gumbel_weights: ', gumbel_weights.shape)
        #print('max_length: ', max_length)
        #print(sample_mask.shape)
        #print(sample_mask)
        #print('argmax_id: ', argmax_id)
        #if tokenizer is not None:
        #    print(tokenizer.decode(argmax_id))
        #exit()

        assert logits.size(1) == max_length + 1

        return (logits, gumbel_weights, sample_mask)


    # #####################################################


    def decode(self, z, strategy, K=10):
        """generate samples from z given strategy
        Args:
            z: [batch, nsamples, nz]
            strategy: "beam" or "greedy" or "sample"
            K: the beam width parameter
        Returns: List1
            List1: a list of decoded word sequence
        """

        if strategy == "beam":
            return self.decoder.beam_search_decode(z, K)
        elif strategy == "greedy":
            return self.decoder.greedy_decode(z)
        elif strategy == "sample":
            return self.decoder.sample_decode(z)
        else:
            raise ValueError("the decoding strategy is not supported")


    def reconstruct(self, x, decoding_strategy="greedy", K=5):
        """reconstruct from input x
        Args:
            x: (batch, *)
            decoding_strategy: "beam" or "greedy" or "sample"
            K: the beam width parameter
        Returns: List1
            List1: a list of decoded word sequence
        """

        raise NotImplementedError  # TODO(hzt)

        z = self.sample_from_inference(x).squeeze(1)

        return self.decode(z, decoding_strategy, K)

    def log_probability(self, x, z):
        """Cross Entropy in the language case
        Args:
            x: (batch_size, seq_len)
            z: (batch_size, n_sample, nz)
        Returns:
            log_p: (batch_size, n_sample).
                log_p(x|z) across different x and z
        """
        outputs = self.decoder(input_ids=x, past=z, labels=x, label_ignore=self.pad_token_id)
        loss_rec = outputs[0]
        return -loss_rec



    def loss_iw(self, x0, x1, attributes, nsamples=50, ns=1):
        """
        Args:
            x: if the data is constant-length, x is the data tensor with
                shape (batch, *). Otherwise x is a tuple that contains
                the data tensor and length list
        Returns: Tensor1, Tensor2, Tensor3
            Tensor1: total loss [batch]
            Tensor2: reconstruction loss shape [batch]
            Tensor3: KL loss shape [batch]
        """

        # encoding into bert features
        if isinstance(self.encoder, BertForLatentConnector):
            bert_fea = self.encoder(x0)[1]
        elif isinstance(self.encoder, GPT2Encoder):
            bert_fea = self.encoder(x0, end_token_id_or_embeds=50259)
        else:
            raise NotImplementedError

        # (batch_size, nz)

        mu, logvar = self.encoder.linear(bert_fea).chunk(2, -1)

        mu, logvar = mu[:, :-self.nattr], logvar[:, :-self.nattr]  # TODO(hzt)

        ##################
        # compute KL
        ##################
        KL = 0.5 * (mu.pow(2) + logvar.exp() - logvar - 1).sum(dim=1)

        # mu, logvar = mu.squeeze(0), logvar.squeeze(0)
        ll_tmp, rc_tmp = [], []
        for _ in range(int(nsamples / ns)):

            # (batch, nsamples, nz)
            z = self.reparameterize(mu, logvar, ns)

            z_full = self._add_attributes(z, self._get_sentiment_attributes(attributes, self.args.multi_attribute))  # TODO(hzt)

            # past = self.decoder.linear(z)
            past = z_full

            # [batch, nsamples]
            log_prior = self.eval_prior_dist(z)
            log_gen = self.eval_cond_ll(x1, past)
            log_infer = self.eval_inference_dist(z, (mu, logvar))

            log_gen = log_gen.unsqueeze(0).contiguous().view(z.shape[0],-1)


            rc_tmp.append(log_gen)
            ll_tmp.append(log_gen + log_prior - log_infer)



        log_prob_iw = log_sum_exp(torch.cat(ll_tmp, dim=-1), dim=-1) - math.log(nsamples)
        log_gen_iw = torch.mean(torch.cat(rc_tmp, dim=-1), dim=-1)

        return log_prob_iw, log_gen_iw , KL


    def nll_iw(self, x0, x1, nsamples, ns=1):
        """compute the importance weighting estimate of the log-likelihood
        Args:
            x0, x1:  two different tokenization results of x, where x is the data tensor with shape (batch, *).
            nsamples: Int
                the number of samples required to estimate marginal data likelihood
        Returns: Tensor1
            Tensor1: the estimate of log p(x), shape [batch]
        """

        # compute iw every ns samples to address the memory issue
        # nsamples = 500, ns = 100
        # nsamples = 500, ns = 10

        # TODO: note that x is forwarded twice in self.encoder.sample(x, ns) and self.eval_inference_dist(x, z, param)
        #.      this problem is to be solved in order to speed up

        tmp = []
        for _ in range(int(nsamples / ns)):
            # [batch, ns, nz]

            # Chunyuan:
            # encoding into bert features
            if isinstance(self.encoder, BertForLatentConnector):
                pooled_hidden_fea = self.encoder(x0)[1]
            elif isinstance(self.encoder, GPT2Encoder):
                pooled_hidden_fea = self.encoder(x0, end_token_id_or_embeds=50259)
            else:
                raise NotImplementedError

            # param is the parameters required to evaluate q(z|x)
            z, param = self.encoder_sample(pooled_hidden_fea, ns)

            # [batch, ns]
            log_comp_ll = self.eval_complete_ll(x1, z)
            log_infer_ll = self.eval_inference_dist(z, param)

            tmp.append(log_comp_ll - log_infer_ll)

        ll_iw = log_sum_exp(torch.cat(tmp, dim=-1), dim=-1) - math.log(nsamples)

        return ll_iw

    def KL(self, x):
        _, KL = self.encode(x, 1)

        return KL

    def eval_prior_dist(self, zrange):
        """perform grid search to calculate the true posterior
        Args:
            zrange: tensor
                different z points that will be evaluated, with
                shape (k^2, nz), where k=(zmax - zmin)/space
        """

        # (k^2)
        return self.prior.log_prob(zrange).sum(dim=-1)

    def eval_complete_ll(self, x, z):
        """compute log p(z,x)
        Args:
            x: Tensor
                input with shape [batch, seq_len]
            z: Tensor
                evaluation points with shape [batch, nsamples, nz]
        Returns: Tensor1
            Tensor1: log p(z,x) Tensor with shape [batch, nsamples]
        """

        # [batch, nsamples]
        log_prior = self.eval_prior_dist(z)
        log_gen = self.eval_cond_ll(x, z)

        return log_prior + log_gen



    def eval_cond_ll(self, x, z):
        """compute log p(x|z)
        """
        x_shape = list(x.size())
        z_shape = list(z.size())
        if len(z_shape) == 3:
            x = x.unsqueeze(1).repeat(1, z_shape[1], 1).contiguous().view(x_shape[0]*z_shape[1], x_shape[-1])
            z = z.contiguous().view(x_shape[0]*z_shape[1], z_shape[-1])

        return self.log_probability(x, z)



    def eval_log_model_posterior(self, x, grid_z):
        """perform grid search to calculate the true posterior
         this function computes p(z|x)
        Args:
            grid_z: tensor
                different z points that will be evaluated, with
                shape (k^2, nz), where k=(zmax - zmin)/pace
        Returns: Tensor
            Tensor: the log posterior distribution log p(z|x) with
                    shape [batch_size, K^2]
        """
        try:
            batch_size = x.size(0)
        except:
            batch_size = x[0].size(0)

        # (batch_size, k^2, nz)
        grid_z = grid_z.unsqueeze(0).expand(batch_size, *grid_z.size()).contiguous()

        # (batch_size, k^2)
        log_comp = self.eval_complete_ll(x, grid_z)

        # normalize to posterior
        log_posterior = log_comp - log_sum_exp(log_comp, dim=1, keepdim=True)

        return log_posterior

    def sample_from_inference(self, x, nsamples=1):
        """perform sampling from inference net
        Returns: Tensor
            Tensor: samples from infernece nets with
                shape (batch_size, nsamples, nz)
        """
        z, _ = self.encoder.sample(x, nsamples)

        return z


    def sample_from_posterior(self, x, nsamples):
        """perform MH sampling from model posterior
        Returns: Tensor
            Tensor: samples from model posterior with
                shape (batch_size, nsamples, nz)
        """

        # use the samples from inference net as initial points
        # for MCMC sampling. [batch_size, nsamples, nz]
        cur = self.encoder.sample_from_inference(x, 1)
        cur_ll = self.eval_complete_ll(x, cur)
        total_iter = self.args.mh_burn_in + nsamples * self.args.mh_thin
        samples = []
        for iter_ in range(total_iter):
            next = torch.normal(mean=cur,
                std=cur.new_full(size=cur.size(), fill_value=self.args.mh_std))
            # [batch_size, 1]
            next_ll = self.eval_complete_ll(x, next)
            ratio = next_ll - cur_ll

            accept_prob = torch.min(ratio.exp(), ratio.new_ones(ratio.size()))

            uniform_t = accept_prob.new_empty(accept_prob.size()).uniform_()

            # [batch_size, 1]
            mask = (uniform_t < accept_prob).float()
            mask_ = mask.unsqueeze(2)

            cur = mask_ * next + (1 - mask_) * cur
            cur_ll = mask * next_ll + (1 - mask) * cur_ll

            if iter_ >= self.args.mh_burn_in and (iter_ - self.args.mh_burn_in) % self.args.mh_thin == 0:
                samples.append(cur.unsqueeze(1))

        return torch.cat(samples, dim=1)


    def calc_model_posterior_mean(self, x, grid_z):
        """compute the mean value of model posterior, i.e. E_{z ~ p(z|x)}[z]
        Args:
            grid_z: different z points that will be evaluated, with
                    shape (k^2, nz), where k=(zmax - zmin)/pace
            x: [batch, *]
        Returns: Tensor1
            Tensor1: the mean value tensor with shape [batch, nz]
        """

        # [batch, K^2]
        log_posterior = self.eval_log_model_posterior(x, grid_z)
        posterior = log_posterior.exp()

        # [batch, nz]
        return torch.mul(posterior.unsqueeze(2), grid_z.unsqueeze(0)).sum(1)

    def calc_infer_mean(self, x):
        """
        Returns: Tensor1
            Tensor1: the mean of inference distribution, with shape [batch, nz]
        """

        raise NotImplementedError  # TODO(hzt)

        mean, logvar = self.encoder.forward(x)

        return mean




    def eval_inference_dist(self, z, param):
        """this function computes log q(z | x)
        Args:
            z: tensor
                different z points that will be evaluated, with
                shape [batch, nsamples, nz]
        Returns: Tensor1
            Tensor1: log q(z|x) with shape [batch, nsamples]
        """

        nz = z.size(2)
        mu, logvar = param

        # (batch_size, 1, nz)
        mu, logvar = mu.unsqueeze(1), logvar.unsqueeze(1)
        var = logvar.exp()

        # (batch_size, nsamples, nz)
        dev = z - mu

        # (batch_size, nsamples)
        log_density = -0.5 * ((dev ** 2) / var).sum(dim=-1) - \
            0.5 * (nz * math.log(2 * math.pi) + logvar.sum(-1))

        return log_density



    def calc_mi(self, test_data_batch, args):

        raise NotImplementedError  # TODO(hzt)

        # calc_mi_v3
        import math
        from modules.utils import log_sum_exp

        mi = 0
        num_examples = 0

        mu_batch_list, logvar_batch_list = [], []
        neg_entropy = 0.
        for batch_data in test_data_batch:

            x0, _, _ = batch_data
            x0 = x0.to(args.device)

            # encoding into bert features
            if isinstance(self.encoder, BertForLatentConnector):
                bert_fea = self.encoder(x0)[1]
            elif isinstance(self.encoder, GPT2Encoder):
                bert_fea = self.encoder(x0, end_token_id_or_embeds=50259)
            else:
                raise NotImplementedError

            (batch_size, nz)
            mu, logvar = self.encoder.linear(bert_fea).chunk(2, -1)

            x_batch, nz = mu.size()

            #print(x_batch, end=' ')

            num_examples += x_batch

            # E_{q(z|x)}log(q(z|x)) = -0.5*nz*log(2*\pi) - 0.5*(1+logvar).sum(-1)

            neg_entropy += (-0.5 * nz * math.log(2 * math.pi)- 0.5 * (1 + logvar).sum(-1)).sum().item()
            mu_batch_list += [mu.cpu()]
            logvar_batch_list += [logvar.cpu()]


        neg_entropy = neg_entropy / num_examples
        ##print()

        num_examples = 0
        log_qz = 0.
        for i in range(len(mu_batch_list)):
            ###############
            # get z_samples
            ###############
            mu, logvar = mu_batch_list[i].cuda(), logvar_batch_list[i].cuda()

            # [z_batch, 1, nz]

            z_samples = self.reparameterize(mu, logvar, 1)

            z_samples = z_samples.view(-1, 1, nz)
            num_examples += z_samples.size(0)

            ###############
            # compute density
            ###############
            # [1, x_batch, nz]
            #mu, logvar = mu_batch_list[i].cuda(), logvar_batch_list[i].cuda()
            #indices = list(np.random.choice(np.arange(len(mu_batch_list)), 10)) + [i]
            indices = np.arange(len(mu_batch_list))
            mu = torch.cat([mu_batch_list[_] for _ in indices], dim=0).cuda()
            logvar = torch.cat([logvar_batch_list[_] for _ in indices], dim=0).cuda()
            x_batch, nz = mu.size()

            mu, logvar = mu.unsqueeze(0), logvar.unsqueeze(0)
            var = logvar.exp()

            # (z_batch, x_batch, nz)
            dev = z_samples - mu

            # (z_batch, x_batch)
            log_density = -0.5 * ((dev ** 2) / var).sum(dim=-1) - \
                0.5 * (nz * math.log(2 * math.pi) + logvar.sum(-1))

            # log q(z): aggregate posterior
            # [z_batch]
            log_qz += (log_sum_exp(log_density, dim=1) - math.log(x_batch)).sum(-1)

        log_qz /= num_examples
        mi = neg_entropy - log_qz

        return mi



    def calc_au(self, eval_dataloader, args, delta=0.01):
        """compute the number of active units
        """

        raise NotImplementedError  # TODO(hzt)

        cnt = 0
        for batch_data in eval_dataloader:

            x0, _, _ = batch_data
            x0 = x0.to(args.device)

            # encoding into bert features
            if isinstance(self.encoder, BertForLatentConnector):
                bert_fea = self.encoder(x0)[1]
            elif isinstance(self.encoder, GPT2Encoder):
                bert_fea = self.encoder(x0, end_token_id_or_embeds=50259)
            else:
                raise NotImplementedError

            # (batch_size, nz)
            mean, logvar = self.encoder.linear(bert_fea).chunk(2, -1)

            if cnt == 0:
                means_sum = mean.sum(dim=0, keepdim=True)
            else:
                means_sum = means_sum + mean.sum(dim=0, keepdim=True)
            cnt += mean.size(0)

        # (1, nz)
        mean_mean = means_sum / cnt

        cnt = 0
        for batch_data in eval_dataloader:

            x0, _, _ = batch_data
            x0 = x0.to(args.device)

            # encoding into bert features
            if isinstance(self.encoder, BertForLatentConnector):
                bert_fea = self.encoder(x0)[1]
            elif isinstance(self.encoder, GPT2Encoder):
                bert_fea = self.encoder(x0, end_token_id_or_embeds=50259)
            else:
                raise NotImplementedError

            # (batch_size, nz)
            mean, _ = self.encoder.linear(bert_fea).chunk(2, -1)

            if cnt == 0:
                var_sum = ((mean - mean_mean) ** 2).sum(dim=0)
            else:
                var_sum = var_sum + ((mean - mean_mean) ** 2).sum(dim=0)
            cnt += mean.size(0)

        # (nz)
        au_var = var_sum / (cnt - 1)

        return (au_var >= delta).sum().item(), au_var
