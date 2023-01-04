import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.decoder.top_k_top_p_filtering import top_k_top_p_filtering


class CARA(nn.Module):
    def __init__(
        self,
        encoder,
        decoder,
        tokenizer_encoder,
        tokenizer_decoder,
        latent_size,
        device,
        label_size=2,
        block_size=64,
        beta_cls=1.0,
        soft_temperature=0.5,
        temperature=1.0,
        top_k=5,
        top_p=0,
        latent_scale_factor=1.0,
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.tokenizer_encoder = tokenizer_encoder
        self.tokenizer_decoder = tokenizer_decoder

        self.nz = latent_size
        self.label_size = label_size
        self.block_size = block_size
        self.beta_cls = beta_cls
        self.soft_temperature = soft_temperature
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.latent_scale_factor = latent_scale_factor

        self.device = device

        self.bos_token_id_list = self.tokenizer_decoder.encode(
            self.tokenizer_decoder.bos_token
        )
        self.pad_token_id = self.tokenizer_decoder.pad_token_id

        # connector: from Bert hidden units to the latent space
        self.linear = nn.Linear(
            encoder.config.hidden_size,
            self.nz,
            bias=False,
        )

        # use the same size as latent_z so as to use the same decoder.linear()
        self.label_embedding = nn.Embedding(
            self.label_size, self.nz, padding_idx=0
        )
        self.latent_generator = nn.Linear(self.nz, self.nz)
        self.latent_classifier = nn.Linear(
            self.nz, label_size if label_size > 2 else 1
        )
        self.latent_discriminator = nn.Linear(self.nz, 1)

        self.gpt_embeddings = nn.Embedding(
            self.decoder.config.vocab_size, self.decoder.config.n_embd
        )
        self.gpt_embeddings.weight.data = decoder.transformer.wte.weight.data

        self.conv1 = nn.Conv1d(
            self.encoder.config.hidden_size, self.encoder.config.hidden_size, 3
        )
        self.classifier = nn.Linear(
            self.encoder.config.hidden_size,
            1 if label_size <= 2 else label_size,
        )

        self.CrossEntropyLoss = torch.nn.CrossEntropyLoss()
        self.BCEWithLogitsLoss = torch.nn.BCEWithLogitsLoss()

    def forward(self, input_seq_ids, tgt_seq_ids, cond_labels, attention_mask):
        """
        Inputs:
            inputs: (B, seq_len)
            labels: (B, seq_len)
            cond_labels: (B), conditional labels.
        """
        ones_label = torch.ones_like(cond_labels).to(dtype=torch.float32)
        zeros_label = torch.zeros_like(cond_labels).to(dtype=torch.float32)
        random_noise = torch.nn.init.normal_(
            torch.empty(input_seq_ids.size(0), self.nz)
        ).to(device=input_seq_ids.device, dtype=torch.float32)

        # Encode inputs
        outputs = self.encoder(input_seq_ids, attention_mask=attention_mask)
        pooled_hidden_fea = outputs[1]  # (B, dim_h)

        # Encode z
        latent_z = self.linear(pooled_hidden_fea)  # (B, nz)

        # Generate z
        gen_z = self.latent_generator(random_noise)  # (B, nz)

        # ----- Latent discriminator for sampling from a simple distribution
        prob_encode_z_dis = (
            self.latent_discriminator(latent_z).squeeze(1).float()
        )  # (B)
        prob_gen_z_dis = (
            self.latent_discriminator(gen_z).squeeze(1).float()
        )  # (B)
        # Train latent discriminator
        loss_lsd = self.BCEWithLogitsLoss(
            prob_gen_z_dis, zeros_label
        ) + self.BCEWithLogitsLoss(prob_encode_z_dis, ones_label)
        acc_encode_z_dis = (
            (prob_encode_z_dis >= 0).float() == ones_label
        ).float()
        acc_gen_z_dis = ((prob_gen_z_dis >= 0).float() == zeros_label).float()
        # Train sampler adversarially
        loss_lsg = self.BCEWithLogitsLoss(prob_gen_z_dis, ones_label)

        # ----- Latent classifier for disentanglement
        prob_encode_z_cls = self.latent_classifier(latent_z)  # (B, n_labels)
        if self.label_size <= 2:
            prob_encode_z_cls = prob_encode_z_cls.squeeze(1)  # (B)
            # Train latent classifier
            loss_lsc = self.BCEWithLogitsLoss(
                prob_encode_z_cls, cond_labels.float()
            )
            acc_encode_z_cls = (
                (prob_encode_z_cls >= 0).float() == cond_labels.float()
            ).float()
            # Train encoder adversarially
            loss_encoder = 1 - self.BCEWithLogitsLoss(
                prob_encode_z_cls, cond_labels.float()
            )
        else:
            # Train latent classifier
            loss_lsc = self.CrossEntropyLoss(prob_encode_z_cls, cond_labels)
            acc_encode_z_cls = (
                torch.argmax(prob_encode_z_cls, dim=-1) == cond_labels
            ).float()
            # Train encoder adversarially
            loss_encoder = 1 - self.CrossEntropyLoss(
                prob_encode_z_cls, cond_labels
            )

        # ----- Recontruction loss with latent z and label emb
        # Embed labels
        label_emb = self.label_embedding(cond_labels)  # (B, hidden_size)
        if self.label_size <= 2:
            sampled_cond_labels = 1 - cond_labels
        else:
            # TODO: Implemented for multi-class labels.
            raise NotImplementedError
        # (B, hidden_size)
        sampled_label_emb = self.label_embedding(sampled_cond_labels)
        past_sampled_label = sampled_label_emb

        # Generate based on encoded z and gt labels. (reconstruction)
        past_z = latent_z
        gen_past_z = gen_z  # (B, n_blocks * hidden_size)

        past = latent_z + label_emb  # (B, n_blocks * hidden_size)

        outputs = self.decoder(
            input_ids=tgt_seq_ids,
            past=past,
            labels=tgt_seq_ids,
            label_ignore=self.pad_token_id,
        )
        loss_rec = outputs[0]

        # ----- Train a classifier in the observation space
        tgt_emb = self.gpt_embeddings(tgt_seq_ids)
        # (B, dim_h, seq_len)
        tgt_encode = self.conv1(tgt_emb.transpose(1, 2))
        tgt_encode = torch.mean(tgt_encode, dim=-1)  # (B, dim_h)
        prob_cls = self.classifier(tgt_encode)  # (B, n_labels)
        if self.label_size <= 2:
            prob_cls = prob_cls.squeeze(1)
            loss_cls = self.BCEWithLogitsLoss(prob_cls, cond_labels.float())
            pred_cls = (prob_cls >= 0).to(dtype=torch.long)
        else:
            loss_cls = self.CrossEntropyLoss(prob_cls, cond_labels)
            pred_cls = torch.argmax(prob_cls, dim=-1)
        acc_cls = (pred_cls == cond_labels).float()

        # Loss
        loss_latent_space = (
            (loss_encoder + loss_lsc)
            + (loss_lsd + loss_lsg)
            + self.beta_cls * loss_cls
        )
        loss = loss_rec + self.latent_scale_factor * loss_latent_space

        if not self.training:
            # Generate based on encoded z and gt labels
            generated = self.sample_sequence_conditional_batch(
                past=past, context=self.bos_token_id_list
            )

            # Attribute Transfer
            # Generate based on encoded z and sampled labels
            # (B, n_blocks * hidden_size)
            at_past = past_z + past_sampled_label
            # (B, seq_len)
            at_generated = self.sample_sequence_conditional_batch(
                past=at_past, context=self.bos_token_id_list
            )

            # Conditional Generation
            # Generate based on sampled z and sampled labels.
            # (B, n_blocks * hidden_size)
            cg_past = gen_past_z + past_sampled_label
            cg_generated = self.sample_sequence_conditional_batch(
                past=cg_past, context=self.bos_token_id_list
            )  # (B, seq_len)

            # classifier on gt generated sentences.
            ge_emb = self.gpt_embeddings(generated)
            # (B, dim_h, seq_len)
            ge_encode = self.conv1(ge_emb.transpose(1, 2))
            ge_encode = torch.mean(ge_encode, dim=-1)  # (B, dim_h)
            prob_ge_cls = self.classifier(ge_encode)  # (B, 1)

            if self.label_size <= 2:
                pred_ge_cls = (prob_ge_cls.squeeze(1) >= 0).to(torch.long)
            else:
                pred_ge_cls = torch.argmax(prob_ge_cls, dim=-1)
            acc_ge_cls = (pred_ge_cls == cond_labels).float()

            # classifier on attribute transfer generated sentences.
            at_emb = self.gpt_embeddings(at_generated)
            # (B, dim_h, seq_len)
            at_encode = self.conv1(at_emb.transpose(1, 2))
            at_encode = torch.mean(at_encode, dim=-1)  # (B, dim_h)
            prob_at_cls = self.classifier(at_encode)  # (B, 1)
            if self.label_size <= 2:
                pred_at_cls = (prob_at_cls.squeeze(1) >= 0).to(torch.long)
            else:
                pred_at_cls = torch.argmax(prob_at_cls, dim=-1)
            acc_at_cls = (pred_at_cls == sampled_cond_labels).float()

            # classifier on conditional generated sentences.
            cg_emb = self.gpt_embeddings(cg_generated)
            # (B, dim_h, seq_len)
            cg_encode = self.conv1(cg_emb.transpose(1, 2))
            cg_encode = torch.mean(cg_encode, dim=-1)  # (B, dim_h)
            prob_cg_cls = self.classifier(cg_encode)  # (B, 1)
            if self.label_size <= 2:
                pred_cg_cls = (prob_cg_cls.squeeze(1) >= 0).to(torch.long)
            else:
                pred_cg_cls = torch.argmax(prob_cg_cls, dim=-1)
            acc_cg_cls = (pred_cg_cls == sampled_cond_labels).float()

            result = {
                "sampled_cond_labels": sampled_cond_labels,
                "cond_labels": cond_labels,
                "tgt_seq_ids": tgt_seq_ids,
                "generated": generated,
                "at_generated": at_generated,
                "cg_generated": cg_generated,
                "acc_encode_z_dis": acc_encode_z_dis,
                "acc_gen_z_dis": acc_gen_z_dis,
                "acc_encode_z_cls": acc_encode_z_cls,
                "acc_cls": acc_cls,
                "acc_ge_cls": acc_ge_cls,
                "acc_at_cls": acc_at_cls,
                "acc_cg_cls": acc_cg_cls,
                "pred_cls": pred_cls,
                "pred_ge_cls": pred_ge_cls,
                "pred_at_cls": pred_at_cls,
                "pred_cg_cls": pred_cg_cls,
            }

            return result

        loss_dict = {
            "loss": loss,
            "loss_rec": loss_rec,
            "loss_encoder": loss_encoder,
            "loss_lsc": loss_lsc,
            "loss_lsd": loss_lsd,
            "loss_lsg": loss_lsg,
            "loss_cls": loss_cls,
        }
        acc_dict = {
            "acc_encode_z_dis": acc_encode_z_dis,
            "acc_gen_z_dis": acc_gen_z_dis,
            "acc_encode_z_cls": acc_encode_z_cls,
            "acc_cls": acc_cls,
        }
        return loss_dict, acc_dict

    def sample_sequence_conditional_batch(self, past, context):
        # context: a single id of <BOS>
        # past: (B, past_seq_len dim_h)
        num_samples = past.size(0)
        context = torch.tensor(context, dtype=torch.long, device=past.device)
        context = context.unsqueeze(0).repeat(num_samples, 1)
        generated = context  # (B, 1)

        while generated.size(-1) < self.block_size:
            inputs = {"input_ids": generated, "past": past}
            outputs = self.decoder(**inputs)
            lm_logits = outputs[0]

            # softmax sample
            # (B, 1, vocab_size)
            next_tokens_logits = lm_logits[:, -1, :] / self.temperature
            # TODO: Make this function batch compatible
            filtered_logits = top_k_top_p_filtering(
                next_tokens_logits, top_k=self.top_k, top_p=self.top_p
            )  # (B, 1, vocab_size)
            filtered_logits = F.softmax(filtered_logits, dim=-1)
            # (B, 1)
            next_tokens = torch.multinomial(filtered_logits, num_samples=1)
            # (B, seq_len+1)
            generated = torch.cat((generated, next_tokens), dim=1)

            not_finished = (
                next_tokens != self.tokenizer_decoder.encode("<EOS>")[0]
            )
            if torch.sum(not_finished) == 0:
                break

        return generated  # (B, seq_len)
