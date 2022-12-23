from transformers import BertTokenizer, GPT2Tokenizer, PreTrainedTokenizer


def bert_pretrained_tokeniser() -> PreTrainedTokenizer:
    return BertTokenizer.from_pretrained("bert-base-cased")


def gpt2_pretrained_tokeniser() -> PreTrainedTokenizer:
    tokeniser_decoder = GPT2Tokenizer.from_pretrained("gpt2")
    tokeniser_decoder.add_special_tokens(
        {"pad_token": "<PAD>", "bos_token": "<BOS>", "eos_token": "<EOS>"}
    )
    return tokeniser_decoder
