import torch
from torch.utils.data import Dataset

class BilingualDataset(Dataset):

    def __init__(self, dataset, source_tokenizer, target_tokenizer, source_language, target_language, sequence_length):
        super().__init__()
        self.sequence_length = sequence_length

        self.dataset = dataset
        self.source_tokenizer = source_tokenizer
        self.target_tokenizer = target_tokenizer
        self.source_language = source_language
        self.target_language = target_language

        self.sos_token = torch.tensor([target_tokenizer.token_to_id("[SOS]")], dtype=torch.int64)
        self.eos_token = torch.tensor([target_tokenizer.token_to_id("[EOS]")], dtype=torch.int64)
        self.pad_token = torch.tensor([target_tokenizer.token_to_id("[PAD]")], dtype=torch.int64)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        src_target_pair = self.dataset[idx]
        src_text = src_target_pair['translation'][self.source_language]
        tgt_text = src_target_pair['translation'][self.target_language]

        # Transform the text into tokens
        enc_input_tokens = self.source_tokenizer.encode(src_text).ids
        dec_input_tokens = self.target_tokenizer.encode(tgt_text).ids

        # Add sos, eos and padding to each sentence
        enc_num_padding_tokens = self.sequence_length - len(enc_input_tokens) - 2  # We will add <s> and </s>
        # We will only add <s>, and </s> only on the label
        dec_num_padding_tokens = self.sequence_length - len(dec_input_tokens) - 1

        # Make sure the number of padding tokens is not negative. If it is, the sentence is too long
        if enc_num_padding_tokens < 0 or dec_num_padding_tokens < 0:
            raise ValueError("Sentence is too long")

        # Add <s> and </s> token
        encoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(enc_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * enc_num_padding_tokens, dtype=torch.int64),
            ],
            dim=0,
        )

        # Add only <s> token
        decoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64),
            ],
            dim=0,
        )

        # Add only </s> token
        label = torch.cat(
            [
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64),
            ],
            dim=0,
        )

        # Double check the size of the tensors to make sure they are all sequence_length long
        assert encoder_input.size(0) == self.sequence_length
        assert decoder_input.size(0) == self.sequence_length
        assert label.size(0) == self.sequence_length

        return {
            "encoder_input": encoder_input,  # (sequence_length)
            "decoder_input": decoder_input,  # (sequence_length)
            "encoder_mask": (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(), # (1, 1, sequence_length)
            "decoder_mask": (decoder_input != self.pad_token).unsqueeze(0).int() & causal_mask(decoder_input.size(0)), # (1, sequence_length) & (1, sequence_length, sequence_length),
            "label": label,  # (sequence_length)
            "src_text": src_text,
            "tgt_text": tgt_text,
        }
    
def causal_mask(size):
    mask = torch.triu(torch.ones((1, size, size)), diagonal=1).type(torch.int)
    return mask == 0