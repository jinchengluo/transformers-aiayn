from constants import *

from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader, random_split

# Huggingface datasets and tokenizers
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

class EnglishToFrenchDataset(Dataset):

    def __init__(self, dataset, src_tokenizer, trg_tokenizer, src_language, trg_language, sequence_length):
        super().__init__()
        self.sequence_length = sequence_length

        self.dataset = dataset
        self.src_tokenizer = src_tokenizer
        self.trg_tokenizer = trg_tokenizer
        self.src_language = src_language
        self.trg_language = trg_language

        self.sos_token = torch.tensor([trg_tokenizer.token_to_id(SOS_TOKEN)], dtype=torch.int64)
        self.eos_token = torch.tensor([trg_tokenizer.token_to_id(EOS_TOKEN)], dtype=torch.int64)
        self.pad_token = torch.tensor([trg_tokenizer.token_to_id(PAD_TOKEN)], dtype=torch.int64)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        src_target_pair = self.dataset[idx]
        src_text = src_target_pair['translation'][self.src_language]
        trg_text = src_target_pair['translation'][self.trg_language]

        # Transform the text into tokens
        enc_input_tokens = self.src_tokenizer.encode(src_text).ids 
        dec_input_tokens = self.trg_tokenizer.encode(trg_text).ids

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
            "trg_text": trg_text,
        }
    
def causal_mask(size):
    mask = torch.triu(torch.ones((1, size, size)), diagonal=1).type(torch.int)
    return mask == 0


def get_all_sentences(dataset, language):
    for item in dataset:
        yield item["translation"][language]


def get_tokenizer(dataset, language):
    tokenizer_path = Path("tokenizer_{0}.json".format(language))
    if not Path.exists(tokenizer_path):
        # Most code taken from: https://huggingface.co/docs/tokenizers/quicktour
        tokenizer = Tokenizer(WordLevel(unk_token=UNK_TOKEN))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens=[UNK_TOKEN, PAD_TOKEN, SOS_TOKEN, EOS_TOKEN], min_frequency=2)
        tokenizer.train_from_iterator(get_all_sentences(dataset, language), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))

    return tokenizer

def get_dataset():
    dataset = load_dataset(DATASET_NAME, SOURCE_LANGUAGE + "-" + TARGET_LANGUAGE, split="train")

    src_tokenizer = get_tokenizer(dataset, SOURCE_LANGUAGE)
    trg_tokenizer = get_tokenizer(dataset, TARGET_LANGUAGE)

    train_size = int(0.8 * len(dataset))
    validation_size = len(dataset) - train_size

    train_dataset, valisation_dataset = random_split(dataset, [train_size, validation_size])

    train_dataset = EnglishToFrenchDataset(train_dataset, src_tokenizer, trg_tokenizer, SOURCE_LANGUAGE, TARGET_LANGUAGE, SEQUENCE_LENGTH)
    validation_dataset = EnglishToFrenchDataset(valisation_dataset, src_tokenizer, trg_tokenizer, SOURCE_LANGUAGE, TARGET_LANGUAGE, SEQUENCE_LENGTH)

    # Find the maximum length of each sentence in the source and target sentence
    src_max_length = 0
    trg_max_length = 0

    for item in dataset:
        src_ids = src_tokenizer.encode(item['translation'][SOURCE_LANGUAGE]).ids
        trg_ids = trg_tokenizer.encode(item['translation'][TARGET_LANGUAGE]).ids
        src_max_length = max(src_max_length, len(src_ids))
        trg_max_length = max(trg_max_length, len(trg_ids))

    print(f'Max length of source sentence: {src_max_length}')
    print(f'Max length of target sentence: {trg_max_length}')
    

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(validation_dataset, batch_size=1, shuffle=True) # Process the sentences one by one

    return train_dataloader, val_dataloader, src_tokenizer, trg_tokenizer