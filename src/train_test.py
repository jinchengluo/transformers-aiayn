import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchtext.datasets import Multi30k
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.nn.utils.rnn import pad_sequence

from transformer import Transformer


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SRC_LANGUAGE = 'en'
TGT_LANGUAGE = 'fr'


# Tokenizers
token_transform = {
    'en': get_tokenizer('spacy', language='en_core_web_sm'),
    'de': get_tokenizer('spacy', language='de_core_news_sm')
}


# Build vocab from iterator
def yield_tokens(data_iter, language):
    for src, tgt in data_iter:
        yield token_transform[language](src if language == SRC_LANGUAGE else tgt)


train_iter = Multi30k(split='train', language_pair=(SRC_LANGUAGE, TGT_LANGUAGE))

vocab_transform = {
    SRC_LANGUAGE: build_vocab_from_iterator(yield_tokens(train_iter, SRC_LANGUAGE),
                                            specials=["<unk>", "<pad>", "<bos>", "<eos>"],
                                            min_freq=2),
    TGT_LANGUAGE: build_vocab_from_iterator(yield_tokens(train_iter, TGT_LANGUAGE),
                                            specials=["<unk>", "<pad>", "<bos>", "<eos>"],
                                            min_freq=2)
}

for vocab in vocab_transform.values():
    vocab.set_default_index(vocab["<unk>"])

PAD_IDX = vocab_transform[SRC_LANGUAGE]["<pad>"]
BOS_IDX = vocab_transform[SRC_LANGUAGE]["<bos>"]
EOS_IDX = vocab_transform[SRC_LANGUAGE]["<eos>"]

# =========================
#     PIPELINE & DATALOADER
# =========================

def tensor_transform(token_ids):
    return torch.cat([torch.tensor([BOS_IDX]), torch.tensor(token_ids), torch.tensor([EOS_IDX])])

def collate_fn(batch):
    src_batch, tgt_batch = [], []
    for src_sample, tgt_sample in batch:
        src_tokens = vocab_transform[SRC_LANGUAGE](token_transform[SRC_LANGUAGE](src_sample))
        tgt_tokens = vocab_transform[TGT_LANGUAGE](token_transform[TGT_LANGUAGE](tgt_sample))
        src_batch.append(tensor_transform(src_tokens))
        tgt_batch.append(tensor_transform(tgt_tokens))
    src_batch = pad_sequence(src_batch, padding_value=PAD_IDX)
    tgt_batch = pad_sequence(tgt_batch, padding_value=PAD_IDX)
    return src_batch.to(device), tgt_batch.to(device)

train_iter = Multi30k(split='train', language_pair=(SRC_LANGUAGE, TGT_LANGUAGE))
train_dataloader = DataLoader(list(train_iter), batch_size=32, shuffle=True, collate_fn=collate_fn)

# =========================
#      MODEL + TRAINING
# =========================

model = Transformer(
    len(vocab_transform[SRC_LANGUAGE]),
    len(vocab_transform[TGT_LANGUAGE]),
    512,
    8,
    6,
    2028
).to(device)

loss_fn = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
optimizer = optim.Adam(model.parameters(), lr=1e-4)

def train_epoch(model, dataloader):
    model.train()
    total_loss = 0

    for src, tgt in dataloader:
        tgt_input = tgt[:-1, :]
        tgt_output = tgt[1:, :]

        src_mask = None
        tgt_mask = model.generate_square_subsequent_mask(tgt_input.size(0)).to(device)

        logits = model(src, tgt_input, src_mask=src_mask, tgt_mask=tgt_mask)
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_output.reshape(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


NUM_EPOCHS = 10

for epoch in range(1, NUM_EPOCHS + 1):
    loss = train_epoch(model, train_dataloader)
    print(f"Epoch {epoch}, Loss: {loss:.4f}")
