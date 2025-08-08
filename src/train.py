from dataset import get_dataset, causal_mask
from transformer import Transformer
from config import *
from constants import *

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR

import warnings
from tqdm import tqdm
import os
from pathlib import Path

import torchmetrics
from torch.utils.tensorboard import SummaryWriter


def get_transformer(src_vocab_size, trg_vocab_size):
    transformer = Transformer(model_dimension=MODEL_DIMENSION,
                              inner_layer_dimension=MODEL_INNER_LAYER_DIMENSION,
                              number_of_layers=MODEL_NUMBER_OF_LAYERS,
                              number_of_heads=MODEL_NUMBER_OF_HEADS,
                              src_vocabulary_size=src_vocab_size,
                              trg_vocabulary_size=trg_vocab_size
                            )
    return transformer


def greedy_decode(model, source, source_mask, src_tokenizer, trg_tokenizer, max_len, device):
    sos_idx = trg_tokenizer.token_to_id('[SOS]')
    eos_idx = trg_tokenizer.token_to_id('[EOS]')

    # Precompute the encoder output and reuse it for every step
    encoder_output = model.encode(source, source_mask)
    # Initialize the decoder input with the sos token
    decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(source).to(device)
    while True:
        if decoder_input.size(1) == max_len:
            break

        # build mask for target
        decoder_mask = causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)

        # calculate output
        out = model.decode(encoder_output, decoder_input, source_mask, decoder_mask)

        # get next token
        prob = model.project(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        decoder_input = torch.cat(
            [decoder_input, torch.empty(1, 1).type_as(source).fill_(next_word.item()).to(device)], dim=1
        )

        if next_word == eos_idx:
            break

    return decoder_input.squeeze(0)


def run_validation(model, validation_ds, src_tokenizer, trg_tokenizer, max_len, device, print_msg, global_step, writer, num_examples=2):
    model.eval()
    count = 0

    source_texts = []
    expected = []
    predicted = []

    try:
        # get the console window width
        with os.popen('stty size', 'r') as console:
            _, console_width = console.read().split()
            console_width = int(console_width)
    except:
        # If we can't get the console width, use 80 as default
        console_width = 80

    with torch.no_grad():
        for batch in validation_ds:
            count += 1
            encoder_input = batch["encoder_input"].to(device) # (b, seq_len)
            encoder_mask = batch["encoder_mask"].to(device) # (b, 1, 1, seq_len)

            # check that the batch size is 1
            assert encoder_input.size(
                0) == 1, "Batch size must be 1 for validation"

            model_out = greedy_decode(model, encoder_input, encoder_mask, src_tokenizer, trg_tokenizer, max_len, device)

            source_text = batch["src_text"][0]
            target_text = batch["trg_text"][0]
            model_out_text = trg_tokenizer.decode(model_out.detach().cpu().numpy())

            source_texts.append(source_text)
            expected.append(target_text)
            predicted.append(model_out_text)
            
            # Print the source, target and model output
            print_msg('-'*console_width)
            print_msg(f"{f'SOURCE: ':>12}{source_text}")
            print_msg(f"{f'TARGET: ':>12}{target_text}")
            print_msg(f"{f'PREDICTED: ':>12}{model_out_text}")

            if count == num_examples:
                print_msg('-'*console_width)
                break
    
    if writer:
        # Evaluate the character error rate
        # Compute the char error rate 
        metric = torchmetrics.CharErrorRate()
        cer = metric(predicted, expected)
        writer.add_scalar('validation cer', cer, global_step)
        writer.flush()

        # Compute the word error rate
        metric = torchmetrics.WordErrorRate()
        wer = metric(predicted, expected)
        writer.add_scalar('validation wer', wer, global_step)
        writer.flush()

        # Compute the BLEU metric
        metric = torchmetrics.BLEUScore()
        bleu = metric(predicted, expected)
        writer.add_scalar('validation BLEU', bleu, global_step)
        writer.flush()


def learning_rate(step):
    if step == 0:
        step = 1
    return MODEL_DIMENSION ** (-0.5) * min(step ** (-0.5), step * WARMUP_STEPS ** (-1.5))


def train_model():
    # Define the device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)
    if (device == 'cuda'):
        print(f"Device name: {torch.cuda.get_device_name(device.index)}")
        print(f"Device memory: {torch.cuda.get_device_properties(device.index).total_memory / 1024 ** 3} GB")
    device = torch.device(device)

    # Make sure the weights folder exists
    Path(f"{DATASET_NAME}_{MODEL_FOLDER}").mkdir(parents=True, exist_ok=True)

    train_dataloader, val_dataloader, src_tokenizer, trg_tokenizer = get_dataset()
    model = get_transformer(src_tokenizer.get_vocab_size(), trg_tokenizer.get_vocab_size()).to(device)
    
    # Tensorboard
    writer = SummaryWriter(EXPERIMENT_FOLDER)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, betas=(BETA1, BETA2), eps=EPSILON)    
    lr_scheduler = LambdaLR(optimizer=optimizer, lr_lambda=lambda step: learning_rate(step))

    # If the user specified a model to preload before training, load it
    initial_epoch = 1
    global_step = 0
    preload = MODEL_PRELOAD
    model_filename = latest_weights_file_path() if preload == 'latest' else get_weights_file_path(preload) if preload else None
    if model_filename:
        print(f'Preloading model {model_filename}')
        state = torch.load(model_filename)
        model.load_state_dict(state['model_state_dict'])
        initial_epoch = state['epoch'] + 1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step = state['global_step']
    else:
        print('No model to preload, starting from scratch')

    loss_function = nn.CrossEntropyLoss(ignore_index=src_tokenizer.token_to_id('[PAD]'), label_smoothing=0.1).to(device)

    for epoch in range(initial_epoch, NUMBER_OF_EPOCHS + 1):
        torch.cuda.empty_cache()
        model.train()
        batch_iterator = tqdm(train_dataloader, desc=f"Processing Epoch {epoch:02d}")
        for batch in batch_iterator:

            encoder_input = batch['encoder_input'].to(device) # (B, seq_len)
            decoder_input = batch['decoder_input'].to(device) # (B, seq_len)
            encoder_mask = batch['encoder_mask'].to(device) # (B, 1, 1, seq_len)
            decoder_mask = batch['decoder_mask'].to(device) # (B, 1, seq_len, seq_len)

            # Run the tensors through the encoder, decoder and the projection layer
            model_output = model(encoder_input, decoder_input, encoder_mask, decoder_mask)

            # Compare the output with the label
            label = batch['label'].to(device) # (B, seq_len)

            # Compute the loss using a simple cross entropy
            loss = loss_function(model_output.view(-1, trg_tokenizer.get_vocab_size()), label.view(-1))
            batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"})

            # Log the loss
            writer.add_scalar('train loss', loss.item(), global_step)
            writer.flush()

            # Backpropagate the loss
            loss.backward()

            # Update the weights
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            lr_scheduler.step()

            global_step += 1

        # Run validation at the end of every epoch
        run_validation(model, val_dataloader, src_tokenizer, trg_tokenizer, SEQUENCE_LENGTH, device, lambda msg: batch_iterator.write(msg), global_step, writer)

        # Save the model at the end of every epoch
        model_filename = get_weights_file_path(f"{epoch:02d}")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'global_step': global_step
        }, model_filename)


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    train_model()