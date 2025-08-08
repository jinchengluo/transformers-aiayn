import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from transformers import AutoTokenizer
import numpy as np
from tqdm import tqdm
import json
import os
from torch.nn.utils.rnn import pad_sequence

from transformer import Transformer

class WMTDataset(Dataset):
    """Dataset pour WMT14 EN-FR"""
    def __init__(self, tokenizer_en, tokenizer_fr, split='train', max_length=128):
        self.tokenizer_en = tokenizer_en
        self.tokenizer_fr = tokenizer_fr
        self.max_length = max_length
        
        # Charger le dataset WMT14
        print(f"Chargement du dataset WMT14 EN-FR ({split})...")
        self.dataset = load_dataset('wmt14', 'fr-en', split=split)
        print(f"Dataset chargé: {len(self.dataset)} exemples")
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        example = self.dataset[idx]
        
        # Extraire les textes
        src_text = example['translation']['en']
        trg_text = example['translation']['fr']
        
        # Tokeniser
        src_tokens = self.tokenizer_en.encode(src_text, max_length=self.max_length, 
                                            truncation=True, add_special_tokens=True)
        trg_tokens = self.tokenizer_fr.encode(trg_text, max_length=self.max_length, 
                                            truncation=True, add_special_tokens=True)
        
        return {
            'src_tokens': torch.tensor(src_tokens, dtype=torch.long),
            'trg_tokens': torch.tensor(trg_tokens, dtype=torch.long),
            'src_text': src_text,
            'trg_text': trg_text
        }

def collate_fn(batch):
    """Fonction pour créer des batches avec padding"""
    src_tokens = [item['src_tokens'] for item in batch]
    trg_tokens = [item['trg_tokens'] for item in batch]
    
    # Padding
    src_padded = pad_sequence(src_tokens, batch_first=True, padding_value=0)
    trg_padded = pad_sequence(trg_tokens, batch_first=True, padding_value=0)
    
    return {
        'src_tokens': src_padded,
        'trg_tokens': trg_padded,
        'src_texts': [item['src_text'] for item in batch],
        'trg_texts': [item['trg_text'] for item in batch]
    }

def create_masks(src, trg, pad_idx=0):
    """Créer les masques d'attention"""
    # Masque de padding pour l'encoder
    src_mask = (src != pad_idx).unsqueeze(1).unsqueeze(2)
    
    # Masque combiné pour le decoder (padding + causal)
    trg_len = trg.size(1)
    trg_pad_mask = (trg != pad_idx).unsqueeze(1).unsqueeze(3)
    trg_causal_mask = torch.tril(torch.ones(trg_len, trg_len)).bool()
    trg_mask = trg_pad_mask & trg_causal_mask.unsqueeze(0).unsqueeze(0)
    
    return src_mask, trg_mask

class LabelSmoothingLoss(nn.Module):
    """Loss avec label smoothing pour améliorer la généralisation"""
    def __init__(self, vocab_size, smoothing=0.1, ignore_index=0):
        super().__init__()
        self.vocab_size = vocab_size
        self.smoothing = smoothing
        self.ignore_index = ignore_index
        
    def forward(self, pred, target):
        # pred: (batch_size, seq_len, vocab_size)
        # target: (batch_size, seq_len)
        
        pred = pred.view(-1, self.vocab_size)
        target = target.view(-1)
        
        # Masquer les tokens de padding
        mask = target != self.ignore_index
        pred = pred[mask]
        target = target[mask]
        
        if len(target) == 0:
            return torch.tensor(0.0, device=pred.device)
        
        # Label smoothing
        confidence = 1.0 - self.smoothing
        smooth_target = torch.full_like(pred, self.smoothing / (self.vocab_size - 1))
        smooth_target.scatter_(1, target.unsqueeze(1), confidence)
        
        return torch.mean(torch.sum(-smooth_target * torch.log_softmax(pred, dim=1), dim=1))

def get_lr_scheduler(optimizer, warmup_steps=4000):
    """Learning rate scheduler avec warmup (comme dans le papier)"""
    def lr_lambda(step):
        if step == 0:
            return 0
        return min(step ** (-0.5), step * warmup_steps ** (-1.5))
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

class TransformerTrainer:
    def __init__(self, model, train_loader, val_loader, tokenizer_en, tokenizer_fr, 
                 device='cuda', lr=1e-4, warmup_steps=4000):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.tokenizer_en = tokenizer_en
        self.tokenizer_fr = tokenizer_fr
        self.device = device
        
        # Optimizer et scheduler
        self.optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-9)
        self.scheduler = get_lr_scheduler(self.optimizer, warmup_steps)
        
        # Loss function
        self.criterion = LabelSmoothingLoss(
            vocab_size=tokenizer_fr.vocab_size,
            smoothing=0.1,
            ignore_index=tokenizer_fr.pad_token_id
        )
        
        # Métriques
        self.train_losses = []
        self.val_losses = []
        
    def train_epoch(self):
        """Entraîner une époque"""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc="Training")
        for batch in pbar:
            src_tokens = batch['src_tokens'].to(self.device)
            trg_tokens = batch['trg_tokens'].to(self.device)
            
            # Préparer les inputs/outputs
            trg_input = trg_tokens[:, :-1]  # Exclure le dernier token pour l'input
            trg_output = trg_tokens[:, 1:]  # Exclure le premier token pour l'output
            
            # Créer les masques
            src_mask, trg_mask = create_masks(src_tokens, trg_input)
            src_mask = src_mask.to(self.device)
            trg_mask = trg_mask.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            pred = self.model(src_tokens, trg_input, src_mask, trg_mask)
            
            # Calculer la loss
            loss = self.criterion(pred, trg_output)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.scheduler.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            # Mettre à jour la barre de progression
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'lr': f'{self.scheduler.get_last_lr()[0]:.2e}'
            })
        
        avg_loss = total_loss / num_batches
        self.train_losses.append(avg_loss)
        return avg_loss
    
    def validate(self):
        """Valider le modèle"""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validating"):
                src_tokens = batch['src_tokens'].to(self.device)
                trg_tokens = batch['trg_tokens'].to(self.device)
                
                trg_input = trg_tokens[:, :-1]
                trg_output = trg_tokens[:, 1:]
                
                src_mask, trg_mask = create_masks(src_tokens, trg_input)
                src_mask = src_mask.to(self.device)
                trg_mask = trg_mask.to(self.device)
                
                pred = self.model(src_tokens, trg_input, src_mask, trg_mask)
                loss = self.criterion(pred, trg_output)
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        self.val_losses.append(avg_loss)
        return avg_loss
    
    def save_checkpoint(self, epoch, val_loss, filepath):
        """Sauvegarder un checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_loss': val_loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }
        torch.save(checkpoint, filepath)
        print(f"Checkpoint sauvegardé: {filepath}")

def main():
    # Configuration
    config = {
        'model_dimension': 512,
        'inner_layer_dimension': 2048,
        'number_of_layers': 6,
        'number_of_heads': 8,
        'max_length': 128,
        'batch_size': 32,
        'learning_rate': 1e-4,
        'warmup_steps': 4000,
        'num_epochs': 10,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    print("Configuration:", config)
    
    # Tokenizers
    print("Chargement des tokenizers...")
    tokenizer_en = AutoTokenizer.from_pretrained('bert-base-uncased')
    tokenizer_fr = AutoTokenizer.from_pretrained('flaubert/flaubert_base_cased')
    
    # Si les tokenizers n'ont pas de pad_token, l'ajouter
    if tokenizer_en.pad_token is None:
        tokenizer_en.pad_token = tokenizer_en.unk_token
    if tokenizer_fr.pad_token is None:
        tokenizer_fr.pad_token = tokenizer_fr.unk_token
    
    # Datasets
    print("Création des datasets...")
    train_dataset = WMTDataset(tokenizer_en, tokenizer_fr, split='train[:10000]', 
                              max_length=config['max_length'])  # Limité pour test
    val_dataset = WMTDataset(tokenizer_en, tokenizer_fr, split='validation[:1000]', 
                            max_length=config['max_length'])
    
    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], 
                             shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], 
                           shuffle=False, collate_fn=collate_fn)
    
    # Modèle
    print("Initialisation du modèle...")
    model = Transformer(
        model_dimension=config['model_dimension'],
        inner_layer_dimension=config['inner_layer_dimension'],
        number_of_layers=config['number_of_layers'],
        number_of_heads=config['number_of_heads'],
        src_vocabulary_size=tokenizer_en.vocab_size,
        trg_vocabulary_size=tokenizer_fr.vocab_size
    )
    
    print(f"Nombre de paramètres: {sum(p.numel() for p in model.parameters()):,}")
    
    # Trainer
    trainer = TransformerTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        tokenizer_en=tokenizer_en,
        tokenizer_fr=tokenizer_fr,
        device=config['device'],
        lr=config['learning_rate'],
        warmup_steps=config['warmup_steps']
    )
    
    # Boucle d'entraînement
    print("Début de l'entraînement...")
    best_val_loss = float('inf')
    
    for epoch in range(config['num_epochs']):
        print(f"\nÉpoque {epoch + 1}/{config['num_epochs']}")
        
        # Entraînement
        train_loss = trainer.train_epoch()
        
        # Validation
        val_loss = trainer.validate()
        
        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Sauvegarder le meilleur modèle
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            trainer.save_checkpoint(epoch, val_loss, 'best_model.pth')
        
        # Sauvegarder checkpoint régulier
        if (epoch + 1) % 2 == 0:
            trainer.save_checkpoint(epoch, val_loss, f'checkpoint_epoch_{epoch+1}.pth')
    
    print("Entraînement terminé!")

if __name__ == "__main__":
    main()