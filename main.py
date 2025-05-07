# -*- coding: utf-8 -*-
"""
Fully updated main.py for DeepSC + SemRect integration
"""

import os
import argparse
import time
import json
import torch
import random
import torch.nn as nn
import numpy as np
from utils import SNR_to_noise, initNetParams, train_step, val_step, train_mi
from dataset import EurDataset, collate_data
from models.transceiver import DeepSC
from SemRect import SemRect
from SemRectFIM import SemRectFIM
from deepsc_semrect import DeepSCWithSemRect
from torch.utils.data import DataLoader
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--vocab-file', default='europarl/vocab.json', type=str)
parser.add_argument('--checkpoint-path', default='checkpoints/deepsc-Rayleigh', type=str)
parser.add_argument('--channel', default='Rayleigh', type=str, help='AWGN, Rayleigh, or Rician')
parser.add_argument('--MAX-LENGTH', default=30, type=int)
parser.add_argument('--MIN-LENGTH', default=4, type=int)
parser.add_argument('--d-model', default=128, type=int)
parser.add_argument('--dff', default=512, type=int)
parser.add_argument('--num-layers', default=4, type=int)
parser.add_argument('--num-heads', default=8, type=int)
parser.add_argument('--batch-size', default=128, type=int)
parser.add_argument('--pretrain-epochs', default=50, type=int, help='Epochs for DeepSC pre-training')
parser.add_argument('--protection-epochs', default=30, type=int, help='Epochs for semantic protection training')
parser.add_argument('--finetune-epochs', default=1, type=int, help='Epochs for end-to-end fine-tuning')
parser.add_argument('--use-protection', choices=['none', 'semrect', 'semrectfim'], default='none',
                    help='Semantic protection type: none, semrect, or semrectfim')
parser.add_argument('--semrect-latent-dim', default=100, type=int, help='Latent dimension for semantic protection')
parser.add_argument('--snr-min', default=0, type=float, help='Minimum SNR value for training (dB)')
parser.add_argument('--snr-max', default=30, type=float, help='Maximum SNR value for training (dB)')
parser.add_argument('--test-snrs', default=[0, 5, 10, 15, 20], type=float, nargs='+', help='SNR values for testing')
parser.add_argument('--compare-protections', action='store_true', help='Run comparison between SemRect and SemRectFIM')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def setup_seed(seed):
    """Set random seeds for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def validate(epoch, args, net, pad_idx, criterion):
    """Validation loop with proper semantic signature handling"""
    test_eur = EurDataset('test')
    test_iterator = DataLoader(test_eur, batch_size=args.batch_size, num_workers=0,
                              pin_memory=True, collate_fn=collate_data)
    net.eval()
    pbar = tqdm(test_iterator)
    total_loss = 0
    
    with torch.no_grad():
        for sents in pbar:
            sents = sents.to(device)
            
            if isinstance(net, DeepSCWithSemRect):
                # Use semantic protection during validation
                output = net(sents, sents, 0.1, args.channel, use_protection=True)
                trg_real = sents[:, 1:]
                loss = criterion(output.reshape(-1, output.size(-1)), trg_real.reshape(-1))
                
                # Mask padding tokens
                mask = (trg_real != pad_idx).type_as(loss.data)
                loss = (loss * mask.reshape(-1)).sum() / mask.sum()
            else:
                loss = val_step(net, sents, sents, 0.1, pad_idx, criterion, args.channel)

            total_loss += loss
            pbar.set_description(
                f'Epoch: {epoch+1}; VAL; Loss: {loss:.5f}'
            )
    
    return total_loss / len(test_iterator)

def train_epoch(epoch, args, net, criterion, optimizer, pad_idx):
    """Single training epoch with proper gradient handling"""
    train_eur = EurDataset('train')
    train_loader = DataLoader(train_eur, batch_size=args.batch_size, num_workers=0,
                             pin_memory=True, collate_fn=collate_data)
    
    net.train()
    pbar = tqdm(train_loader)
    total_loss = 0
    
    for sents in pbar:
        sents = sents.to(device)
        
        # Random noise level based on SNR range
        snr = np.random.uniform(args.snr_min, args.snr_max)
        noise_std = SNR_to_noise(snr)
        
        # Forward pass
        sents_input = sents[:, :-1]
        sents_target = sents[:, 1:]
        
        if isinstance(net, DeepSCWithSemRect):
            optimizer.zero_grad()
            # For SemRectFIM, pass SNR value
            if args.use_protection == 'semrectfim':
                snr_tensor = torch.tensor([[snr]], device=device).expand(sents.size(0), 1)
                output = net(sents, sents, noise_std, args.channel, use_protection=True, snr=snr_tensor)
            else:
                output = net(sents, sents, noise_std, args.channel, use_protection=True)
                
            loss = criterion(output.reshape(-1, output.size(-1)), sents_target.reshape(-1))
            
            # Apply padding mask
            mask = (sents_target != pad_idx).float()
            loss = (loss * mask.view(-1)).sum() / mask.sum()
            
            loss.backward()
            optimizer.step()
            
            pbar.set_description(f'Epoch: {epoch+1}; Train; Loss: {loss.item():.5f}')
        else:
            loss = train_step(net, sents, sents, noise_std, pad_idx, optimizer, criterion, args.channel)
            pbar.set_description(f'Epoch: {epoch+1}; Train; Loss: {loss:.5f}')
        
        total_loss += loss
    
    return total_loss / len(train_loader)

def train_protection_module(args, model, train_loader):
    """Train semantic protection module (SemRect or SemRectFIM)"""
    print(f"Training {args.use_protection.upper()} component...")
    
    # Train the protection module
    model.train_protection_module(
        train_loader, 
        epochs=args.protection_epochs, 
        lr=1e-4, 
        epsilon=0.1,
        snr_range=(args.snr_min, args.snr_max)
    )
    
    # Save intermediate model
    torch.save(model.state_dict(), os.path.join(args.checkpoint_path, f'deepsc_with_{args.use_protection}.pth'))

def evaluate_adversarial_robustness(args, model, test_loader):
    """Evaluate model robustness against adversarial attacks at multiple SNR levels"""
    results = {}
    
    for snr in args.test_snrs:
        print(f"\nTesting adversarial robustness at SNR = {snr} dB")
        result = model.test_with_adversarial_attack(
            test_loader, 
            epsilon=0.1, 
            channel_type=args.channel,
            snr=snr
        )
        results[f"snr_{snr}"] = result
        
        print(f"Results at SNR = {snr} dB:")
        print(f"  Protection type: {result['protection_type']}")
        print(f"  Attack success rate: {result['attack_success_rate']:.4f}")
        print(f"  BLEU clean: {result['bleu_clean']:.4f}")
        print(f"  BLEU adversarial: {result['bleu_adversarial']:.4f}")
        print(f"  BLEU defended: {result['bleu_defended']:.4f}")
        print(f"  Defense effectiveness: {result['defense_effectiveness']:.4f}")
    
    return results

def compare_protection_methods(args):
    """Compare SemRect and SemRectFIM performance across different SNR levels"""
    print("\n=== Running comparison between SemRect and SemRectFIM ===")
    
    # Prepare dataset
    vocab = json.load(open(args.vocab_file, 'rb'))
    token_to_idx = vocab['token_to_idx']
    num_vocab = len(token_to_idx)
    pad_idx = token_to_idx["<PAD>"]
    
    test_data = EurDataset('test')
    test_loader = DataLoader(test_data, batch_size=args.batch_size, 
                           collate_fn=collate_data, num_workers=0)
    
    results = {
        'semrect': {},
        'semrectfim': {}
    }
    
    # Test each protection method
    for protection_type in ['semrect', 'semrectfim']:
        print(f"\nTesting {protection_type.upper()}")
        
        # Load or create model
        model_path = os.path.join(args.checkpoint_path, f'deepsc_with_{protection_type}.pth')
        
        model = DeepSCWithSemRect(
            num_layers=args.num_layers,
            src_vocab_size=num_vocab,
            trg_vocab_size=num_vocab,
            src_max_len=args.MAX_LENGTH,
            trg_max_len=args.MAX_LENGTH,
            d_model=args.d_model,
            num_heads=args.num_heads,
            dff=args.dff,
            dropout=0.1,
            semrect_latent_dim=args.semrect_latent_dim,
            protection_type=protection_type,
            device=device
        ).to(device)
        
        if os.path.exists(model_path):
            print(f"Loading saved {protection_type} model...")
            model.load_state_dict(torch.load(model_path))
        else:
            print(f"No saved {protection_type} model found. Skipping...")
            continue
        
        # Evaluate
        results[protection_type] = evaluate_adversarial_robustness(args, model, test_loader)
    
    # Save comparison results
    os.makedirs(args.checkpoint_path, exist_ok=True)
    with open(os.path.join(args.checkpoint_path, 'protection_comparison.json'), 'w') as f:
        json.dump(results, f, indent=4)
    
    return results

def main():
    args = parser.parse_args()
    args.vocab_file = '/content/data/txt/' + args.vocab_file
    
    # Set random seed for reproducibility
    setup_seed(42)

    """ Run protection method comparison if requested """
    if args.compare_protections:
        compare_protection_methods(args)
        return

    """ Prepare dataset and vocab """
    vocab = json.load(open(args.vocab_file, 'rb'))
    token_to_idx = vocab['token_to_idx']
    num_vocab = len(token_to_idx)
    pad_idx = token_to_idx["<PAD>"]
    
    """ Build model """
    if args.use_protection != 'none':
        print(f"Using DeepSC with {args.use_protection.upper()} protection")
        model_path = os.path.join(args.checkpoint_path, 'deepsc_pretrained.pth')
        
        if os.path.exists(model_path):
            print("Loading pre-trained DeepSC weights...")
            
            # Load base model with strict=False to skip positional encoding
            base_model = DeepSC(
                args.num_layers, num_vocab, num_vocab,
                num_vocab, num_vocab, args.d_model, args.num_heads,
                args.dff, 0.1
            ).to(device)
            
            # Filter out positional encoding weights
            state_dict = torch.load(model_path)
            filtered_state_dict = {
                k: v for k, v in state_dict.items() 
                if not k.startswith('encoder.pos_encoding') and not k.startswith('decoder.pos_encoding')
            }
            
            base_model.load_state_dict(filtered_state_dict, strict=False)
            
            # Build integrated model
            model = DeepSCWithSemRect(
                num_layers=args.num_layers,
                src_vocab_size=num_vocab,
                trg_vocab_size=num_vocab,
                src_max_len=args.MAX_LENGTH,
                trg_max_len=args.MAX_LENGTH,
                d_model=args.d_model,
                num_heads=args.num_heads,
                dff=args.dff,
                dropout=0.1,
                semrect_latent_dim=args.semrect_latent_dim,
                protection_type=args.use_protection,
                device=device
            ).to(device)
            
            # Load filtered weights into DeepSC component
            model.deepsc.load_state_dict(filtered_state_dict, strict=False)
        else:
            print("Pre-trained DeepSC not found. Training from scratch...")
            model = DeepSCWithSemRect(
                num_layers=args.num_layers,
                src_vocab_size=num_vocab,
                trg_vocab_size=num_vocab,
                src_max_len=args.MAX_LENGTH,
                trg_max_len=args.MAX_LENGTH,
                d_model=args.d_model,
                num_heads=args.num_heads,
                dff=args.dff,
                dropout=0.1,
                semrect_latent_dim=args.semrect_latent_dim,
                protection_type=args.use_protection,
                device=device
            ).to(device)
    else:
        print("Using standard DeepSC model")
        model = DeepSC(
            args.num_layers, num_vocab, num_vocab,
            num_vocab, num_vocab, args.d_model, args.num_heads,
            args.dff, 0.1
        ).to(device)
        initNetParams(model)

    """ Define loss and optimizers """
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
    
    # Initial optimizer for pre-training
    optimizer = torch.optim.Adam(model.parameters(), 
                               lr=1e-4, betas=(0.9, 0.98), 
                               eps=1e-8, weight_decay=5e-4)

    """ Stage 1: Pre-train DeepSC without protection """
    if args.use_protection == 'none':
        print("Stage 1: Pre-training DeepSC...")
        best_val = float('inf')
        for epoch in range(args.pretrain_epochs):
            loss = train_epoch(epoch, args, model, criterion, optimizer, pad_idx)
            val_loss = validate(epoch, args, model, pad_idx, criterion)
            
            if val_loss < best_val:
                best_val = val_loss
                os.makedirs(args.checkpoint_path, exist_ok=True)
                torch.save(model.state_dict(), os.path.join(args.checkpoint_path, 'deepsc_pretrained.pth'))

    """ Stage 2: Train protection module with frozen DeepSC """
    if args.use_protection != 'none':
        print(f"Stage 2: Training {args.use_protection.upper()} with frozen DeepSC...")
        train_data = EurDataset('train')
        train_loader = DataLoader(train_data, batch_size=args.batch_size, 
                               collate_fn=collate_data, num_workers=0)
        
        # Train using dedicated protection module training method
        train_protection_module(args, model, train_loader)
        
        """ Stage 3: Evaluate against adversarial attacks """
        print("Stage 3: Evaluating robustness against adversarial attacks...")
        test_data = EurDataset('test')
        test_loader = DataLoader(test_data, batch_size=args.batch_size, 
                               collate_fn=collate_data, num_workers=0)
        
        results = evaluate_adversarial_robustness(args, model, test_loader)
        
        # Save results
        os.makedirs(args.checkpoint_path, exist_ok=True)
        with open(os.path.join(args.checkpoint_path, f'{args.use_protection}_results.json'), 'w') as f:
            json.dump(results, f, indent=4)

if __name__ == '__main__':
    main()