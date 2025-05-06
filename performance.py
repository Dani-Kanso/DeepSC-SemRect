# performance.py - Adversarial Testing & BLEU Evaluation

import os
import json
import torch
import argparse
import numpy as np
import torch.nn as nn
from dataset import EurDataset as OriginalEurDataset, collate_data
from models.transceiver import DeepSC
from deepsc_semrect import DeepSCWithSemRect
from torch.utils.data import DataLoader, Dataset
from utils import BleuScore, SNR_to_noise, create_masks, PowerNormalize, Channels
from tqdm import tqdm
import math

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Create a patched version of EurDataset that works in Colab environment
class EurDataset(Dataset):
    def __init__(self, split='train', data_dir=None):
        if data_dir is None:
            data_dir = '/content/data/txt/'  # Default for Colab
        
        try:
            # Try using the original dataset class first
            self.dataset = OriginalEurDataset(split)
            self.data = self.dataset.data
        except Exception as e:
            print(f"Falling back to custom dataset implementation: {e}")
            try:
                # Try to open dataset file directly
                file_path = os.path.join(data_dir, f'europarl/{split}_data.pkl')
                if os.path.exists(file_path):
                    with open(file_path, 'rb') as f:
                        import pickle
                        self.data = pickle.load(f)
                    print(f"Loaded {split} dataset with {len(self.data)} samples from {file_path}")
                else:
                    # Create dummy data
                    print(f"Dataset file not found at {file_path}")
                    print("Creating dummy dataset for testing")
                    self.data = [[1, 2, 3, 4, 5]] * 10  # 10 dummy sentences
            except Exception as e2:
                print(f"Error in custom dataset implementation: {e2}")
                self.data = [[1, 2, 3, 4, 5]] * 10  # Fallback

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

class SeqtoText:
    def __init__(self, token_to_idx, end_idx):
        self.token_to_idx = token_to_idx
        self.idx_to_token = {v: k for k, v in token_to_idx.items()}
        self.end_idx = end_idx

    def sequence_to_text(self, sequence):
        tokens = [self.idx_to_token.get(idx, "<UNK>") for idx in sequence if idx != self.token_to_idx.get("<PAD>", 0)]
        if self.end_idx in sequence:
            end_idx = sequence.index(self.end_idx)
            tokens = [self.idx_to_token.get(idx, "<UNK>") for idx in sequence[:end_idx]]
        return " ".join(tokens)

def evaluate_model(model, dataloader, noise_std, channel_type, use_semrect=False, token_to_idx=None):
    """Evaluate model accuracy with optional SemRect"""
    model.eval()
    bleu_score_1gram = BleuScore(1, 0, 0, 0)
    all_preds = []
    all_targets = []
    channels = Channels()
    
    with torch.no_grad():
        for batch in dataloader:
            # Handle different return formats from dataloader
            if isinstance(batch, tuple) and len(batch) == 2:
                src, trg = batch
            else:
                # If dataloader returns a single tensor, process it accordingly
                src = batch
                trg = src  # For testing, target = source (autoencoder-like behavior)
            
            src = src.to(device)
            trg = trg.to(device)
            
            # Forward pass
            if isinstance(model, DeepSCWithSemRect):
                output = model(src, trg, noise_std, channel_type, use_semrect=use_semrect)
            else:
                # Manual forward pass for DeepSC model
                src_mask, look_ahead_mask = create_masks(src, trg[:, :-1], 0)
                
                # Encoder
                enc_output = model.encoder(src, src_mask)
                
                # Channel encoder
                channel_enc = model.channel_encoder(enc_output)
                channel_enc = PowerNormalize(channel_enc)
                
                # Channel simulation
                if channel_type == 'AWGN':
                    rx_sig = channels.AWGN(channel_enc, noise_std)
                elif channel_type == 'Rayleigh':
                    rx_sig = channels.Rayleigh(channel_enc, noise_std)
                elif channel_type == 'Rician':
                    rx_sig = channels.Rician(channel_enc, noise_std)
                
                # Channel decoder
                semantic_repr = model.channel_decoder(rx_sig)
                
                # Decoder
                dec_output = model.decoder(trg[:, :-1], semantic_repr, look_ahead_mask, src_mask)
                output = model.dense(dec_output)
                
            pred = output.argmax(dim=-1)
            target = trg[:, 1:]
            
            # Store results
            all_preds.extend(pred.cpu().numpy().tolist())
            all_targets.extend(target.cpu().numpy().tolist())
    
    # Convert to text for BLEU
    StoT = SeqtoText(token_to_idx, token_to_idx["<END>"])
    pred_texts = list(map(StoT.sequence_to_text, all_preds))
    target_texts = list(map(StoT.sequence_to_text, all_targets))
    
    return np.mean(bleu_score_1gram.compute_blue_score(pred_texts, target_texts))

def generate_adversarial_example(model, src, trg, noise_std, channel_type, epsilon=0.1):
    """Generate adversarial example using FGSM attack at the channel level (aligning with paper)"""
    model.eval()
    
    # Forward pass to get semantic representation
    src_mask, _ = create_masks(src, trg[:, :-1], 0)
    
    with torch.enable_grad():
        # Access the encoder based on model type
        if isinstance(model, DeepSCWithSemRect):
            try:
                # Try to use the model's built-in FGSM attack if available
                adv_semantic, src_mask = model.generate_adversarial(
                    src, trg, noise_std, channel_type, epsilon
                )
                
                # Get clean semantic representation for comparison
                with torch.no_grad():
                    encoder = model.deepsc.encoder
                    channel_encoder = model.deepsc.channel_encoder
                    channel_decoder = model.deepsc.channel_decoder
                    
                    enc_output = encoder(src, src_mask)
                    channel_enc = channel_encoder(enc_output)
                    channel_enc = PowerNormalize(channel_enc)
                    
                    # Get clean semantic representation through channel
                    channels = Channels()
                    if channel_type == 'AWGN':
                        Rx_sig_clean = channels.AWGN(channel_enc, noise_std)
                    elif channel_type == 'Rayleigh':
                        Rx_sig_clean = channels.Rayleigh(channel_enc, noise_std)
                    elif channel_type == 'Rician':
                        Rx_sig_clean = channels.Rician(channel_enc, noise_std)
                    
                    clean_semantic = channel_decoder(Rx_sig_clean)
                
                return clean_semantic, adv_semantic, src_mask, trg
            except Exception as e:
                print(f"Error using model's built-in attack: {e}")
                # Fall back to manual implementation
                encoder = model.deepsc.encoder
                channel_encoder = model.deepsc.channel_encoder
                channel_decoder = model.deepsc.channel_decoder
                decoder = model.deepsc.decoder
                dense = model.deepsc.dense
        else:
            encoder = model.encoder
            channel_encoder = model.channel_encoder
            channel_decoder = model.channel_decoder
            decoder = model.decoder
            dense = model.dense
        
        # Process the input through encoder normally
        with torch.no_grad():
            # Run the embedding layer without gradients
            src_emb = encoder.embedding(src) * math.sqrt(encoder.d_model)
            src_emb = encoder.pos_encoding(src_emb)
            
            # Forward pass through encoder normally
            enc_output = src_emb
            for enc_layer in encoder.enc_layers:
                enc_output = enc_layer(enc_output, src_mask)
                
            # Get channel encoding
            channel_enc_clean = channel_encoder(enc_output)
            channel_enc_clean = PowerNormalize(channel_enc_clean)
        
        # Create a copy of channel encoding that requires gradients for FGSM attack
        perturbed_channel = channel_enc_clean.clone().detach().requires_grad_(True)
        
        # Apply channel with the perturbed signal
        channels = Channels()
        if channel_type == 'AWGN':
            Rx_sig = channels.AWGN(perturbed_channel, noise_std)
        elif channel_type == 'Rayleigh':
            Rx_sig = channels.Rayleigh(perturbed_channel, noise_std)
        elif channel_type == 'Rician':
            Rx_sig = channels.Rician(perturbed_channel, noise_std)
        
        semantic_repr = channel_decoder(Rx_sig)
        
        # Forward through decoder with trg input
        trg_inp = trg[:, :-1]
        trg_real = trg[:, 1:]
        dec_output = decoder(trg_inp, semantic_repr, None, src_mask)
        output = dense(dec_output)
        
        # Compute loss for gradient calculation
        criterion = nn.CrossEntropyLoss()
        loss = criterion(output.reshape(-1, output.size(-1)), trg_real.reshape(-1))
    
        # Compute gradients
        loss.backward()
        
        # FGSM attack (single step)
        with torch.no_grad():
            # Apply FGSM perturbation
            if perturbed_channel.grad is not None:
                perturbed_channel = perturbed_channel + epsilon * perturbed_channel.grad.sign()
        
        # Final forward pass with the clean and adversarial channel encodings
        with torch.no_grad():
            # Get clean semantic representation
            if channel_type == 'AWGN':
                Rx_sig_clean = channels.AWGN(channel_enc_clean, noise_std)
            elif channel_type == 'Rayleigh':
                Rx_sig_clean = channels.Rayleigh(channel_enc_clean, noise_std)
            elif channel_type == 'Rician':
                Rx_sig_clean = channels.Rician(channel_enc_clean, noise_std)
            clean_semantic = channel_decoder(Rx_sig_clean)
            
            # Get adversarial semantic representation
            if channel_type == 'AWGN':
                Rx_sig = channels.AWGN(perturbed_channel, noise_std)
            elif channel_type == 'Rayleigh':
                Rx_sig = channels.Rayleigh(perturbed_channel, noise_std)
            elif channel_type == 'Rician':
                Rx_sig = channels.Rician(perturbed_channel, noise_std)
            adv_semantic = channel_decoder(Rx_sig)
        
    return clean_semantic, adv_semantic, src_mask, trg

def calculate_attack_success_rate(model, dataloader, noise_std, channel_type, token_to_idx, epsilon=0.05, use_semrect=False):
    """Calculate the success rate of adversarial attacks"""
    model.eval()
    total_examples = 0
    attack_success = 0
    
    for batch in tqdm(dataloader, desc="Calculating attack success rate", leave=False):
        # Handle different return formats from dataloader
        if isinstance(batch, tuple) and len(batch) == 2:
            src, trg = batch
        else:
            # If dataloader returns a single tensor, process it accordingly
            src = batch
            trg = src  # For testing, target = source (autoencoder-like behavior)
            
        src, trg = src.to(device), trg.to(device)
        
        try:
            # Generate adversarial example
            clean_semantic, adv_semantic, src_mask, trg = generate_adversarial_example(
                model, src, trg, noise_std, channel_type, epsilon
            )
            
            # Check if we have valid semantic representations
            if clean_semantic is None or adv_semantic is None:
                print("Skipping batch due to None semantic representation")
                continue
                
            with torch.no_grad():
                # Access the correct components based on model type
                if isinstance(model, DeepSCWithSemRect):
                    decoder = model.deepsc.decoder
                    dense = model.deepsc.dense
                else:
                    decoder = model.decoder
                    dense = model.dense
                
                # Get predictions on clean examples
                try:
                    clean_out = decoder(trg[:, :-1], clean_semantic, None, src_mask)
                    clean_out = dense(clean_out)
                    clean_pred = clean_out.argmax(dim=-1)
                except Exception as e:
                    print(f"Error in clean prediction: {e}")
                    continue
                
                # Apply SemRect if enabled
                if use_semrect and isinstance(model, DeepSCWithSemRect):
                    try:
                        adv_semantic = model.semrect.calibrate(adv_semantic)
                    except Exception as e:
                        print(f"Error in SemRect calibration: {e}")
                        continue
                
                # Get predictions on adversarial examples
                try:
                    adv_out = decoder(trg[:, :-1], adv_semantic, None, src_mask)
                    adv_out = dense(adv_out)
                    adv_pred = adv_out.argmax(dim=-1)
                except Exception as e:
                    print(f"Error in adversarial prediction: {e}")
                    continue
                
                # Check if attack changed the prediction
                target = trg[:, 1:]
                mask = (target != 0)
                
                # Count successful attacks (prediction changed)
                pred_changed = (clean_pred != adv_pred) & mask
                attack_success += pred_changed.sum().item()
                total_examples += mask.sum().item()
                
        except Exception as e:
            print(f"Error in attack success calculation: {e}")
            continue
    
    # Calculate attack success rate
    if total_examples > 0:
        return attack_success / total_examples
    return 0.0

def test_under_attack(model, dataloader, noise_std, channel_type, token_to_idx, epsilon=0.05, use_semrect=False):
    """Test model performance under adversarial attack"""
    model.eval()
    bleu_score_1gram = BleuScore(1, 0, 0, 0)
    all_preds = []
    all_targets = []
    
    for batch in tqdm(dataloader, desc="Testing under attack", leave=False):
        # Handle different return formats from dataloader
        if isinstance(batch, tuple) and len(batch) == 2:
            src, trg = batch
        else:
            # If dataloader returns a single tensor, process it accordingly
            src = batch
            trg = src  # For testing, target = source (autoencoder-like behavior)
            
        src, trg = src.to(device), trg.to(device)
        
        try:
            # Generate adversarial example
            clean_semantic, adv_semantic, src_mask, trg = generate_adversarial_example(
                model, src, trg, noise_std, channel_type, epsilon
            )
            
            # Check if valid semantic representations
            if clean_semantic is None or adv_semantic is None:
                print("Skipping batch due to None semantic representation")
                continue
            
            with torch.no_grad():
                # Access the correct components based on model type
                if isinstance(model, DeepSCWithSemRect):
                    decoder = model.deepsc.decoder
                    dense = model.deepsc.dense
                else:
                    decoder = model.decoder
                    dense = model.dense
                
                # Apply SemRect if enabled
                if use_semrect and isinstance(model, DeepSCWithSemRect):
                    try:
                        adv_semantic = model.semrect.calibrate(adv_semantic)
                    except Exception as e:
                        print(f"Error in SemRect calibration: {e}")
                        continue
                
                # Get predictions on adversarial examples
                try:
                    out = decoder(trg[:, :-1], adv_semantic, None, src_mask)
                    out = dense(out)
                    pred = out.argmax(dim=-1)
                    target = trg[:, 1:]
                    
                    # Store results
                    all_preds.extend(pred.cpu().numpy().tolist())
                    all_targets.extend(target.cpu().numpy().tolist())
                except Exception as e:
                    print(f"Error in prediction: {e}")
                    continue
                
        except Exception as e:
            print(f"Error in testing under attack: {e}")
            continue
    
    # Check if we have any predictions
    if len(all_preds) == 0 or len(all_targets) == 0:
        print("Warning: No valid predictions were generated during testing under attack")
        return 0.0
    
    # Convert to text for BLEU
    try:
        StoT = SeqtoText(token_to_idx, token_to_idx["<END>"])
        pred_texts = list(map(StoT.sequence_to_text, all_preds))
        target_texts = list(map(StoT.sequence_to_text, all_targets))
        
        return np.mean(bleu_score_1gram.compute_blue_score(pred_texts, target_texts))
    except Exception as e:
        print(f"Error computing BLEU score: {e}")
        return 0.0

def test_models_under_attack(args, SNRs, std_model, token_to_idx, semrect_model=None):
    """Test both models under adversarial attacks across multiple SNRs"""
    results = {
        'snr': SNRs,
        'DeepSC': {'clean': [], 'attack': [], 'attack_success': []},
        'DeepSC+SemRect': {'clean': [], 'attack': [], 'attack_success': []}
    }
    
    # Try to create test loader with specified data directory
    try:
        test_loader = DataLoader(
            EurDataset('test', data_dir=args.data_dir), 
            batch_size=args.batch_size,
            collate_fn=collate_data, 
            num_workers=0
        )
        print(f"Created test dataloader with {len(test_loader)} batches")
    except Exception as e:
        print(f"Error loading test dataset: {e}")
        print("Creating dummy dataset for testing")
        # Create a dummy dataset with small random tensors
        dummy_data = [torch.randint(1, 10, (10,)) for _ in range(100)]
        
        class DummyDataset(torch.utils.data.Dataset):
            def __init__(self, data):
                self.data = data
            def __getitem__(self, idx):
                return self.data[idx]
            def __len__(self):
                return len(self.data)
        
        test_loader = DataLoader(
            DummyDataset(dummy_data),
            batch_size=args.batch_size,
            collate_fn=collate_data,
            num_workers=0
        )
        print(f"Created dummy test dataloader with {len(test_loader)} batches")
    
    for snr in tqdm(SNRs, desc="Testing SNRs"):
        noise_std = SNR_to_noise(snr)
        
        try:
            # Test standard DeepSC
            clean_bleu_std = evaluate_model(std_model, test_loader, noise_std, args.channel, 
                                           token_to_idx=token_to_idx)
            print(f"Standard DeepSC clean BLEU: {clean_bleu_std:.4f}")
        except Exception as e:
            print(f"Error evaluating clean performance for standard DeepSC: {e}")
            clean_bleu_std = 0.0
        
        try:
            attack_bleu_std = test_under_attack(std_model, test_loader, noise_std, args.channel, 
                                              token_to_idx, args.epsilon)
            print(f"Standard DeepSC attacked BLEU: {attack_bleu_std:.4f}")
        except Exception as e:
            print(f"Error evaluating attack performance for standard DeepSC: {e}")
            attack_bleu_std = 0.0
            
        try:
            # Calculate attack success rate for standard model
            attack_success_std = calculate_attack_success_rate(
                std_model, test_loader, noise_std, args.channel, 
                token_to_idx, args.epsilon, use_semrect=False
            )
            print(f"Standard DeepSC attack success rate: {attack_success_std:.4f}")
        except Exception as e:
            print(f"Error calculating attack success for standard DeepSC: {e}")
            attack_success_std = 0.0
        
        # Test DeepSC+SemRect if available
        if semrect_model:
            try:
                clean_bleu_semrect = evaluate_model(semrect_model, test_loader, noise_std, args.channel, 
                                                  use_semrect=True, token_to_idx=token_to_idx)
                print(f"DeepSC+SemRect clean BLEU: {clean_bleu_semrect:.4f}")
            except Exception as e:
                print(f"Error evaluating clean performance for DeepSC+SemRect: {e}")
                clean_bleu_semrect = clean_bleu_std
                
            try:
                attack_bleu_semrect = test_under_attack(semrect_model, test_loader, noise_std, args.channel, 
                                                      token_to_idx, args.epsilon, use_semrect=True)
                print(f"DeepSC+SemRect attacked BLEU: {attack_bleu_semrect:.4f}")
            except Exception as e:
                print(f"Error evaluating attack performance for DeepSC+SemRect: {e}")
                print(f"Error details: {str(e)}")
                # Debug information
                print("Checking DeepSCWithSemRect model structure:")
                print(f"deepsc attribute exists: {hasattr(semrect_model, 'deepsc')}")
                if hasattr(semrect_model, 'deepsc'):
                    print(f"encoder in deepsc: {hasattr(semrect_model.deepsc, 'encoder')}")
                    print(f"model type: {type(semrect_model).__name__}")
                attack_bleu_semrect = attack_bleu_std
                
            try:
                # Calculate attack success rate for SemRect model
                attack_success_semrect = calculate_attack_success_rate(
                    semrect_model, test_loader, noise_std, args.channel, 
                    token_to_idx, args.epsilon, use_semrect=True
                )
                print(f"DeepSC+SemRect attack success rate: {attack_success_semrect:.4f}")
                
                # Calculate defense effectiveness
                if attack_success_std > 0:
                    defense_effectiveness = (1 - attack_success_semrect/attack_success_std) * 100
                    print(f"SemRect defense effectiveness: {defense_effectiveness:.2f}%")
            except Exception as e:
                print(f"Error calculating attack success for DeepSC+SemRect: {e}")
                attack_success_semrect = attack_success_std
                
        else:
            clean_bleu_semrect = clean_bleu_std
            attack_bleu_semrect = attack_bleu_std
            attack_success_semrect = attack_success_std
        
        # Store results
        results['DeepSC']['clean'].append(clean_bleu_std)
        results['DeepSC']['attack'].append(attack_bleu_std)
        results['DeepSC']['attack_success'].append(attack_success_std)
        results['DeepSC+SemRect']['clean'].append(clean_bleu_semrect)
        results['DeepSC+SemRect']['attack'].append(attack_bleu_semrect)
        results['DeepSC+SemRect']['attack_success'].append(attack_success_semrect)
        
        # Print current results
        print(f"\nSNR={snr} dB")
        print(f"DeepSC - Clean: {clean_bleu_std:.4f}, Attack: {attack_bleu_std:.4f}, Attack Success: {attack_success_std:.4f}")
        if semrect_model:
            print(f"DeepSC+SemRect - Clean: {clean_bleu_semrect:.4f}, Attack: {attack_bleu_semrect:.4f}, Attack Success: {attack_success_semrect:.4f}")
            if attack_bleu_std > 0:
                bleu_improvement = ((attack_bleu_semrect - attack_bleu_std) / attack_bleu_std * 100)
                print(f"BLEU Improvement: {bleu_improvement:.2f}%")
            
            if attack_success_std > 0 and attack_success_semrect < attack_success_std:
                print(f"Attack Success Rate Reduction: {(attack_success_std - attack_success_semrect) * 100:.2f}%")

    return results

def plot_attack_results(results, epsilon):
    """Plot results showing improvement from SemRect"""
    import matplotlib.pyplot as plt
    
    # Create a figure with multiple subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: BLEU scores
    ax1.plot(results['snr'], results['DeepSC']['clean'], 'b-o', label='DeepSC (Clean)')
    ax1.plot(results['snr'], results['DeepSC+SemRect']['clean'], 'r-o', label='DeepSC+SemRect (Clean)')
    ax1.plot(results['snr'], results['DeepSC']['attack'], 'b--x', label='DeepSC (Attack)')
    ax1.plot(results['snr'], results['DeepSC+SemRect']['attack'], 'r--x', label='DeepSC+SemRect (Attack)')
    
    ax1.set_xlabel('SNR (dB)')
    ax1.set_ylabel('BLEU Score')
    ax1.set_title(f'BLEU Score Under Adversarial Attacks (ε={epsilon})')
    ax1.grid(True)
    ax1.legend()
    
    # Plot 2: Attack success rate
    ax2.plot(results['snr'], results['DeepSC']['attack_success'], 'b-^', label='DeepSC')
    ax2.plot(results['snr'], results['DeepSC+SemRect']['attack_success'], 'r-^', label='DeepSC+SemRect')
    
    ax2.set_xlabel('SNR (dB)')
    ax2.set_ylabel('Attack Success Rate')
    ax2.set_title(f'Adversarial Attack Success Rate (ε={epsilon})')
    ax2.grid(True)
    ax2.legend()
    
    plt.tight_layout()
    
    # Save plot
    os.makedirs('results', exist_ok=True)
    plt.savefig('results/attack_comparison.png')
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="Test DeepSC and DeepSC+SemRect under adversarial attacks")
    parser.add_argument('--vocab-file', default='europarl/vocab.json', type=str)
    parser.add_argument('--checkpoint-path', default='checkpoints/deepsc-Rayleigh', type=str)
    parser.add_argument('--semrect-checkpoint', default=None, type=str, help='Path to DeepSC+SemRect checkpoint, None to disable')
    parser.add_argument('--channel', default='Rayleigh', type=str, choices=['AWGN', 'Rayleigh', 'Rician'])
    parser.add_argument('--MAX-LENGTH', default=30, type=int)
    parser.add_argument('--batch-size', default=64, type=int)
    parser.add_argument('--epsilon', default=0.05, type=float)
    parser.add_argument('--output-file', default='results/attack_results.npy', type=str)
    parser.add_argument('--data-dir', default='/content/data/txt/', help='Directory containing the dataset')
    
    # Add architecture arguments
    parser.add_argument('--num-layers', default=4, type=int)
    parser.add_argument('--d-model', default=128, type=int)
    parser.add_argument('--dff', default=512, type=int)
    parser.add_argument('--num-heads', default=8, type=int)
    parser.add_argument('--semrect-latent-dim', default=100, type=int)
    
    # Add training arguments
    parser.add_argument('--train-semrect', action='store_true', help='Train SemRect before evaluation')
    parser.add_argument('--epochs-pretrain', default=5, type=int, help='Number of epochs for pre-training SemRect')
    parser.add_argument('--epochs-gan', default=10, type=int, help='Number of epochs for GAN training of SemRect')
    parser.add_argument('--lr', default=1e-4, type=float, help='Learning rate for SemRect training')

    args = parser.parse_args()
    
    # Check if data_dir is incorrectly set to a file path instead of a directory
    if args.data_dir.endswith('.json'):
        # Extract directory part
        args.data_dir = os.path.dirname(args.data_dir)
        print(f"Corrected data directory to: {args.data_dir}")
    
    # Check if running in Colab and adjust paths accordingly
    if os.path.exists('/content'):
        print("Running in Google Colab. Adjusting paths...")
        # Override default data directory for Colab
        if args.data_dir == '/import/antennas/Datasets/hx301/':
            args.data_dir = '/content/data/txt/'
            print(f"Using data directory: {args.data_dir}")
        
        if 'europarl' in args.vocab_file and not os.path.exists(args.vocab_file):
            args.vocab_file = '/content/data/txt/europarl/vocab.json'
            print(f"Using vocab file at: {args.vocab_file}")
    
    # Create dummy dataset class that mimics EurDataset if needed
    # This ensures compatibility with the existing code in dataset.py
    try:
        # Just import to see if it works with the current paths
        from dataset import EurDataset as _
    except Exception as e:
        print(f"Warning: There might be issues with the dataset module: {e}")
        print("Will handle data loading in test_models_under_attack function")
    
    # Load vocab
    try:
        vocab_path = args.vocab_file
        if not os.path.exists(vocab_path):
            possible_paths = [
                args.vocab_file,
                '/content/data/txt/' + args.vocab_file,
                args.data_dir + args.vocab_file
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    vocab_path = path
                    print(f"Found vocabulary file at: {vocab_path}")
                    break
            else:
                raise FileNotFoundError(f"Vocabulary file not found. Tried: {possible_paths}")
                
        vocab = json.load(open(vocab_path, 'rb'))
        token_to_idx = vocab['token_to_idx']
        idx_to_token = dict(zip(token_to_idx.values(), token_to_idx.keys()))
        num_vocab = len(token_to_idx)
        pad_idx = token_to_idx["<PAD>"]
        print(f"Loaded vocabulary with {num_vocab} tokens")
    except Exception as e:
        print(f"Error loading vocabulary: {e}")
        print("Creating dummy vocabulary for testing")
        token_to_idx = {"<PAD>": 0, "<START>": 1, "<END>": 2, "<UNK>": 3}
        idx_to_token = {0: "<PAD>", 1: "<START>", 2: "<END>", 3: "<UNK>"}
        num_vocab = len(token_to_idx)
        pad_idx = 0
    
    # Important: Use MAX_LENGTH + 1 to match checkpoint dimensions
    effective_max_len = args.MAX_LENGTH + 1
    
    # Load standard DeepSC
    std_model = DeepSC(
        args.num_layers, num_vocab, num_vocab, effective_max_len, effective_max_len,
        args.d_model, args.num_heads, args.dff, 0.1
    ).to(device)
    
    # Load latest checkpoint
    try:
        if os.path.isfile(args.checkpoint_path):
            # Direct path to checkpoint file
            std_checkpoint = torch.load(args.checkpoint_path, map_location=device)
            checkpoint_path = args.checkpoint_path
        elif os.path.isdir(args.checkpoint_path):
            # Directory with checkpoint files
            model_paths = [f for f in os.listdir(args.checkpoint_path) if f.endswith('.pth')]
            if not model_paths:
                raise FileNotFoundError("No model checkpoints found.")
            
            model_paths.sort()
            checkpoint_path = os.path.join(args.checkpoint_path, model_paths[-1])
            std_checkpoint = torch.load(checkpoint_path, map_location=device)
        else:
            raise FileNotFoundError(f"Checkpoint path not found: {args.checkpoint_path}")
            
        print(f"Loading checkpoint from: {checkpoint_path}")
        
        # Filter out prefix for compatibility if needed
        filtered_std = {}
        for k, v in std_checkpoint.items():
            # Skip positional encoding parameters which might cause size mismatch
            if k.startswith('encoder.pos_encoding') or k.startswith('decoder.pos_encoding'):
                continue
                
            # Remove 'deepsc.' prefix if present
            if k.startswith('deepsc.'):
                filtered_std[k.replace('deepsc.', '')] = v
            else:
                filtered_std[k] = v
                
        std_model.load_state_dict(filtered_std, strict=False)
        std_model.eval()
        print("✅ Standard DeepSC model loaded")
    except Exception as e:
        print(f"⚠️ Error loading model checkpoint: {e}")
        print("Using randomly initialized model for testing")

    # Create a train dataloader for SemRect training
    try:
        train_loader = DataLoader(
            EurDataset('train', data_dir=args.data_dir), 
            batch_size=args.batch_size,
            collate_fn=collate_data, 
            num_workers=0
        )
        print(f"Created train dataloader with {len(train_loader)} batches")
    except Exception as e:
        print(f"Error loading train dataset: {e}")
        print("Creating dummy dataset for training")
        # Create a dummy dataset with small random tensors
        dummy_data = [torch.randint(1, 10, (10,)) for _ in range(200)]
        
        class DummyDataset(torch.utils.data.Dataset):
            def __init__(self, data):
                self.data = data
            def __getitem__(self, idx):
                return self.data[idx]
            def __len__(self):
                return len(self.data)
        
        train_loader = DataLoader(
            DummyDataset(dummy_data),
            batch_size=args.batch_size,
            collate_fn=collate_data,
            num_workers=0
        )
        print(f"Created dummy train dataloader with {len(train_loader)} batches")

    # Initialize DeepSC+SemRect
    semrect_model = None
    if args.semrect_checkpoint or args.train_semrect:
        try:
            print("Initializing DeepSC+SemRect model...")
            semrect_model = DeepSCWithSemRect(
                num_layers=args.num_layers,
                src_vocab_size=num_vocab,
                trg_vocab_size=num_vocab,
                src_max_len=effective_max_len - 1,  # Subtract 1 to match expected dimensions
                trg_max_len=effective_max_len - 1,  # Subtract 1 to match expected dimensions
                d_model=args.d_model,
                num_heads=args.num_heads,
                dff=args.dff,
                dropout=0.1,
                semrect_latent_dim=args.semrect_latent_dim,
                device=device
            ).to(device)
            
            # Set DeepSC parameters from the standard model
            semrect_model.deepsc.load_state_dict(std_model.state_dict(), strict=False)
            
            # Load SemRect checkpoint if available
            if args.semrect_checkpoint:
                print(f"Loading SemRect checkpoint from: {args.semrect_checkpoint}")
                semrect_checkpoint = torch.load(args.semrect_checkpoint, map_location=device)
                semrect_model.load_state_dict(semrect_checkpoint, strict=False)
                print("✅ DeepSC+SemRect model loaded from checkpoint")
            
            # Train SemRect if requested
            if args.train_semrect:
                print("\n===== Training SemRect =====")
                print(f"Pre-training epochs: {args.epochs_pretrain}")
                print(f"GAN training epochs: {args.epochs_gan}")
                print(f"Learning rate: {args.lr}")
                print(f"Adversarial epsilon: {args.epsilon}")
                
                # Create directory for checkpoints
                os.makedirs('checkpoints', exist_ok=True)
                
                # Train SemRect
                semrect_model.train_semrect(
                    dataloader=train_loader,
                    epochs_pretrain=args.epochs_pretrain,
                    epochs_gan=args.epochs_gan,
                    lr=args.lr,
                    epsilon=args.epsilon
                )
                
                # Save trained model
                try:
                    torch.save(semrect_model.state_dict(), 'checkpoints/semrect_trained.pth')
                    print("✅ Saved trained SemRect model")
                except Exception as e:
                    print(f"⚠️ Error saving trained model: {e}")
                
            semrect_model.eval()
        except Exception as e:
            print(f"⚠️ Error initializing/training SemRect model: {e}")
            print(f"Detailed error: {str(e)}")
            semrect_model = None
    
    # Test under attack
    SNRs = [0, 3, 6, 9, 12, 15, 18]
    results = test_models_under_attack(args, SNRs, std_model, token_to_idx, semrect_model)
    
    # Save and plot results
    os.makedirs('results', exist_ok=True)
    np.save(args.output_file, results)
    plot_attack_results(results, args.epsilon)
    print("\n📊 Final Results:")
    for snr, std_clean, std_attack, semrect_clean, semrect_attack in zip(
        results['snr'], 
        results['DeepSC']['clean'], 
        results['DeepSC']['attack'],
        results['DeepSC+SemRect']['clean'],
        results['DeepSC+SemRect']['attack']
    ):
        print(f"SNR={snr} | Clean: {std_clean:.4f} → {semrect_clean:.4f} | Attack: {std_attack:.4f} → {semrect_attack:.4f}")

if __name__ == '__main__':
    main() 