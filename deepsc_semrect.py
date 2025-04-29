import torch
import torch.nn as nn
from models.transceiver import DeepSC
from SemRect import SemRect
from utils import SNR_to_noise, create_masks, PowerNormalize, Channels

class DeepSCWithSemRect(nn.Module):
    """
    Integration of DeepSC with SemRect for semantic integrity protection.
    SemRect is inserted between the channel decoder and the transformer decoder
    to protect against adversarial perturbations.
    """
    def __init__(self, num_layers, src_vocab_size, trg_vocab_size, src_max_len,
                 trg_max_len, d_model, num_heads, dff, dropout=0.1, 
                 semrect_latent_dim=100, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        super(DeepSCWithSemRect, self).__init__()
        
        # Initialize DeepSC
        self.deepsc = DeepSC(num_layers, src_vocab_size, trg_vocab_size, src_max_len + 1,
                             trg_max_len + 1, d_model, num_heads, dff, dropout)
        
        # Initialize SemRect with matching dimensions
        self.semrect = SemRect(
            latent_dim=semrect_latent_dim,
            output_dim=d_model,  # Match DeepSC's d_model dimension
            seq_len=src_max_len + 1,  # Match sequence length including the extra token
            device=device
        )
        
        # Dimensionality of semantic representation (output of channel decoder)
        self.semantic_dim = d_model
        
        # Channels for simulation
        self.channels = Channels()
        
        # Device
        self.device = device
        
        # Move model to device
        self.to(device)
        
    def forward(self, src, trg, noise_std, channel_type='AWGN', use_semrect=True):
        # Create masks
        src_mask, look_ahead_mask = create_masks(src, trg[:, :-1], 0)
        
        # Encoder and channel processing 
        enc_output = self.deepsc.encoder(src, src_mask)
        channel_enc_output = self.deepsc.channel_encoder(enc_output)
        Tx_sig = PowerNormalize(channel_enc_output)
        
        # Channel simulation
        if channel_type == 'AWGN':
            Rx_sig = self.channels.AWGN(Tx_sig, noise_std)
        elif channel_type == 'Rayleigh':
            Rx_sig = self.channels.Rayleigh(Tx_sig, noise_std)
        elif channel_type == 'Rician':
            Rx_sig = self.channels.Rician(Tx_sig, noise_std)
        else:
            raise ValueError("Please choose from AWGN, Rayleigh, and Rician")
        
        # Channel decoder to get semantic representation
        semantic_repr = self.deepsc.channel_decoder(Rx_sig)
        
        # Apply SemRect only if explicitly requested
        if use_semrect:
            semantic_repr = self.semrect.calibrate(semantic_repr)
            
        # Decoder and output generation
        dec_output = self.deepsc.decoder(trg[:, :-1], semantic_repr, look_ahead_mask, src_mask)
        output = self.deepsc.dense(dec_output)
        
        return output
    
    def train_semrect(self, dataloader, epochs=50, lr=1e-4):
        """
        Train the SemRect GAN on clean semantic representations.
        
        Args:
            dataloader: DataLoader with training data
            epochs: Number of training epochs
            lr: Learning rate
        """
        # Define a semantic encoder function that extracts semantic representations
        # from the input text using DeepSC's encoder and channel encoder
        class SemanticExtractor(nn.Module):
            def __init__(self, deepsc_model):
                super(SemanticExtractor, self).__init__()
                self.encoder = deepsc_model.encoder
                self.channel_encoder = deepsc_model.channel_encoder
                self.channel_decoder = deepsc_model.channel_decoder
            
            def forward(self, x):
                # Create mask (assuming 0 is pad_idx)
                src_mask = (x == 0).unsqueeze(-2).type(torch.FloatTensor).to(x.device)
                
                # Extract semantic representation
                enc_output = self.encoder(x, src_mask)
                channel_enc = self.channel_encoder(enc_output)
                # No channel effects for training SemRect on clean semantics
                semantic_repr = self.channel_decoder(channel_enc)
                
                return semantic_repr
        
        # Create semantic extractor
        semantic_extractor = SemanticExtractor(self.deepsc)
        
        # Train SemRect using the semantic extractor
        self.semrect.train_gan(
            semantic_encoder=semantic_extractor,
            dataloader=dataloader,
            epochs=epochs,
            lr=lr
        )
    
    def test_with_adversarial_attack(self, dataloader, epsilon=0.1, channel_type='AWGN', snr=10):
        """
        Test the model with and without SemRect defense against adversarial attacks.
        
        Args:
            dataloader: DataLoader with test data
            epsilon: FGSM perturbation strength
            channel_type: Type of channel
            snr: Signal-to-noise ratio (dB)
            
        Returns:
            Dictionary of results
        """
        # Convert SNR to noise standard deviation
        noise_std = SNR_to_noise(snr)
        
        # Test with no defense
        results_no_defense = self._evaluate(dataloader, noise_std, channel_type, use_semrect=False)
        
        # Test with SemRect defense
        results_with_defense = self._evaluate(dataloader, noise_std, channel_type, use_semrect=True)
        
        # Test with adversarial attack, no defense
        results_attack_no_defense = self._evaluate_with_attack(
            dataloader, noise_std, channel_type, epsilon, use_semrect=False
        )
        
        # Test with adversarial attack and SemRect defense
        results_attack_with_defense = self._evaluate_with_attack(
            dataloader, noise_std, channel_type, epsilon, use_semrect=True
        )
        
        return {
            'clean_no_defense': results_no_defense,
            'clean_with_defense': results_with_defense,
            'adversarial_no_defense': results_attack_no_defense,
            'adversarial_with_defense': results_attack_with_defense
        }
    
    def _evaluate(self, dataloader, noise_std, channel_type, use_semrect=True):
        """Evaluate model accuracy without attacks"""
        self.eval()
        total = 0
        correct = 0
        
        with torch.no_grad():
            for src, trg in dataloader:
                src = src.to(self.device)
                trg = trg.to(self.device)
                
                output = self(src, trg, noise_std, channel_type, use_semrect)
                pred = output.argmax(dim=-1)
                
                # Compare with actual target tokens
                target = trg[:, 1:]  # Shift by 1 since output doesn't include the first token
                mask = (target != 0)  # Assuming 0 is pad_idx
                
                correct += ((pred == target) * mask).sum().item()
                total += mask.sum().item()
        
        return correct / total if total > 0 else 0
    
    def _evaluate_with_attack(self, dataloader, noise_std, channel_type, epsilon, use_semrect=True):
        """Evaluate model accuracy with adversarial attack"""
        self.eval()
        total = 0
        correct = 0
        
        for src, trg in dataloader:
            src = src.to(self.device)
            trg = trg.to(self.device)
            
            # Create masks
            src_mask, _ = create_masks(src, trg[:, :-1], 0)  # Assuming 0 is pad_idx
            
            # Enable gradients for attack
            src.requires_grad = True
            
            # Forward through encoder and channel encoder
            enc_output = self.deepsc.encoder(src, src_mask)
            channel_enc_output = self.deepsc.channel_encoder(enc_output)
            semantic_repr = self.deepsc.channel_decoder(channel_enc_output)
            
            # Get output of a single forward pass to compute gradients
            # We use a dummy target here just to compute loss
            dummy_trg = trg[:, :-1]
            dummy_out = self.deepsc.decoder(dummy_trg, semantic_repr, None, src_mask)
            dummy_logits = self.deepsc.dense(dummy_out)
            
            # Compute loss for gradient calculation
            criterion = nn.CrossEntropyLoss()
            dummy_loss = criterion(dummy_logits.reshape(-1, dummy_logits.size(-1)), 
                                  trg[:, 1:].reshape(-1))
            
            # Compute gradients
            dummy_loss.backward()
            
            # Create adversarial perturbation
            with torch.no_grad():
                # Get gradient w.r.t. semantic representation
                semantic_grad = semantic_repr.grad.sign()
                
                # Create adversarial example
                adv_semantic = semantic_repr + epsilon * semantic_grad
                
                # Run through model with adversarial semantic
                if use_semrect:
                    # Apply SemRect defense
                    adv_semantic = self.semrect.calibrate(adv_semantic)
                
                # Decoder with adversarial semantic
                dec_output = self.deepsc.decoder(trg[:, :-1], adv_semantic, None, src_mask)
                output = self.deepsc.dense(dec_output)
                
                # Compute accuracy
                pred = output.argmax(dim=-1)
                target = trg[:, 1:]  # Shift by 1
                mask = (target != 0)  # Assuming 0 is pad_idx
                
                correct += ((pred == target) * mask).sum().item()
                total += mask.sum().item()
        
        return correct / total if total > 0 else 0


# Example usage
if __name__ == "__main__":
    import argparse
    from torch.utils.data import DataLoader
    from dataset import EurDataset, collate_data
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint-path', default='checkpoints/deepsc-semrect', type=str)
    parser.add_argument('--channel', default='AWGN', type=str, help='Please choose AWGN, Rayleigh, or Rician')
    parser.add_argument('--d-model', default=128, type=int)
    parser.add_argument('--dff', default=512, type=int)
    parser.add_argument('--num-layers', default=4, type=int)
    parser.add_argument('--num-heads', default=8, type=int)
    parser.add_argument('--batch-size', default=128, type=int)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--semrect-epochs', default=30, type=int)
    parser.add_argument('--snr', default=10, type=float, help='Signal-to-noise ratio in dB')
    parser.add_argument('--epsilon', default=0.1, type=float, help='FGSM attack strength')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create dataset
    train_dataset = EurDataset('train')
    test_dataset = EurDataset('test')
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, 
                             num_workers=0, pin_memory=True, collate_fn=collate_data)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                            num_workers=0, pin_memory=True, collate_fn=collate_data)
    
    # Example vocab size (should match your actual vocab size)
    vocab_size = 10000  # Replace with actual vocab size
    
    # Create model
    model = DeepSCWithSemRect(
        num_layers=args.num_layers,
        src_vocab_size=vocab_size,
        trg_vocab_size=vocab_size,
        src_max_len=30,  # Adjust as needed
        trg_max_len=30,  # Adjust as needed
        d_model=args.d_model,
        num_heads=args.num_heads,
        dff=args.dff,
        dropout=0.1,
        semrect_latent_dim=100,
        device=device
    )
    
    # Train SemRect
    print("Training SemRect...")
    model.train_semrect(train_loader, epochs=args.semrect_epochs, lr=1e-4)
    
    # Test with adversarial attacks
    print("\nTesting with adversarial attacks...")
    results = model.test_with_adversarial_attack(
        test_loader,
        epsilon=args.epsilon,
        channel_type=args.channel,
        snr=args.snr
    )
    
    # Print results
    print("\nResults:")
    print(f"Clean (No Defense): {results['clean_no_defense']:.4f}")
    print(f"Clean (With SemRect): {results['clean_with_defense']:.4f}")
    print(f"Adversarial (No Defense): {results['adversarial_no_defense']:.4f}")
    print(f"Adversarial (With SemRect): {results['adversarial_with_defense']:.4f}")
    
    # Calculate improvement
    adv_improvement = ((results['adversarial_with_defense'] - results['adversarial_no_defense']) / 
                      results['adversarial_no_defense']) * 100 if results['adversarial_no_defense'] > 0 else float('inf')
    
    print(f"\nSemRect improves adversarial robustness by {adv_improvement:.2f}%")