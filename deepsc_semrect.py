import torch
import torch.nn as nn
from models.transceiver import DeepSC
from SemRect import SemRect
from SemRectFIM import SemRectFIM
from utils import SNR_to_noise, create_masks, PowerNormalize, Channels, BleuScore
import numpy as np

class DeepSCWithSemRect(nn.Module):
    """
    Integration of DeepSC with SemRect for semantic integrity protection.
    SemRect is inserted between the channel decoder and the transformer decoder
    to protect against adversarial perturbations using trained GAN-based signatures.
    
    This implementation follows the paper's description where SemRect uses a
    Defense-GAN approach to generate semantic signatures that calibrate adversarial perturbations.
    """
    def __init__(self, num_layers, src_vocab_size, trg_vocab_size, src_max_len,
                 trg_max_len, d_model, num_heads, dff, dropout=0.1, 
                 semrect_latent_dim=100, protection_type='semrect',
                 device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        super(DeepSCWithSemRect, self).__init__()
        
        # Initialize DeepSC
        self.deepsc = DeepSC(num_layers, src_vocab_size, trg_vocab_size, src_max_len + 1,
                             trg_max_len + 1, d_model, num_heads, dff, dropout)

        # Choose protection type
        self.protection_type = protection_type
        
        # Initialize semantic protection module based on type
        if protection_type == 'semrectfim':
            self.semantic_protection = SemRectFIM(
                latent_dim=semrect_latent_dim,
                output_dim=d_model,
                seq_len=src_max_len + 1,
                device=device
            )
        else:  # default to semrect
            self.semantic_protection = SemRect(
                latent_dim=semrect_latent_dim,
                output_dim=d_model,
                seq_len=src_max_len + 1,
                device=device
            )

        # Device
        self.device = device
        self.to(device)
        
    def get_semantic_encoder(self):
        """
        Returns a callable that extracts semantic representations from input texts.
        This is used for training SemRect.
        """
        class SemanticExtractor(nn.Module):
            def __init__(self, deepsc_model, device):
                super(SemanticExtractor, self).__init__()
                self.encoder = deepsc_model.encoder
                self.channel_encoder = deepsc_model.channel_encoder
                self.channel_decoder = deepsc_model.channel_decoder
                self.device = device

            def forward(self, x):
                # Create source mask
                src_mask = (x == 0).unsqueeze(-2).type(torch.FloatTensor).to(x.device)
                # Extract semantic representation
                enc_output = self.encoder(x, src_mask)
                channel_enc = self.channel_encoder(enc_output)
                # No channel simulation during training
                semantic_repr = self.channel_decoder(channel_enc)  
                return semantic_repr
                
        return SemanticExtractor(self.deepsc, self.device)

    def forward(self, src, trg, noise_std, channel_type='AWGN', use_protection=True, snr=None):
        # Create masks
        src_mask, look_ahead_mask = create_masks(src, trg[:, :-1], 0)

        # Encoder
        enc_output = self.deepsc.encoder(src, src_mask)

        # Channel encoder
        channel_enc_output = self.deepsc.channel_encoder(enc_output)
        Tx_sig = PowerNormalize(channel_enc_output)

        # Simulate channel
        if channel_type == 'AWGN':
            Rx_sig = Channels().AWGN(Tx_sig, noise_std)
        elif channel_type == 'Rayleigh':
            Rx_sig = Channels().Rayleigh(Tx_sig, noise_std)
        elif channel_type == 'Rician':
            Rx_sig = Channels().Rician(Tx_sig, noise_std)
        else:
            raise ValueError("Channel type not supported")

        # Channel decoder -> semantic representation
        semantic_repr = self.deepsc.channel_decoder(Rx_sig)

        # Apply semantic protection if enabled
        if use_protection:
            # For SemRectFIM, we need to convert noise_std to SNR for better adaptation
            if self.protection_type == 'semrectfim':
                # If SNR not provided, estimate it from noise_std
                if snr is None:
                    # Simple approximation: SNR(dB) = 10*log10(1/noise_std^2) for normalized signal power
                    snr_value = 10 * torch.log10(1 / (noise_std ** 2 + 1e-10))
                    snr = snr_value * torch.ones((src.size(0), 1), device=self.device)
                
                semantic_repr = self.semantic_protection.calibrate(semantic_repr, snr)
            else:
                semantic_repr = self.semantic_protection.calibrate(semantic_repr)

        # Decoder and output
        dec_output = self.deepsc.decoder(trg[:, :-1], semantic_repr, look_ahead_mask, src_mask)
        output = self.deepsc.dense(dec_output)
        return output

    def generate_adversarial(self, src, trg, noise_std, channel_type, epsilon):
        """
        Generate adversarial examples using FGSM (Fast Gradient Sign Method)
        as described in the SemRect paper.
        """
        self.eval()
        
        # Create masks
        src_mask, _ = create_masks(src, trg[:, :-1], 0)
        
        # Forward pass through encoder
        enc_output = self.deepsc.encoder(src, src_mask)
        
        # Channel encoder
        channel_enc = self.deepsc.channel_encoder(enc_output)
        channel_enc_clean = PowerNormalize(channel_enc.clone())  # Save clean version
        channel_enc = PowerNormalize(channel_enc)
        
        # Set requires_grad to true for attack
        channel_enc.requires_grad_(True)
        
        # Forward through channel
        channels = Channels()
        if channel_type == 'AWGN':
            Rx_sig = channels.AWGN(channel_enc, noise_std)
        elif channel_type == 'Rayleigh':
            Rx_sig = channels.Rayleigh(channel_enc, noise_std)
        elif channel_type == 'Rician':
            Rx_sig = channels.Rician(channel_enc, noise_std)
        else:
            # Default to AWGN if unknown channel type
            Rx_sig = channels.AWGN(channel_enc, noise_std)
        
        # Channel decoder
        semantic_repr = self.deepsc.channel_decoder(Rx_sig)
        
        # Forward through decoder 
        dec_output = self.deepsc.decoder(trg[:, :-1], semantic_repr, None, src_mask)
        output = self.deepsc.dense(dec_output)
        
        # Compute loss
        criterion = nn.CrossEntropyLoss()
        loss = criterion(output.reshape(-1, output.size(-1)), trg[:, 1:].reshape(-1))
        
        # Get gradient sign for FGSM attack
        loss.backward()
        
        # Create adversarial example using FGSM
        with torch.no_grad():
            if channel_enc.grad is not None:
                adv_channel = channel_enc + epsilon * channel_enc.grad.sign()
            else:
                # If no gradient, use the original channel encoding with small random noise
                adv_channel = channel_enc + epsilon * torch.randn_like(channel_enc)
                
            # Forward through channel with adversarial input
            if channel_type == 'AWGN':
                adv_Rx_sig = channels.AWGN(adv_channel, noise_std)
            elif channel_type == 'Rayleigh':
                adv_Rx_sig = channels.Rayleigh(adv_channel, noise_std)
            elif channel_type == 'Rician':
                adv_Rx_sig = channels.Rician(adv_channel, noise_std)
            else:
                # Default to AWGN if unknown channel type
                adv_Rx_sig = channels.AWGN(adv_channel, noise_std)
            
            # Get adversarial semantic representation
            adv_semantic = self.deepsc.channel_decoder(adv_Rx_sig)
            
        return adv_semantic, src_mask
        
    def train_protection_module(self, dataloader, epochs=None, epochs_pretrain=5, epochs_gan=10, 
                               lr=1e-4, epsilon=0.1, snr_range=(0, 30)):
        """
        Train the semantic protection module (SemRect or SemRectFIM).
        
        Args:
            dataloader: DataLoader providing text data
            epochs: If provided, will be used for both pretrain and GAN phases (overrides epochs_pretrain and epochs_gan)
            epochs_pretrain: Number of pre-training epochs
            epochs_gan: Number of GAN training epochs
            lr: Learning rate
            epsilon: FGSM attack strength
            snr_range: Range of SNR values for SemRectFIM training
        """
        print(f"Training {self.protection_type.upper()}...")
        
        # If epochs is provided, use it for both phases
        if epochs is not None:
            epochs_pretrain = epochs
            epochs_gan = epochs
        
        # Freeze DeepSC parameters during protection module training
        for param in self.deepsc.parameters():
            param.requires_grad = False
            
        # Create semantic encoder for training
        semantic_encoder = self.get_semantic_encoder()
        
        # Training based on protection type
        if self.protection_type == 'semrectfim':
            # Phase 1: Pre-train generator on clean data with SNR awareness
            self.semantic_protection.pretrain_generator(
                semantic_encoder, 
                dataloader, 
                epochs=epochs_pretrain, 
                lr=lr, 
                snr_range=snr_range
            )
            
            # Phase 2: Train GAN with adversarial examples and SNR awareness
            self.semantic_protection.train_gan(
                semantic_encoder, 
                dataloader, 
                epochs=epochs_gan, 
                lr=lr, 
                epsilon=epsilon, 
                snr_range=snr_range
            )
        else:
            # Standard SemRect training
            # Phase 1: Pre-train generator on clean data
            self.semantic_protection.pretrain_generator(
                semantic_encoder, 
                dataloader, 
                epochs=epochs_pretrain, 
                lr=lr
            )
            
            # Phase 2: Train GAN with adversarial examples
            self.semantic_protection.train_gan(
                semantic_encoder, 
                dataloader, 
                epochs=epochs_gan, 
                lr=lr, 
                epsilon=epsilon
            )
        
        # Unfreeze DeepSC parameters
        for param in self.deepsc.parameters():
            param.requires_grad = True
            
        print(f"{self.protection_type.upper()} training complete!")

    def test_with_adversarial_attack(self, dataloader, epsilon=0.1, channel_type='AWGN', snr=10):
        """
        Test model robustness under FGSM adversarial attacks.
        Returns attack success rate and BLEU scores.
        """
        noise_std = SNR_to_noise(snr)
        total_samples = 0
        attack_success = 0
        
        # Create a SeqtoText converter
        # Note: This needs to be updated with your actual token_to_idx dictionary
        token_to_idx = getattr(self, 'token_to_idx', {'<PAD>': 0, '<END>': 2})
        bleu_score_1gram = BleuScore(1, 0, 0, 0)
        
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
                
        StoT = SeqtoText(token_to_idx, token_to_idx.get("<END>", 2))
        
        # For BLEU score calculation
        all_preds_clean = []
        all_preds_adv = []
        all_preds_defended = []
        all_targets = []
        
        self.eval()
        with torch.no_grad():
            for batch in dataloader:
                # Handle different return formats
                if isinstance(batch, tuple) and len(batch) == 2:
                    src, trg = batch
                else:
                    src = batch
                    trg = src
                
                src, trg = src.to(self.device), trg.to(self.device)
                
                # Generate adversarial examples using FGSM
                adv_semantic, src_mask = self.generate_adversarial(src, trg, noise_std, channel_type, epsilon)
                
                # Convert noise_std to SNR tensor for SemRectFIM
                snr_tensor = torch.tensor([[snr]], device=self.device).expand(src.size(0), 1)
                
                # Get predictions on clean examples
                output_clean = self(src, trg, noise_std, channel_type, use_protection=False)
                pred_clean = output_clean.argmax(dim=-1)
                
                # Get predictions on adversarial examples without defense
                dec_output_adv = self.deepsc.decoder(trg[:, :-1], adv_semantic, None, src_mask)
                output_adv = self.deepsc.dense(dec_output_adv)
                pred_adv = output_adv.argmax(dim=-1)
                
                # Get predictions on adversarial examples with semantic protection
                if self.protection_type == 'semrectfim':
                    calibrated_semantic = self.semantic_protection.calibrate(adv_semantic, snr_tensor)
                else:
                    calibrated_semantic = self.semantic_protection.calibrate(adv_semantic)
                    
                dec_output_def = self.deepsc.decoder(trg[:, :-1], calibrated_semantic, None, src_mask)
                output_def = self.deepsc.dense(dec_output_def)
                pred_def = output_def.argmax(dim=-1)
                
                # Calculate attack success rate (when prediction changes due to attack)
                target = trg[:, 1:]
                mask = (target != 0)
                attack_changed = (pred_clean != pred_adv) & mask
                attack_success += attack_changed.sum().item()
                total_samples += mask.sum().item()
                
                # Store predictions for BLEU score
                all_preds_clean.extend(pred_clean.cpu().numpy().tolist())
                all_preds_adv.extend(pred_adv.cpu().numpy().tolist())
                all_preds_defended.extend(pred_def.cpu().numpy().tolist())
                all_targets.extend(target.cpu().numpy().tolist())
        
        # Calculate attack success rate
        asr = attack_success / total_samples if total_samples > 0 else 0
        
        # Calculate BLEU scores
        pred_texts_clean = list(map(StoT.sequence_to_text, all_preds_clean))
        pred_texts_adv = list(map(StoT.sequence_to_text, all_preds_adv))
        pred_texts_def = list(map(StoT.sequence_to_text, all_preds_defended))
        target_texts = list(map(StoT.sequence_to_text, all_targets))
        
        bleu_clean = np.mean(bleu_score_1gram.compute_blue_score(pred_texts_clean, target_texts))
        bleu_adv = np.mean(bleu_score_1gram.compute_blue_score(pred_texts_adv, target_texts))
        bleu_def = np.mean(bleu_score_1gram.compute_blue_score(pred_texts_def, target_texts))
        
        results = {
            'protection_type': self.protection_type,
            'snr': snr,
            'attack_success_rate': asr,
            'bleu_clean': bleu_clean,
            'bleu_adversarial': bleu_adv,
            'bleu_defended': bleu_def,
            'defense_effectiveness': (bleu_def - bleu_adv) / (bleu_clean - bleu_adv) if (bleu_clean - bleu_adv) != 0 else 0
        }
        
        return results