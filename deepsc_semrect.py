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
            output_dim=d_model,
            seq_len=src_max_len + 1,
            device=device
        )

        # Device
        self.device = device
        self.to(device)

    def forward(self, src, trg, noise_std, channel_type='AWGN', use_semrect=True):
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

        # Apply SemRect if enabled
        if use_semrect:
            semantic_repr = self.semrect.calibrate(semantic_repr)

        # Decoder and output
        dec_output = self.deepsc.decoder(trg[:, :-1], semantic_repr, look_ahead_mask, src_mask)
        output = self.deepsc.dense(dec_output)
        return output

    def train_semrect(self, dataloader, epochs=50, lr=1e-4):
        """
        Train SemRect on clean semantic representations (no channel effects).
        Uses WGAN-GP style training for stability.
        """
        class SemanticExtractor(nn.Module):
            def __init__(self, deepsc_model):
                super(SemanticExtractor, self).__init__()
                self.encoder = deepsc_model.encoder
                self.channel_encoder = deepsc_model.channel_encoder
                self.channel_decoder = deepsc_model.channel_decoder

            def forward(self, x):
                src_mask = (x == 0).unsqueeze(-2).type(torch.FloatTensor).to(x.device)
                enc_output = self.encoder(x, src_mask)
                channel_enc = self.channel_encoder(enc_output)
                semantic_repr = self.channel_decoder(channel_enc)  # No channel effects
                return semantic_repr
        
        def compute_gradient_penalty(D, real_samples, fake_samples):
            """Compute WGAN-GP gradient penalty"""
            # Random weight for interpolation
            alpha = torch.randn(real_samples.size(0), 1, 1, device=self.device)
            
            # Interpolated samples
            interpolates = alpha * real_samples + (1 - alpha) * fake_samples
            interpolates.requires_grad_(True)
            
            # Get discriminator output for interpolated samples
            d_interpolates = D(interpolates)
            
            # Compute gradients
            fake = torch.ones(d_interpolates.size(), device=self.device)
            gradients = torch.autograd.grad(
                outputs=d_interpolates,
                inputs=interpolates,
                grad_outputs=fake,
                create_graph=True,
                retain_graph=True,
                only_inputs=True,
            )[0]
            
            # Compute gradient penalty
            gradients = gradients.reshape(gradients.size(0), -1)
            gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
            return gradient_penalty

        # Freeze DeepSC
        for param in self.deepsc.parameters():
            param.requires_grad = False

        semantic_extractor = SemanticExtractor(self.deepsc)
        
        # Optimizers with different learning rates
        opt_g = torch.optim.Adam(self.semrect.G.parameters(), lr=lr, betas=(0.5, 0.9))
        opt_d = torch.optim.Adam(self.semrect.D.parameters(), lr=lr*2, betas=(0.5, 0.9))
        
        # Training loop
        for epoch in range(epochs):
            g_losses = []
            d_losses = []
            
            for i, texts in enumerate(dataloader):
                if not isinstance(texts, torch.Tensor):
                    texts = texts[0]  # Handle case where dataloader might return a tuple/list
                    
                texts = texts.to(self.device)
                with torch.no_grad():
                    # Get real semantic representation 
                    real_repr = semantic_extractor(texts)

                batch_size = real_repr.size(0)
                
                # Train discriminator more frequently for stability
                for _ in range(2):  # Multiple D updates per G update
                    opt_d.zero_grad()
                    
                    # Real samples
                    d_real = self.semrect.D(real_repr)
                    d_real_loss = -torch.mean(d_real)  # WGAN loss
                    
                    # Fake samples
                    z = torch.randn(batch_size, self.semrect.latent_dim, device=self.device)
                    fake_repr = self.semrect.G(real_repr, z)
                    d_fake = self.semrect.D(fake_repr.detach())
                    d_fake_loss = torch.mean(d_fake)  # WGAN loss
                    
                    # Gradient penalty
                    gp = compute_gradient_penalty(self.semrect.D, real_repr, fake_repr.detach())
                    
                    # Total D loss
                    d_loss = d_real_loss + d_fake_loss + 10.0 * gp
                    d_loss.backward()
                    opt_d.step()
                
                # Train generator
                opt_g.zero_grad()
                fake_repr = self.semrect.G(real_repr, z)
                g_loss_adv = -torch.mean(self.semrect.D(fake_repr))  # WGAN adversarial loss
                
                # Reconstruction loss
                g_loss_rec = torch.nn.functional.mse_loss(fake_repr, real_repr)
                
                # Combined loss
                g_loss = g_loss_adv + 10.0 * g_loss_rec
                g_loss.backward()
                opt_g.step()
                
                # Track losses
                g_losses.append(g_loss.item())
                d_losses.append(d_loss.item())
                
                # Print progress for larger batches
                if i % 10 == 0:
                    print(f"Batch {i}/{len(dataloader)}: D={d_loss.item():.4f}, G={g_loss.item():.4f}")
            
            # Print epoch summary
            avg_d_loss = sum(d_losses) / len(d_losses)
            avg_g_loss = sum(g_losses) / len(g_losses)
            print(f"Epoch {epoch+1}/{epochs} D_loss={avg_d_loss:.4f} G_loss={avg_g_loss:.4f}")
            
            # Save checkpoint periodically
            if (epoch+1) % 5 == 0:
                torch.save(self.state_dict(), f"checkpoints/semrect_epoch_{epoch+1}.pth")

        # Optionally unfreeze DeepSC after training
        for param in self.deepsc.parameters():
            param.requires_grad = True

    def test_with_adversarial_attack(self, dataloader, epsilon=0.1, channel_type='AWGN', snr=10):
        """
        Test model robustness under FGSM adversarial attacks.
        """
        noise_std = SNR_to_noise(snr)
        results = {
            'clean_no_defense': self._evaluate(dataloader, noise_std, channel_type, use_semrect=False),
            'clean_with_defense': self._evaluate(dataloader, noise_std, channel_type, use_semrect=True),
            'adversarial_no_defense': self._evaluate_with_attack(dataloader, noise_std, channel_type, epsilon, use_semrect=False),
            'adversarial_with_defense': self._evaluate_with_attack(dataloader, noise_std, channel_type, epsilon, use_semrect=True)
        }
        return results

    def _evaluate(self, dataloader, noise_std, channel_type, use_semrect=True):
        self.eval()
        total = correct = 0
        with torch.no_grad():
            for src, trg in dataloader:
                src, trg = src.to(self.device), trg.to(self.device)
                output = self(src, trg, noise_std, channel_type, use_semrect)
                pred = output.argmax(dim=-1)
                target = trg[:, 1:]
                mask = (target != 0)
                correct += ((pred == target) * mask).sum().item()
                total += mask.sum().item()
        return correct / total if total > 0 else 0

    def _evaluate_with_attack(self, dataloader, noise_std, channel_type, epsilon, use_semrect=True):
        self.eval()
        total = correct = 0
        criterion = nn.CrossEntropyLoss()

        for src, trg in dataloader:
            src, trg = src.to(self.device), trg.to(self.device)
            src.requires_grad = True

            # Forward pass to compute loss for gradient
            src_mask, _ = create_masks(src, trg[:, :-1], 0)
            enc_output = self.deepsc.encoder(src, src_mask)
            channel_enc = self.deepsc.channel_encoder(enc_output)
            semantic_repr = self.deepsc.channel_decoder(channel_enc)

            dummy_trg = trg[:, :-1]
            dec_output = self.deepsc.decoder(dummy_trg, semantic_repr, None, src_mask)
            logits = self.deepsc.dense(dec_output)
            loss = criterion(logits.view(-1, logits.size(-1)), trg[:, 1:].view(-1))

            # Backward to get gradients
            loss.backward()

            # Generate adversarial example
            with torch.no_grad():
                adv_semantic = semantic_repr + epsilon * semantic_repr.grad.sign()
                if use_semrect:
                    adv_semantic = self.semrect.calibrate(adv_semantic)

                dec_output_adv = self.deepsc.decoder(trg[:, :-1], adv_semantic, None, src_mask)
                output_adv = self.deepsc.dense(dec_output_adv)
                pred = output_adv.argmax(dim=-1)
                target = trg[:, 1:]
                mask = (target != 0)
                correct += ((pred == target) * mask).sum().item()
                total += mask.sum().item()

        return correct / total if total > 0 else 0