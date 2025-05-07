import torch
import torch.nn as nn
import torch.optim as optim
from SemRect import SemRect, SemanticGenerator, SemanticDiscriminator

class FeatureImportanceMapping(nn.Module):
    """
    Feature Importance Mapping (FIM) module for SNR-aware feature adaptation.
    Dynamically adjusts semantic features based on channel conditions (SNR).
    """
    def __init__(self, feature_dim, hidden_dim=256, dropout=0.1):
        super(FeatureImportanceMapping, self).__init__()
        
        # SNR embedding network
        self.snr_embedding = nn.Sequential(
            nn.Linear(1, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, hidden_dim // 2),
            nn.ReLU()
        )
        
        # Feature importance network
        self.importance_net = nn.Sequential(
            nn.Linear(feature_dim + hidden_dim // 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, feature_dim),
            nn.Sigmoid()  # Outputs importance weights between 0 and 1
        )
    
    def forward(self, features, snr):
        """
        Generate importance weights for features based on SNR conditions.
        
        Args:
            features: Semantic features [batch_size, seq_len, feature_dim]
            snr: Signal-to-Noise Ratio value [batch_size, 1]
        
        Returns:
            Weighted features based on importance
        """
        batch_size, seq_len, feature_dim = features.shape
        
        # Embed SNR
        snr_embedded = self.snr_embedding(snr)  # [batch_size, hidden_dim//2]
        
        # Expand SNR embedding to match feature dimensions
        snr_expanded = snr_embedded.unsqueeze(1).expand(-1, seq_len, -1)  # [batch_size, seq_len, hidden_dim//2]
        
        # Flatten features for easier processing
        features_reshaped = features.reshape(batch_size * seq_len, feature_dim)
        snr_expanded_reshaped = snr_expanded.reshape(batch_size * seq_len, -1)
        
        # Concatenate features with SNR embedding
        combined = torch.cat([features_reshaped, snr_expanded_reshaped], dim=1)
        
        # Generate importance weights
        importance_weights = self.importance_net(combined)  # [batch_size*seq_len, feature_dim]
        importance_weights = importance_weights.reshape(batch_size, seq_len, feature_dim)
        
        # Apply importance weights to features
        weighted_features = features * importance_weights
        
        return weighted_features, importance_weights

class SNRAwareSemanticGenerator(SemanticGenerator):
    """
    Extends the Semantic Generator with SNR-awareness capabilities.
    """
    def __init__(self, latent_dim=100, output_dim=128, seq_len=31):
        super().__init__(latent_dim, output_dim, seq_len)
        
        # Add SNR input to the latent vector
        self.snr_embedding = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, 64)
        )
        
        # Redefine main network to accommodate SNR embedding
        self.main = nn.Sequential(
            # Expand combined latent dim to hidden features
            nn.Linear(latent_dim + 64, 512),
            nn.LayerNorm(512),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Project to sequence length × feature dimension
            nn.Linear(512, 1024),
            nn.LayerNorm(1024),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Final projection to output dimension
            nn.Linear(1024, seq_len * output_dim),
        )
    
    def forward(self, z, snr):
        """
        Forward pass to generate semantic representation from random vector and SNR
        
        Args:
            z: Random latent vector [batch_size, latent_dim]
            snr: Signal-to-Noise Ratio [batch_size, 1]
        Returns:
            Semantic representation [batch_size, seq_len, output_dim]
        """
        batch_size = z.size(0)
        
        # Embed SNR
        snr_embedded = self.snr_embedding(snr)
        
        # Concatenate latent vector with SNR embedding
        z_combined = torch.cat([z, snr_embedded], dim=1)
        
        # Generate semantic representation
        x = self.main(z_combined)
        x = x.view(batch_size, self.seq_len, self.output_dim)
        return x

class SemRectFIM(SemRect):
    """
    SemRectFIM: An enhanced version of SemRect that integrates Feature Importance
    Mapping for SNR-aware semantic signature generation and calibration.
    
    This implementation extends SemRect by incorporating SNR information to:
    1. Generate adaptive semantic signatures based on channel conditions
    2. Apply dynamic calibration strength based on SNR
    3. Focus on features most resilient to specific channel conditions
    """
    def __init__(self, latent_dim=100, output_dim=128, seq_len=31,
                 device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        super().__init__(latent_dim, output_dim, seq_len, device)
        
        # Replace standard generator with SNR-aware generator
        self.G = SNRAwareSemanticGenerator(latent_dim, output_dim, seq_len).to(device)
        
        # Add Feature Importance Mapping module
        self.fim = FeatureImportanceMapping(output_dim).to(device)
        
        # SNR-aware calibration strength adaptation
        self.calibration_adapter = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()  # Output between 0 and 1
        ).to(device)
        
        # Additional loss for FIM
        self.fim_loss = nn.MSELoss()
        
        self.to(device)
    
    def calibrate(self, h_adv, snr, base_signature_scale=0.1):
        """
        SNR-aware calibration of potentially corrupted semantic representation.
        
        Args:
            h_adv: Input semantic representation [batch, seq_len, output_dim]
            snr: Signal-to-Noise Ratio [batch, 1]
            base_signature_scale: Base scaling factor for the signature
        """
        batch_size = h_adv.size(0)
        
        # Apply FIM to highlight SNR-resilient features
        weighted_h_adv, importance_weights = self.fim(h_adv, snr)
        
        # Generate random vector (z) for semantic signature
        z = torch.randn(batch_size, self.latent_dim, device=self.device)
        
        # Generate SNR-aware semantic signature
        with torch.no_grad():
            signature = self.G(z, snr)
            
        # Apply FIM to signature as well to focus on important features
        weighted_signature, _ = self.fim(signature, snr)
        
        # Dynamically adjust signature scale based on SNR
        adaptive_scale = self.calibration_adapter(snr) * base_signature_scale * 2
        
        # Apply the weighted signature to calibrate adversarial input with adaptive scaling
        calibrated = h_adv + adaptive_scale * weighted_signature
        
        return calibrated
    
    def pretrain_generator(self, semantic_encoder, dataloader, epochs=5, lr=1e-4, snr_range=(0, 30)):
        """
        Pre-train generator on clean semantic representations with varying SNR.
        
        Args:
            semantic_encoder: Function/module to extract semantic representations
            dataloader: Dataloader providing clean text inputs
            epochs: Number of pre-training epochs
            lr: Learning rate
            snr_range: Range of SNR values to train on (min, max)
        """
        print("Pre-training SemRectFIM generator on clean semantic data with SNR awareness...")
        
        # Set models to correct modes
        self.G.train()
        self.fim.train()
        if hasattr(semantic_encoder, 'eval'):
            semantic_encoder.eval()
            
        # Optimizer for generator and FIM
        optimizer = optim.Adam(
            list(self.G.parameters()) + list(self.fim.parameters()),
            lr=lr, 
            betas=(0.5, 0.999)
        )
        
        # Training loop
        for epoch in range(epochs):
            running_loss = 0.0
            batch_count = 0
            
            for batch in dataloader:
                # Handle different batch formats
                if isinstance(batch, tuple) and len(batch) == 2:
                    texts = batch[0]  # Assume source texts
                else:
                    texts = batch
                    
                if not isinstance(texts, torch.Tensor):
                    continue
                    
                # Move to device
                texts = texts.to(self.device)
                batch_size = texts.size(0)
                batch_count += 1
                
                # Get clean semantic representation
                with torch.no_grad():
                    clean_repr = semantic_encoder(texts)
                
                # Generate random SNR values within the specified range
                snr = torch.FloatTensor(batch_size, 1).uniform_(snr_range[0], snr_range[1]).to(self.device)
                
                # Generate semantic representation from random vector and SNR
                z = torch.randn(batch_size, self.latent_dim, device=self.device)
                fake_repr = self.G(z, snr)
                
                # Apply FIM to both clean and generated representations
                weighted_clean, _ = self.fim(clean_repr, snr)
                weighted_fake, importance_weights = self.fim(fake_repr, snr)
                
                # Reconstruction loss
                recon_loss = self.recon_loss(weighted_fake, weighted_clean)
                
                # Feature importance regularization (encourage sparse but effective weighting)
                importance_reg = torch.mean(torch.abs(importance_weights))
                
                # Combined loss
                loss = recon_loss + 0.01 * importance_reg
                
                # Update generator and FIM
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                
            # Print epoch stats
            avg_loss = running_loss / max(batch_count, 1)
            print(f"Pre-training epoch {epoch+1}/{epochs}, Avg Loss: {avg_loss:.4f}")
    
    def train_gan(self, semantic_encoder, dataloader, epochs=10, lr=1e-4, epsilon=0.1, snr_range=(0, 30)):
        """
        Train SemRectFIM using the GAN approach with SNR awareness.
        
        Args:
            semantic_encoder: Function/module to extract semantic representations
            dataloader: Dataloader providing clean text inputs
            epochs: Number of training epochs
            lr: Learning rate
            epsilon: FGSM attack strength
            snr_range: Range of SNR values to train on (min, max)
        """
        print("Training SemRectFIM GAN with adversarial examples and SNR awareness...")
        
        # Set models to training mode
        self.G.train()
        self.D.train()
        self.fim.train()
        
        # Optimizers
        opt_G = optim.Adam(list(self.G.parameters()) + list(self.fim.parameters()), 
                          lr=lr, betas=(0.5, 0.999))
        opt_D = optim.Adam(self.D.parameters(), lr=lr, betas=(0.5, 0.999))
        
        # For generating adversarial examples with SNR awareness
        def generate_adversarial(clean_repr, snr, epsilon=0.1):
            # Apply FIM to highlight important features
            weighted_clean, _ = self.fim(clean_repr, snr)
            
            # Simple FGSM implementation with FIM-weighted features
            perturbed = weighted_clean.clone().detach().requires_grad_(True)
            
            # Forward through discriminator
            pred = self.D(perturbed)
            
            # Compute loss to maximize discriminator output
            target = torch.zeros_like(pred)
            loss = self.adv_loss(pred, target)
            
            # Get gradients
            loss.backward()
            
            # FGSM attack
            if perturbed.grad is not None:
                # Scale attack strength by feature importance
                _, importance_weights = self.fim(clean_repr, snr)
                adv_grad = perturbed.grad * importance_weights  # Focus attack on important features
                adv_repr = clean_repr + epsilon * adv_grad.sign()
            else:
                adv_repr = clean_repr
                
            return adv_repr.detach()
        
        # Training loop
        for epoch in range(epochs):
            g_losses, d_losses = [], []
            
            for batch in dataloader:
                # Handle different batch formats
                if isinstance(batch, tuple) and len(batch) == 2:
                    texts = batch[0]  # Assume source texts
                else:
                    texts = batch
                    
                if not isinstance(texts, torch.Tensor):
                    continue
                    
                # Move to device
                texts = texts.to(self.device)
                batch_size = texts.size(0)
                
                # Generate random SNR values
                snr = torch.FloatTensor(batch_size, 1).uniform_(snr_range[0], snr_range[1]).to(self.device)
                
                # Get clean semantic representation
                with torch.no_grad():
                    clean_repr = semantic_encoder(texts)
                
                #--------------------------
                # Train Discriminator
                #--------------------------
                opt_D.zero_grad()
                
                # Real samples
                real_pred = self.D(clean_repr)
                real_target = torch.ones_like(real_pred)
                real_loss = self.adv_loss(real_pred, real_target)
                
                # Generate adversarial examples
                adv_repr = generate_adversarial(clean_repr, snr, epsilon)
                
                # Generate fake samples with SNR awareness
                z = torch.randn(batch_size, self.latent_dim, device=self.device)
                fake_repr = self.G(z, snr)
                fake_pred = self.D(fake_repr.detach())  # Detach to avoid training generator
                fake_target = torch.zeros_like(fake_pred)
                fake_loss = self.adv_loss(fake_pred, fake_target)
                
                # Total discriminator loss
                d_loss = real_loss + fake_loss
                d_loss.backward()
                opt_D.step()
                
                d_losses.append(d_loss.item())
                
                #--------------------------
                # Train Generator
                #--------------------------
                opt_G.zero_grad()
                
                # Adversarial loss (fool discriminator)
                fake_pred_g = self.D(fake_repr)
                g_target = torch.ones_like(fake_pred_g)
                g_loss_adv = self.adv_loss(fake_pred_g, g_target)
                
                # Calibration loss (ability to correct adversarial examples)
                calibrated_repr = self.calibrate(adv_repr, snr)
                calibration_loss = self.recon_loss(calibrated_repr, clean_repr)
                
                # FIM loss (encourage adaptation to SNR)
                _, importance_weights = self.fim(clean_repr, snr)
                fim_reg = 0.1 * torch.mean((1.0 - importance_weights).pow(2))  # Encourage using more features at high SNR
                
                # Total generator loss
                g_loss = g_loss_adv + calibration_loss + fim_reg
                g_loss.backward()
                opt_G.step()
                
                g_losses.append(g_loss.item())
            
            # Print epoch stats
            avg_g_loss = sum(g_losses) / len(g_losses) if g_losses else 0
            avg_d_loss = sum(d_losses) / len(d_losses) if d_losses else 0
            print(f"Epoch {epoch+1}/{epochs}, G Loss: {avg_g_loss:.4f}, D Loss: {avg_d_loss:.4f}") 