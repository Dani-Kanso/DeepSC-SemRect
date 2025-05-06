import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import spectral_norm

class SemanticGenerator(nn.Module):
    def __init__(self, latent_dim=100, output_dim=128, seq_len=31):
        super().__init__()
        self.latent_dim = latent_dim
        self.seq_len = seq_len
        self.output_dim = output_dim
        
        # Simple but effective generator architecture
        self.main = nn.Sequential(
            # Expand latent dim to hidden features
            nn.Linear(latent_dim, 512),
            nn.LayerNorm(512),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Project to sequence length × feature dimension
            nn.Linear(512, 1024),
            nn.LayerNorm(1024),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Final projection to output dimension
            nn.Linear(1024, seq_len * output_dim),
        )
    
    def forward(self, z):
        """
        Forward pass to generate semantic representation from random vector
        
        Args:
            z: Random latent vector [batch_size, latent_dim]
        Returns:
            Semantic representation [batch_size, seq_len, output_dim]
        """
        batch_size = z.size(0)
        x = self.main(z)
        x = x.view(batch_size, self.seq_len, self.output_dim)
        return x

class SemanticDiscriminator(nn.Module):
    def __init__(self, input_dim=128, seq_len=31):
        super(SemanticDiscriminator, self).__init__()
        
        # Use spectral normalization for stability
        self.net = nn.Sequential(
            # Initial convolutional layer for sequence processing
            nn.Conv1d(input_dim, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Downsample with strided convolution
            spectral_norm(nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Another downsampling
            spectral_norm(nn.Conv1d(128, 256, kernel_size=3, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Final layers
            nn.Flatten(),
            spectral_norm(nn.Linear(256 * ((seq_len//4) + (1 if seq_len % 4 != 0 else 0)), 512)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Linear(512, 1))
        )
    
    def forward(self, x):
        # x shape: (batch, seq_len, input_dim)
        # Transpose to (batch, input_dim, seq_len) for Conv1d
        x = x.transpose(1, 2)
        return self.net(x)

class SemRect(nn.Module):  
    """
    SemRect implementation as described in the paper:
    A Defense-GAN approach that generates semantic signatures to 
    calibrate adversarial perturbations.
    """
    def __init__(self, latent_dim=100, output_dim=128, seq_len=31,
                 device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        super(SemRect, self).__init__()  
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.seq_len = seq_len
        self.device = device
        
        # Initialize generator and discriminator
        self.G = SemanticGenerator(latent_dim, output_dim, seq_len).to(device)
        self.D = SemanticDiscriminator(output_dim, seq_len).to(device)
        
        # Loss functions
        self.adv_loss = nn.BCEWithLogitsLoss()
        self.recon_loss = nn.MSELoss()
        
        self.to(device) 
    
    def calibrate(self, h_adv, signature_scale=0.1):
        """
        Calibrate potentially corrupted semantic representation using 
        a generated semantic signature.
        
        Args:
            h_adv: Input semantic representation [batch, seq_len, output_dim]
            signature_scale: Scaling factor for the signature
        """
        batch_size = h_adv.size(0)
        
        # Generate random vector (z) for semantic signature
        z = torch.randn(batch_size, self.latent_dim, device=self.device)
        
        # Generate semantic signature using the trained generator
        with torch.no_grad():
            signature = self.G(z)
            
        # Apply the signature to calibrate adversarial input
        calibrated = h_adv + signature_scale * signature
        
        return calibrated
    
    def pretrain_generator(self, semantic_encoder, dataloader, epochs=5, lr=1e-4):
        """
        Pre-train generator on clean semantic representations as described in the paper.
        This follows the Defense-GAN approach of training on clean data.
        
        Args:
            semantic_encoder: Function/module to extract semantic representations
            dataloader: Dataloader providing clean text inputs
            epochs: Number of pre-training epochs
            lr: Learning rate
        """
        print("Pre-training SemRect generator on clean semantic data...")
        
        # Set models to correct modes
        self.G.train()
        if hasattr(semantic_encoder, 'eval'):
            semantic_encoder.eval()
            
        # Optimizer for generator
        optimizer = optim.Adam(self.G.parameters(), lr=lr, betas=(0.5, 0.999))
        
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
                
                # Generate semantic representation from random
                z = torch.randn(batch_size, self.latent_dim, device=self.device)
                fake_repr = self.G(z)
                
                # Reconstruction loss
                loss = self.recon_loss(fake_repr, clean_repr)
                
                # Update generator
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                
            # Print epoch stats
            avg_loss = running_loss / max(batch_count, 1)
            print(f"Pre-training epoch {epoch+1}/{epochs}, Avg Loss: {avg_loss:.4f}")
    
    def train_gan(self, semantic_encoder, dataloader, epochs=10, lr=1e-4, epsilon=0.1):
        """
        Train SemRect using the GAN approach as described in the paper.
        Includes adversarial examples generation using FGSM.
        
        Args:
            semantic_encoder: Function/module to extract semantic representations
            dataloader: Dataloader providing clean text inputs
            epochs: Number of training epochs
            lr: Learning rate
            epsilon: FGSM attack strength
        """
        print("Training SemRect GAN with adversarial examples...")
        
        # Set generator to training mode
        self.G.train()
        self.D.train()
        
        # Optimizers
        opt_G = optim.Adam(self.G.parameters(), lr=lr, betas=(0.5, 0.999))
        opt_D = optim.Adam(self.D.parameters(), lr=lr, betas=(0.5, 0.999))
        
        # For generating adversarial examples
        def generate_adversarial(clean_repr, epsilon=0.1):
            # Simple FGSM implementation as per paper
            perturbed = clean_repr.clone().detach().requires_grad_(True)
            
            # Forward through discriminator
            pred = self.D(perturbed)
            
            # Compute loss to maximize discriminator output
            target = torch.zeros_like(pred)
            loss = self.adv_loss(pred, target)
            
            # Get gradients
            loss.backward()
            
            # FGSM attack
            if perturbed.grad is not None:
                adv_repr = perturbed + epsilon * perturbed.grad.sign()
            else:
                adv_repr = perturbed
                
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
                
                # Get clean semantic representation
                with torch.no_grad():
                    clean_repr = semantic_encoder(texts)
                
                # Create adversarial examples
                adv_repr = generate_adversarial(clean_repr, epsilon)
                
                # Labels for GAN training
                real_label = torch.ones(batch_size, 1, device=self.device)
                fake_label = torch.zeros(batch_size, 1, device=self.device)
                
                # ---------------
                # Train Discriminator
                # ---------------
                opt_D.zero_grad()
                
                # Real samples
                d_real = self.D(clean_repr)
                d_real_loss = self.adv_loss(d_real, real_label)
                
                # Fake samples (generated signatures)
                z = torch.randn(batch_size, self.latent_dim, device=self.device)
                fake_repr = self.G(z)
                d_fake = self.D(fake_repr.detach())
                d_fake_loss = self.adv_loss(d_fake, fake_label)
                
                # Adversarial samples (for calibration)
                # Calibrated = adversarial + signature
                calibrated = adv_repr + 0.1 * fake_repr.detach()
                d_cal = self.D(calibrated.detach())
                d_cal_loss = self.adv_loss(d_cal, fake_label)
                
                # Total discriminator loss
                d_loss = d_real_loss + 0.5 * (d_fake_loss + d_cal_loss)
                d_loss.backward()
                opt_D.step()
                
                # ---------------
                # Train Generator
                # ---------------
                opt_G.zero_grad()
                
                # Adversarial loss (fool discriminator)
                d_fake2 = self.D(fake_repr)
                g_loss_adv = self.adv_loss(d_fake2, real_label)
                
                # Calibration loss (calibrated should be close to clean)
                calibrated = adv_repr + 0.1 * fake_repr
                g_loss_cal = self.recon_loss(calibrated, clean_repr)
                
                # Total generator loss
                g_loss = g_loss_adv + 10.0 * g_loss_cal
                g_loss.backward()
                opt_G.step()
                
                # Store losses
                g_losses.append(g_loss.item())
                d_losses.append(d_loss.item())
            
            # Print epoch stats
            avg_g_loss = sum(g_losses) / max(len(g_losses), 1)
            avg_d_loss = sum(d_losses) / max(len(d_losses), 1)
            print(f"Epoch {epoch+1}/{epochs}, G_loss: {avg_g_loss:.4f}, D_loss: {avg_d_loss:.4f}")
            
            # Save checkpoint every few epochs
            if (epoch + 1) % 5 == 0:
                try:
                    torch.save({
                        'epoch': epoch,
                        'g_state_dict': self.G.state_dict(),
                        'd_state_dict': self.D.state_dict(),
                    }, f"checkpoints/semrect_epoch_{epoch+1}.pth")
                    print(f"Saved checkpoint at epoch {epoch+1}")
                except:
                    print("Could not save checkpoint")
        
        print("SemRect training complete!")
