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
        
        # Encode input semantic representation
        self.encoder = nn.Sequential(
            nn.Linear(output_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.ReLU()
        )
        
        # Project latent vector to sequence
        self.project = nn.Linear(latent_dim, seq_len * output_dim)
        
        # Use LSTM for sequence generation
        self.lstm = nn.LSTM(output_dim, output_dim, num_layers=2, batch_first=True)
        
        # Final projection with residual connection
        self.final = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.LayerNorm(output_dim)
        )
    
    def forward(self, h=None, z=None, target_seq_len=None):
        """
        Forward pass with dependency on input semantic representation
        
        Args:
            h: Input semantic representation [batch_size, seq_len, output_dim]
            z: Optional random noise vector [batch_size, latent_dim]
            target_seq_len: Optional target sequence length
        """
        if h is None and z is None:
            raise ValueError("Either h or z must be provided")
            
        if h is not None:
            B = h.size(0)
            # Get sequence-aware encoding by pooling across sequence
            h_mean = h.mean(dim=1)  # [B, output_dim]
            h_enc = self.encoder(h_mean)  # [B, latent_dim]
        else:
            B = z.size(0)
            h_enc = torch.zeros(B, self.latent_dim, device=z.device)
            
        # Generate or use provided random component
        if z is None:
            z = torch.randn(B, self.latent_dim, device=h_enc.device)
            
        # Combine semantic encoding with random noise
        combined = h_enc + z  # [B, latent_dim]
        
        # Project to sequence
        x = self.project(combined)  # [B, seq_len*output_dim]
        x = x.view(B, self.seq_len, self.output_dim)  # [B, seq_len, output_dim]
        
        # Process with LSTM for sequence-aware generation
        x, _ = self.lstm(x)  # [B, seq_len, output_dim]
        
        # Final projection
        x = self.final(x)  # [B, seq_len, output_dim]
        
        # Adjust sequence length if needed
        seq_len = target_seq_len if target_seq_len is not None else self.seq_len
        if seq_len != self.seq_len:
            if seq_len < self.seq_len:
                x = x[:, :seq_len, :]
            else:
                padding = torch.zeros(B, seq_len - self.seq_len, self.output_dim, 
                                   device=x.device)
                x = torch.cat([x, padding], dim=1)
        
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
    def __init__(self, latent_dim=100, output_dim=128, seq_len=31,
                 device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        super(SemRect, self).__init__()  
        self.latent_dim = latent_dim
        self.device = device
        self.G = SemanticGenerator(latent_dim, output_dim, seq_len).to(device)
        self.D = SemanticDiscriminator(output_dim, seq_len).to(device)
        self.adv_loss = nn.BCEWithLogitsLoss()
        self.recon_loss = nn.MSELoss()
        self.to(device) 
    
    def calibrate(self, h_adv, z_steps=15, z_lr=1e-2, alpha=0.3):
        """
        Calibrate potentially corrupted semantic representation using semantic signature.
        
        Args:
            h_adv: Input semantic representation [batch, seq_len, output_dim]
            z_steps: Number of optimization steps
            z_lr: Learning rate for z optimization
            alpha: Weight for signature correction (higher = stronger correction)
        """
        # Detach h_adv and ensure it's on the correct device
        h_adv = h_adv.detach().to(self.device)
        
        # Get target sequence length from input
        B, seq_len, output_dim = h_adv.size()
        
        # Initialize z with random noise
        z = torch.randn(B, self.latent_dim, device=self.device)
        z.requires_grad_(True)
        
        # Save original training state and set to training mode for backward pass
        was_training = self.G.training
        self.G.train()  # Need training mode for cudnn RNN backward
        
        # Create optimizer for z
        opt_z = torch.optim.Adam([z], lr=z_lr)
        best_z = z.clone()
        min_loss = float('inf')
        
        # Layer normalization for semantic stabilization
        layer_norm = nn.LayerNorm((seq_len, output_dim)).to(self.device)
        h_adv_norm = layer_norm(h_adv)
        
        # Phase 1: Find optimal latent code through iterative optimization
        for step in range(z_steps):
            opt_z.zero_grad()
            
            # Generate signature with h_adv conditioning
            with torch.enable_grad():
                x_rec = self.G(h_adv, z, target_seq_len=seq_len)
                x_rec_norm = layer_norm(x_rec)
                
                # Compute reconstruction loss with normalized tensors
                recon_loss = self.recon_loss(x_rec_norm, h_adv_norm)
                
                # Discriminator-based "reality" loss to ensure semantic validity
                d_score = self.D(x_rec).mean()
                reality_loss = -d_score  # Maximize discriminator score
                
                # Total loss with weighted components
                loss = recon_loss + 0.1 * reality_loss
            
            # Store best result
            if loss.item() < min_loss:
                min_loss = loss.item()
                best_z = z.clone()
            
            # Backward pass for z optimization
            loss.backward(retain_graph=True)
            opt_z.step()
            
            # Clear gradients after each step
            if z.grad is not None:
                z.grad.zero_()

        # Phase 2: Generate and apply the semantic signature 
        with torch.no_grad():
            # Generate semantic signature based on best latent code
            signature = self.G(h_adv, best_z, target_seq_len=seq_len)
            
            # Apply adaptive signature strength based on detection confidence
            # Higher correction where reconstruction loss is lower (more confident)
            detection_confidence = torch.sigmoid(torch.tensor(-min_loss * 5, device=self.device))  # Convert loss to confidence
            adaptive_alpha = alpha * detection_confidence
            
            # Apply signature with residual connection
            final_output = (1 - adaptive_alpha) * h_adv + adaptive_alpha * signature
            
            # Add normalization to ensure stable semantic range
            final_output = layer_norm(final_output) * h_adv.std(dim=(1,2), keepdim=True)
        
        # Restore original training state
        self.G.train(was_training)
            
        return final_output
    
    def train_gan(self,
                  semantic_encoder: nn.Module,
                  dataloader: torch.utils.data.DataLoader,
                  epochs: int = 50,
                  lr: float = 1e-4):
        semantic_encoder.eval()
        for p in semantic_encoder.parameters(): p.requires_grad = False

        opt_G = optim.Adam(self.G.parameters(), lr=lr)
        opt_D = optim.Adam(self.D.parameters(), lr=lr)

        for epoch in range(epochs):
            for texts in dataloader:  # Changed to handle single value from dataloader
                if not isinstance(texts, torch.Tensor):
                    texts = texts[0]  # Handle case where dataloader might return a tuple/list
                    
                texts = texts.to(self.device)
                with torch.no_grad():
                    # Get real semantic representation 
                    # shape: (batch_size, seq_len, output_dim)
                    real_repr = semantic_encoder(texts)

                B = real_repr.size(0)
                real_lbl = torch.ones(B, 1, device=self.device)
                fake_lbl = torch.zeros(B, 1, device=self.device)

                # D step
                z = torch.randn(B, self.latent_dim, device=self.device)
                fake_repr = self.G(real_repr, z)  # Will match real_repr shape
                
                d_real = self.D(real_repr)
                d_fake = self.D(fake_repr.detach())
                d_loss = self.adv_loss(d_real, real_lbl) + self.adv_loss(d_fake, fake_lbl)
                opt_D.zero_grad()
                d_loss.backward()
                opt_D.step()

                # G step
                d_fake2 = self.D(fake_repr)
                g_loss = self.adv_loss(d_fake2, real_lbl)
                opt_G.zero_grad()
                g_loss.backward()
                opt_G.step()

            print(f"Epoch {epoch+1}/{epochs} D_loss={d_loss:.4f} G_loss={g_loss:.4f}")
