import torch
import torch.nn as nn
import torch.optim as optim

class SemanticGenerator(nn.Module):
    def __init__(self, latent_dim=100, output_dim=128, seq_len=31):
        super(SemanticGenerator, self).__init__()
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.seq_len = seq_len
        
        # Project latent vector and reshape for sequential data
        self.project = nn.Sequential(
            nn.Linear(latent_dim, output_dim * seq_len),
            nn.LayerNorm(output_dim * seq_len)
        )
        
        # Process with sequence awareness
        self.net = nn.Sequential(
            nn.Linear(output_dim * seq_len, output_dim * seq_len),
            nn.ReLU(inplace=True),
            nn.Linear(output_dim * seq_len, output_dim * seq_len),
            nn.ReLU(inplace=True),
            nn.Linear(output_dim * seq_len, output_dim * seq_len),
        )
    
    def forward(self, z, target_seq_len=None):
        """
        Forward pass with dynamic sequence length support
        Args:
            z: Input latent vector (batch_size, latent_dim)
            target_seq_len: Optional target sequence length
        """
        B = z.size(0)
        seq_len = target_seq_len if target_seq_len is not None else self.seq_len
        
        # Initial projection
        x = self.project(z)
        x = self.net(x)
        
        # Reshape to target sequence length
        x = x.view(B, self.seq_len, self.output_dim)
        
        # Adjust sequence length if needed
        if target_seq_len and target_seq_len != self.seq_len:
            if target_seq_len < self.seq_len:
                x = x[:, :target_seq_len, :]
            else:
                padding = torch.zeros(B, target_seq_len - self.seq_len, self.output_dim, 
                                   device=x.device)
                x = torch.cat([x, padding], dim=1)
        
        return x

class SemanticDiscriminator(nn.Module):
    def __init__(self, input_dim=128, seq_len=31):
        super(SemanticDiscriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim * seq_len, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 1)
        )
    
    def forward(self, x):
        # x shape: (batch, seq_len, input_dim)
        x = x.view(x.size(0), -1)  # Flatten sequence and features
        return self.net(x)

class SemRect:
    def __init__(self, latent_dim=100, output_dim=128, seq_len=31,
                 device=torch.device('cpu')):
        self.latent_dim = latent_dim
        self.device = device
        self.G = SemanticGenerator(latent_dim, output_dim, seq_len).to(device)
        self.D = SemanticDiscriminator(output_dim, seq_len).to(device)
        self.adv_loss = nn.BCEWithLogitsLoss()
        self.recon_loss = nn.MSELoss()
    
    def calibrate(self, h_adv, z_steps=10, z_lr=1e-2):
        """
        Calibrate potentially corrupted semantic representation with improved semantic preservation.
        
        Args:
            h_adv: Input semantic representation (batch, seq_len, output_dim)
            z_steps: Number of optimization steps
            z_lr: Learning rate for z optimization
        """
        # Detach h_adv and ensure it's on the correct device
        h_adv = h_adv.detach().to(self.device)
        
        # Get target sequence length from input
        B, seq_len, output_dim = h_adv.size()
        
        # Initialize z on the correct device with gradients
        z = torch.randn(B, self.latent_dim, device=self.device)
        z.requires_grad_(True)
        
        # Ensure generator is in eval mode
        self.G.eval()
        
        # Create optimizer for z
        opt_z = torch.optim.Adam([z], lr=z_lr)
        best_z = z.clone()
        min_loss = float('inf')
        
        # Layer normalization for semantic stabilization
        layer_norm = nn.LayerNorm((seq_len, output_dim)).to(self.device)
        h_adv_norm = layer_norm(h_adv)
        
        for step in range(z_steps):
            opt_z.zero_grad()
            
            # Forward pass through generator with target sequence length
            with torch.enable_grad():
                x_rec = self.G(z, target_seq_len=seq_len)
                x_rec_norm = layer_norm(x_rec)
                
                # Compute reconstruction loss with normalized tensors
                recon_loss = self.recon_loss(x_rec_norm, h_adv_norm)
                
                # Add semantic preservation loss
                sem_loss = torch.mean(torch.abs(x_rec - h_adv))
                
                # Total loss with weighted components
                loss = recon_loss + 0.1 * sem_loss
            
            # Store best result
            if loss.item() < min_loss:
                min_loss = loss.item()
                best_z = z.clone()
            
            # Backward pass for z optimization
            loss.backward()
            opt_z.step()

        # Use best found z for final reconstruction with residual connection
        with torch.no_grad():
            final_output = self.G(best_z, target_seq_len=seq_len)
            # Add residual connection to preserve semantic information
            final_output = 0.8 * h_adv + 0.2 * final_output
            
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
                fake_repr = self.G(z)  # Will match real_repr shape
                
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
