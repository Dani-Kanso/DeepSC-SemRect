import torch
import torch.nn as nn
import torch.optim as optim

class SemanticGenerator(nn.Module):
    """
    Generator network (3 ConvTranspose1d layers) that maps a random latent vector z
    to the 1×64 semantic representation space for text.
    """
    def __init__(self, latent_dim: int = 100):
        super(SemanticGenerator, self).__init__()
        # We treat semantic repr as a 1×64 "signal"
        self.net = nn.Sequential(
            # input z: (B, latent_dim, 1)
            nn.ConvTranspose1d(latent_dim, 64, kernel_size=4, stride=4),  # -> (B,64,4)
            nn.ReLU(inplace=True),
            nn.ConvTranspose1d(64, 16, kernel_size=4, stride=4),          # -> (B,16,16)
            nn.ReLU(inplace=True),
            nn.ConvTranspose1d(16, 1, kernel_size=4, stride=4),           # -> (B,1,64)
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z shape: (batch, latent_dim)
        z = z.unsqueeze(-1)  # -> (batch, latent_dim, 1)
        return self.net(z)

class SemanticDiscriminator(nn.Module):
    """
    Discriminator network (4 Conv1d layers) distinguishing real vs. generated semantics.
    """
    def __init__(self):
        super(SemanticDiscriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=4, stride=4),  # -> (B,16,16)
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(16, 32, kernel_size=4, stride=4),  # -> (B,32,4)
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(32, 64, kernel_size=2, stride=2),  # -> (B,64,2)
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(64, 128, kernel_size=2, stride=2), # -> (B,128,1)
            nn.LeakyReLU(0.2, inplace=True),
            nn.Flatten(),                                # -> (B,128)
            nn.Linear(128, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class SemRect:
    """
    Semantic Rectification (SemRect) using a GAN (Defense-GAN style) with
    3 ConvTranspose1d generator layers and 4 Conv1d discriminator layers,
    matching the paper's specifications (Table I).
    """
    def __init__(self,
                 latent_dim: int = 100,
                 device: torch.device = torch.device('cpu')):
        self.latent_dim = latent_dim
        self.device = device
        self.G = SemanticGenerator(latent_dim).to(device)
        self.D = SemanticDiscriminator().to(device)
        self.adv_loss = nn.BCEWithLogitsLoss()
        self.recon_loss = nn.MSELoss()

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
            for texts, _ in dataloader:
                texts = texts.to(self.device)
                with torch.no_grad():
                    # get real repr: shape (B,1,64)
                    real_repr = semantic_encoder(texts).unsqueeze(1)

                B = real_repr.size(0)
                real_lbl = torch.ones(B, 1, device=self.device)
                fake_lbl = torch.zeros(B, 1, device=self.device)

                # D step
                z = torch.randn(B, self.latent_dim, device=self.device)
                fake_repr = self.G(z)
                d_real = self.D(real_repr)
                d_fake = self.D(fake_repr.detach())
                d_loss = self.adv_loss(d_real, real_lbl) + self.adv_loss(d_fake, fake_lbl)
                opt_D.zero_grad(); d_loss.backward(); opt_D.step()

                # G step
                d_fake2 = self.D(fake_repr)
                g_loss = self.adv_loss(d_fake2, real_lbl)
                opt_G.zero_grad(); g_loss.backward(); opt_G.step()

            print(f"Epoch {epoch+1}/{epochs} D_loss={d_loss:.4f} G_loss={g_loss:.4f}")

    def calibrate(self,
                  h_adv: torch.Tensor,
                  z_steps: int = 50,
                  z_lr: float = 1e-2) -> torch.Tensor:
        # h_adv: (B, rep_dim=64) or (B,1,64)
        if h_adv.dim() == 2:
            h_adv = h_adv.unsqueeze(1)
        B = h_adv.size(0)
        z = torch.randn(B, self.latent_dim, device=self.device, requires_grad=True)
        opt_z = optim.Adam([z], lr=z_lr)

        for _ in range(z_steps):
            opt_z.zero_grad()
            x_rec = self.G(z)
            loss = self.recon_loss(x_rec, h_adv)
            loss.backward()
            opt_z.step()

        with torch.no_grad():
            return self.G(z).squeeze(1)
            
    def test_defense(self,
                     semantic_encoder: nn.Module,
                     classifier: nn.Module,
                     dataloader: torch.utils.data.DataLoader,
                     epsilon: float = 0.1,
                     z_steps: int = 500) -> dict:
        """
        Test the SemRect defense against FGSM attacks.
        
        Args:
            semantic_encoder: The encoder that produces semantic representations
            classifier: The classifier model (e.g., TextCNN)
            dataloader: DataLoader for test data
            epsilon: FGSM perturbation strength (default: 0.1)
            z_steps: Number of latent search steps (default: 500)
            
        Returns:
            Dictionary containing accuracy metrics with and without defense
        """
        semantic_encoder.eval()
        classifier.eval()
        
        total = 0
        correct_clean = 0
        correct_adv = 0
        correct_defended = 0
        
        for texts, labels in dataloader:
            texts = texts.to(self.device)
            labels = labels.to(self.device)
            
            # Get clean semantic representations
            with torch.no_grad():
                semantic_repr = semantic_encoder(texts)
            
            # Make semantic_repr require gradients for FGSM
            semantic_repr = semantic_repr.clone().detach().requires_grad_(True)
            
            # Clean accuracy
            with torch.no_grad():
                clean_outputs = classifier(semantic_repr)
                clean_preds = clean_outputs.argmax(dim=1)
                correct_clean += (clean_preds == labels).sum().item()
            
            # Generate FGSM attack
            outputs = classifier(semantic_repr)
            loss = nn.CrossEntropyLoss()(outputs, labels)
            loss.backward()
            
            # Create adversarial examples
            with torch.no_grad():
                # Properly compute gradient w.r.t semantic representation
                semantic_grad = semantic_repr.grad.sign()
                s_adv = semantic_repr + epsilon * semantic_grad
                
                # Adversarial accuracy (without defense)
                adv_outputs = classifier(s_adv)
                adv_preds = adv_outputs.argmax(dim=1)
                correct_adv += (adv_preds == labels).sum().item()
                
                # Apply SemRect defense
                s_defended = self.calibrate(s_adv, z_steps=z_steps)
                
                # Defended accuracy
                defended_outputs = classifier(s_defended)
                defended_preds = defended_outputs.argmax(dim=1)
                correct_defended += (defended_preds == labels).sum().item()
            
            total += labels.size(0)
        
        results = {
            'clean_accuracy': correct_clean / total,
            'adversarial_accuracy': correct_adv / total,
            'defended_accuracy': correct_defended / total
        }
        
        print(f"Clean accuracy: {results['clean_accuracy']:.4f}")
        print(f"Adversarial accuracy: {results['adversarial_accuracy']:.4f}")
        print(f"Defended accuracy: {results['defended_accuracy']:.4f}")
        
        return results

# Usage remains the same, with latent_dim=100 per Table I.
