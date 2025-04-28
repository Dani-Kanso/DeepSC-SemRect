import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from SemRect import SemRect

# Example TextCNN for classification
class TextCNN(nn.Module):
    def __init__(self, input_dim=64, num_classes=2):
        super(TextCNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Example semantic encoder
class SemanticEncoder(nn.Module):
    def __init__(self, vocab_size=10000, embed_dim=128, output_dim=64):
        super(SemanticEncoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.conv = nn.Conv1d(embed_dim, output_dim, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool1d(1)
        
    def forward(self, x):
        # x: (batch_size, seq_len)
        x = self.embedding(x)  # (batch_size, seq_len, embed_dim)
        x = x.transpose(1, 2)  # (batch_size, embed_dim, seq_len)
        x = self.conv(x)       # (batch_size, output_dim, seq_len)
        x = self.pool(x).squeeze(-1)  # (batch_size, output_dim)
        return x

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create dummy data
    batch_size = 32
    seq_len = 20
    vocab_size = 10000
    output_dim = 64
    num_classes = 2
    
    # Generate dummy text data (vocab indices)
    texts = torch.randint(0, vocab_size, (100, seq_len))
    labels = torch.randint(0, num_classes, (100,))
    
    # Create data loader
    dataset = TensorDataset(texts, labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize models
    semantic_encoder = SemanticEncoder(vocab_size, embed_dim=128, output_dim=output_dim).to(device)
    classifier = TextCNN(input_dim=output_dim, num_classes=num_classes).to(device)
    
    # Initialize SemRect
    semrect = SemRect(latent_dim=100, device=device)
    
    # Step 1: Train SemRect GAN on clean semantic representations
    print("Training SemRect GAN...")
    semrect.train_gan(semantic_encoder, dataloader, epochs=5, lr=1e-4)  # Reduced epochs for example
    
    # Step 2: Test SemRect defense against FGSM attack
    print("\nTesting SemRect defense against FGSM attack...")
    results = semrect.test_defense(
        semantic_encoder=semantic_encoder,
        classifier=classifier,
        dataloader=dataloader,
        epsilon=0.1,
        z_steps=10  # Reduced steps for example
    )
    
    print("\nDefense summary:")
    print(f"- Clean accuracy: {results['clean_accuracy']:.4f}")
    print(f"- Under attack (without defense): {results['adversarial_accuracy']:.4f}")
    print(f"- With SemRect defense: {results['defended_accuracy']:.4f}")
    
    # Demonstrate calibration on a single batch
    batch = next(iter(dataloader))
    texts, _ = batch
    texts = texts.to(device)
    
    # Get clean semantic representation
    with torch.no_grad():
        clean_repr = semantic_encoder(texts)
        
    # Add random noise to simulate attack
    noisy_repr = clean_repr + 0.1 * torch.randn_like(clean_repr)
    
    # Calibrate using SemRect
    calibrated_repr = semrect.calibrate(noisy_repr, z_steps=50)
    
    # Compare L2 distances
    noise_l2 = torch.norm(noisy_repr - clean_repr, dim=1).mean().item()
    calibrated_l2 = torch.norm(calibrated_repr - clean_repr, dim=1).mean().item()
    
    print(f"\nL2 distance - Noisy vs. Clean: {noise_l2:.4f}")
    print(f"L2 distance - Calibrated vs. Clean: {calibrated_l2:.4f}")
    print(f"Improvement: {(1 - calibrated_l2/noise_l2) * 100:.2f}%")

if __name__ == "__main__":
    main() 