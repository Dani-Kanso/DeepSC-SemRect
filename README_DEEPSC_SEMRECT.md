# DeepSC + SemRect Integration

This implementation combines DeepSC (Deep Semantic Communication) with SemRect (Semantic Rectification) to create a robust semantic communication system that can defend against adversarial attacks.

## Overview

DeepSC is a transformer-based semantic communication system that transmits text over noisy channels. SemRect is a defense mechanism that protects the semantic integrity of the transmitted signal by projecting potentially corrupted semantic representations back to a manifold of valid semantic representations.

The integration places SemRect between the channel decoder and transformer decoder of DeepSC to ensure robust semantic representation even under adversarial conditions.

## Architecture

The integrated system follows this pipeline:

1. **Encoder**: Transform input text to semantic representation
2. **Channel Encoder**: Encode semantic representation for channel transmission
3. **Channel**: Apply channel effects (AWGN, Rayleigh, or Rician)
4. **Channel Decoder**: Decode received signal back to semantic representation
5. **SemRect**: Apply semantic rectification to defend against adversarial perturbations
6. **Decoder**: Transform protected semantic representation back to text

## How to Use

### Training and Evaluation

```python
from deepsc_semrect import DeepSCWithSemRect
from torch.utils.data import DataLoader
from dataset import EurDataset, collate_data

# Create datasets
train_dataset = EurDataset('train')
test_dataset = EurDataset('test')

train_loader = DataLoader(train_dataset, batch_size=128, collate_fn=collate_data)
test_loader = DataLoader(test_dataset, batch_size=128, collate_fn=collate_data)

# Create model
model = DeepSCWithSemRect(
    num_layers=4,
    src_vocab_size=10000,  # Replace with your vocab size
    trg_vocab_size=10000,  # Replace with your vocab size
    src_max_len=30,
    trg_max_len=30,
    d_model=128,
    num_heads=8,
    dff=512,
    dropout=0.1,
    semrect_latent_dim=100,
    device='cuda'
)

# Train SemRect
model.train_semrect(train_loader, epochs=30, lr=1e-4)

# Test with adversarial attacks
results = model.test_with_adversarial_attack(
    test_loader,
    epsilon=0.1,  # FGSM attack strength
    channel_type='AWGN',
    snr=10  # Signal-to-noise ratio in dB
)

# Print results
print(f"Clean (No Defense): {results['clean_no_defense']:.4f}")
print(f"Clean (With SemRect): {results['clean_with_defense']:.4f}")
print(f"Adversarial (No Defense): {results['adversarial_no_defense']:.4f}")
print(f"Adversarial (With SemRect): {results['adversarial_with_defense']:.4f}")
```

### Using the Command Line

```bash
python deepsc_semrect.py --channel AWGN --snr 10 --epsilon 0.1 --semrect-epochs 30
```

## Key Components

### 1. DeepSCWithSemRect

The main class that integrates DeepSC with SemRect. It provides:

- Forward method for end-to-end processing
- SemRect training method 
- Evaluation methods for clean and adversarial scenarios

### 2. Semantic Extraction

The `SemanticExtractor` nested class pulls clean semantic representations from DeepSC for training SemRect.

### 3. Adversarial Testing

The implementation includes FGSM-style attacks targeting the semantic representations, with and without SemRect defense.

## Parameters

- **num_layers**: Number of transformer layers
- **d_model**: Model dimension for transformer
- **num_heads**: Number of attention heads
- **dff**: Dimension of feedforward network
- **semrect_latent_dim**: Dimension of SemRect's latent space (default: 100)
- **snr**: Signal-to-noise ratio in dB
- **epsilon**: FGSM attack strength
- **channel_type**: 'AWGN', 'Rayleigh', or 'Rician'

## Performance Metrics

- **Clean accuracy**: Performance without attack
- **Adversarial accuracy**: Performance under attack
- **Defense improvement**: How much SemRect improves accuracy under attack

## Requirements

- PyTorch
- DeepSC implementation
- SemRect implementation

## Reference

If you use this implementation, please cite:

- DeepSC: Deep Semantic Communication Systems
- SemProtector & SemRect: Protecting Semantic Communication Systems Against Adversarial Attacks 