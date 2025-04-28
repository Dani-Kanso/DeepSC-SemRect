# SemRect: Semantic Rectification for DeepSC

This module implements the Semantic Rectification (SemRect) component of the SemProtector system for DeepSC. SemRect ensures semantic integrity against adversarial perturbations in semantic communication systems.

## Overview

SemRect is designed to protect text semantic representations from adversarial attacks by projecting potentially corrupted semantics back onto the manifold of valid semantic representations. It uses a Defense-GAN style approach with the following components:

1. **Generator (G)**: A network that maps a random latent vector z to the semantic representation space
2. **Discriminator (D)**: A network that distinguishes between real and generated semantic representations
3. **Calibration**: A procedure to find the closest valid semantic representation for a potentially corrupted one

## Architecture

The implementation uses convolutional architectures:

- **Generator**: 3 ConvTranspose1d layers (100 → 64 → 16 → 1)
- **Discriminator**: 4 Conv1d layers + FC layer (1 → 16 → 32 → 64 → 128 → 1)

This design treats semantic representations as 1×64 "signals" rather than using fully-connected layers.

## Usage

### 1. Initialize SemRect

```python
from SemRect import SemRect

# Initialize SemRect
semrect = SemRect(
    latent_dim=100,  # Dimension of the latent space
    device=torch.device('cuda')  # Device to use
)
```

### 2. Train the GAN

Train the GAN on clean semantic representations:

```python
# semantic_encoder: Your text encoder that produces semantic vectors
# dataloader: DataLoader with text data
semrect.train_gan(
    semantic_encoder=semantic_encoder,
    dataloader=dataloader,
    epochs=50,
    lr=1e-4
)
```

### 3. Calibrate Adversarial Semantics

When a potentially corrupted semantic vector is received, calibrate it:

```python
# s_adv: Potentially corrupted semantic vector (B, rep_dim)
s_defended = semrect.calibrate(
    s_adv,
    z_steps=500,  # Number of latent search steps
    z_lr=1e-2  # Learning rate for latent search
)
```

### 4. Test Defense Against FGSM Attacks

Evaluate the effectiveness of SemRect against FGSM attacks:

```python
results = semrect.test_defense(
    semantic_encoder=semantic_encoder,
    classifier=classifier,  # e.g., TextCNN
    dataloader=test_dataloader,
    epsilon=0.1,  # FGSM perturbation strength
    z_steps=500  # Latent search steps
)

print(f"Clean accuracy: {results['clean_accuracy']}")
print(f"Adversarial accuracy: {results['adversarial_accuracy']}")
print(f"Defended accuracy: {results['defended_accuracy']}")
```

## Example

See `semrect_example.py` for a complete working example with dummy data.

## Integration with DeepSC

To integrate SemRect with DeepSC:

1. Train your DeepSC model
2. Extract the semantic encoder part
3. Train SemRect using this encoder and your training data
4. In the inference pipeline, add the calibration step between semantic encoding and decoding

## Key Parameters

- **latent_dim**: Dimension of the latent space (default: 100)
- **z_steps**: Number of latent search steps for calibration (default: 50, increase to 500 for better results)
- **z_lr**: Learning rate for latent search (default: 1e-2)
- **epsilon**: FGSM perturbation strength for testing (default: 0.1)

## References

This implementation follows the Defense-GAN approach as described in:
- Samangouei, P., Kabkab, M., & Chellappa, R. (2018). Defense-GAN: Protecting classifiers against adversarial attacks using generative models. ICLR 2018. 