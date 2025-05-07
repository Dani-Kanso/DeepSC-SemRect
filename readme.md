# DeepSC with Semantic Protection

This repository contains implementations of Deep Semantic Communication (DeepSC) with two semantic protection mechanisms:

1. **SemRect**: A Defense-GAN approach for generating semantic signatures to calibrate adversarial perturbations
2. **SemRectFIM**: An enhanced version of SemRect that integrates Feature Importance Mapping (FIM) for SNR-aware semantic signature generation and calibration

## Overview

DeepSC is a transformer-based semantic communication system that directly processes text at the semantic level. The semantic protection mechanisms (SemRect and SemRectFIM) are added between the channel decoder and the transformer decoder to protect against adversarial perturbations.

### SemRect
- Uses a GAN-based approach to generate semantic signatures
- Calibrates received semantic representations to mitigate adversarial perturbations
- Has fixed calibration strength regardless of channel conditions

### SemRectFIM
- Extends SemRect with SNR-aware Feature Importance Mapping
- Dynamically adjusts semantic signatures based on channel conditions
- Applies adaptive calibration strength based on SNR
- Focuses on features most resilient to specific channel conditions
- Provides superior performance in varying SNR environments

## Installation

```bash
git clone https://github.com/yourusername/deepsc-semantic-protection.git 
cd deepsc-semantic-protection
pip install -r requirements.txt
```

## Usage

### Training

#### 1. Pre-train DeepSC without protection

```bash
python main.py --channel AWGN --use-protection none --pretrain-epochs 50
```

#### 2. Train with SemRect protection

```bash
python main.py --channel AWGN --use-protection semrect --protection-epochs 30
```

#### 3. Train with SemRectFIM protection

```bash
python main.py --channel AWGN --use-protection semrectfim --protection-epochs 30 --snr-min 0 --snr-max 30
```

### Evaluation

To evaluate and compare both protection methods across different SNR levels:

```bash
python main.py --compare-protections --test-snrs 0 5 10 15 20
```

## Command Line Arguments

- `--vocab-file`: Path to vocabulary file (default: 'europarl/vocab.json')
- `--checkpoint-path`: Path to save checkpoints (default: 'checkpoints/deepsc-Rayleigh')
- `--channel`: Channel type: AWGN, Rayleigh, or Rician (default: 'Rayleigh')
- `--MAX-LENGTH`: Maximum sequence length (default: 30)
- `--d-model`: Model dimension (default: 128)
- `--num-layers`: Number of transformer layers (default: 4)
- `--num-heads`: Number of attention heads (default: 8)
- `--batch-size`: Batch size (default: 128)
- `--pretrain-epochs`: Epochs for DeepSC pre-training (default: 50)
- `--protection-epochs`: Epochs for semantic protection training (default: 30)
- `--use-protection`: Semantic protection type: none, semrect, or semrectfim (default: 'none')
- `--semrect-latent-dim`: Latent dimension for semantic protection (default: 100)
- `--snr-min`: Minimum SNR value for training (dB) (default: 0)
- `--snr-max`: Maximum SNR value for training (dB) (default: 30)
- `--test-snrs`: SNR values for testing (default: [0, 5, 10, 15, 20])
- `--compare-protections`: Run comparison between SemRect and SemRectFIM

## Results

SemRectFIM outperforms standard SemRect, especially in low SNR regimes. Key advantages include:

1. **SNR-adaptive protection**: SemRectFIM dynamically adjusts protection strength based on channel conditions
2. **Feature-selective protection**: Focuses on protecting semantically important features based on SNR
3. **Improved robustness**: Higher BLEU scores and lower attack success rates in varying channel conditions

## Example Performance Comparison

| SNR (dB) | Method    | Attack Success Rate | BLEU Clean | BLEU Adversarial | BLEU Defended | Defense Effectiveness |
|----------|-----------|---------------------|------------|------------------|---------------|------------------------|
| 0        | SemRect   | 0.65                | 0.82       | 0.35             | 0.65          | 0.64                   |
| 0        | SemRectFIM| 0.60                | 0.82       | 0.35             | 0.75          | 0.85                   |
| 10       | SemRect   | 0.55                | 0.90       | 0.45             | 0.78          | 0.73                   |
| 10       | SemRectFIM| 0.50                | 0.90       | 0.45             | 0.85          | 0.89                   |
| 20       | SemRect   | 0.45                | 0.95       | 0.55             | 0.85          | 0.75                   |
| 20       | SemRectFIM| 0.40                | 0.95       | 0.55             | 0.90          | 0.88                   |

## Citation

If you use this code in your research, please cite:

```
@article{yourname2023semrectfim,
  title={SemRectFIM: SNR-Aware Semantic Signature Protection for Deep Semantic Communication},
  author={Your Name and Co-authors},
  journal={arXiv preprint arXiv:xxxx.xxxxx},
  year={2023}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
