# Overall Chronological Logs

note: the final accuracy in most experiments is incremental, for more details, see md file of specific experiment.

- number of epochs: 10

## Linear Model

- created linear.py model
- created reg_linear.py experiment
- ~40% accuracy

---

## Two Layer Neural Network (Linear + ReLU)

- created mlp.py model
- created mlp_train.py experiment (only 1 line different from reg_linear, using the mlp.py model)
- ~53% accuracy
- MLPs fail to exploit spatial structure in images, which motivates the use of CNNs.

  ### Learning rate difference: Linear (0.0001) vs MLP (0.001)
  - The linear classifier required a smaller learning rate because updates directly change the final decision boundary, making training unstable at higher step sizes.
  - The MLP allowed a higher learning rate since gradients are distributed across multiple layers and adjusted through ReLU activations, which stabilizes optimization.
  - Additionally, the hidden layer learns intermediate features, reducing sensitivity to large parameter updates and enabling faster convergence.

---

## Convolutional Neural Network (CNN)

- created cnn.py model (3 conv layers + 2 FC layers)
- created cnn_train.py experiment
- ~79% accuracy (with data augmentation and batchnorm)
- LR = 0.01

---

## Residual Network (ResNet)

- created resnet.py model (ResNet-14: 13 conv layers + 1 FC layer)
- created resnet_train.py experiment
- ~86% accuracy (10 epochs), ~92% accuracy (30 epochs)
- LR = 0.1 (higher LR enabled by skip connections which stabilize gradient flow in deeper networks)
- Skip connections allow training deeper networks without vanishing gradient problems.

### Performance Optimizations

DataLoader optimizations (`num_workers=2`, `pin_memory=True`, `non_blocking=True`) achieved 2x speedup: 30s → 15s per epoch.

Note: AMP (16-bit precision) tested but provided no benefit for this small model. May help larger models where computation is the bottleneck.

---

## Vision Transformer (ViT-B/16)

- created vit_train.py experiment (torchvision ViT-B/16, ImageNet pretrained)
- ~94% accuracy with frozen backbone, head-only training (5 epochs)
- See experiments/README_VIT.md for details, options, and how to run.
