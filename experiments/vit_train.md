# ViT-B/16 on CIFAR-10

- vit_train.py: torchvision ViT-B/16, ImageNet pretrained
- Frozen backbone, head-only training (5 epochs)
- ~94% test accuracy
- Set FREEZE_BACKBONE = False for full fine-tuning
- Super low accuracy for full fine-tuning in first few epochs. Stopped it. Reasons:
  - Takes too much time
  - Needs much more epoch than freeze backbone.
- Also, learning rate should be 10x or more lower in full finetuning, huge step/changes by optimizer scrambles the pre-trained knowledge.
