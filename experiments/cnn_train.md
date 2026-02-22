- Architecture: 3 convolutional layers + 2 FC layers
- First baseline accuracy acheived ~72%
  - LR = 0.01
  - weight_decay = 1e-5
  - batch_size = 128

- data augmentation (random crop & random horizontal flip):
  - increased accuracy by ~3-4% to ~75.9%

- added batchnorm; increased accuracy by ~3% to ~78.9%

- added dropout; slight less accuracy.
  - probably because only 10 epochs
