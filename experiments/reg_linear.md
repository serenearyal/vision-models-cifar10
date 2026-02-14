# CIFAR-10 Linear Classifier Optimization Log

## Stage 1: Baseline

- Model: `SoftmaxLinear` (linear softmax classifier)
- Optimizer: SGD, no momentum
- Learning rate: 0.01
- Regularization: None
- Outcome: Initial performance ~30-35%; room for improvement

---

## Stage 2: Added Regularization

- Added weight decay (L2 regularization)
- Decreased learning rate from 0.01 → 0.001
- Outcome: Marginal improvement

---

## Stage 3: Train/Validation/Test Split

- Split training data into train, validation, and test sets
- Performed weight decay hyperparameter search
- Outcome: Found best weight decay; performance improved slightly

---

## Stage 4: Added Momentum

- Tested optimizers:
  - SGD + momentum > SGD ~= Adam > RMSprop
- Optimizer Used: SGD + momentum (0.9)
- Outcome: Slight improvement; training stabilized

---

## Stage 5: Input Normalization

- Applied standard CIFAR-10 normalization:

```python
transforms.Normalize(mean=[0.4914,0.4822,0.4465],
                     std=[0.2470,0.2435,0.2616])
```

- Decreased LR to 0.0001

- Current Accuracy ~40%
