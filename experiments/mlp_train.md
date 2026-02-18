- just changed from SoftmaxLinear model to TwoLayerMLP model and accuracy jumped from ~40% to ~46%.

- Learning rate changed from 0.0001 to 0.001. Why? Reason in README.md
- Accuracy jumped to ~52%.

- tested different weight decay not much difference.
- momentum + SGD still far better (6% better) than just SGD or adam(5%)

---

- finally, in the TwoLayerMLP model, tested the dim_classes
  - 128: ~51.2% accuracy
  - 256: ~52.1% accuracy
  - 512: ~52.4% accuracy
  - 1024: ~53.3% accuracy
  - 2048: ~53.6% accuracy (diminishing returns noted)
  - 2500: ~53.6% accuracy

- compute/cost scales linearly with hidden size (dim_classes)
