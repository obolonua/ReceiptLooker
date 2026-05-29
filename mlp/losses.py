import numpy as np


# Categorical cross-entropy measures how well predicted class probabilities match the true labels.
class CategoricalCrossEntropy:

    # Clip predictions for numerical stability, then compute the average negative log likelihood.
    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
        correct_confidences = y_pred_clipped[
            range(samples),
            y_true
        ]
        negative_log_likelihoods = -np.log(correct_confidences)
        return np.mean(negative_log_likelihoods)


if __name__ == "__main__":
    y_pred = np.array([
        [0.05, 0.10, 0.85],
        [0.70, 0.20, 0.10],
        [0.02, 0.15, 0.83],
    ])
    y_true = np.array([2, 0, 2])

    loss = CategoricalCrossEntropy().forward(y_pred, y_true)

    print("predictions:")
    print(y_pred)
    print("\nlabels:")
    print(y_true)
    print("\nloss:")
    print(loss)


"""
Categorical Cross-Entropy Loss
==============================

Purpose
-------
Measures how well the predicted class probabilities match the true labels.

The lower the loss:
    - the better the predictions
    - the more confident the network is about the correct classes

The higher the loss:
    - the more incorrect the predictions are
    - especially when the model is confidently wrong

Example
-------

Predicted probabilities:

    [
        [0.05, 0.10, 0.85],
        [0.70, 0.20, 0.10],
        [0.02, 0.15, 0.83]
    ]

True labels:

    [2, 0, 2]

Step 1: Select the probability assigned to each correct class

    Sample 1 -> class 2 -> 0.85
    Sample 2 -> class 0 -> 0.70
    Sample 3 -> class 2 -> 0.83

Result:

    [0.85, 0.70, 0.83]

Step 2: Compute negative log likelihood

    -log(0.85) ≈ 0.163
    -log(0.70) ≈ 0.357
    -log(0.83) ≈ 0.186

Result:

    [0.163, 0.357, 0.186]

Step 3: Compute average loss

    (0.163 + 0.357 + 0.186) / 3

    ≈ 0.235

Final loss:

    0.235

Why use logarithms?
-------------------

Good prediction:

    probability = 0.99

    loss = -log(0.99)

    ≈ 0.01

Almost no penalty.

Bad prediction:

    probability = 0.10

    loss = -log(0.10)

    ≈ 2.30

Large penalty.

Terrible prediction:

    probability = 0.001

    loss = -log(0.001)

    ≈ 6.91

Huge penalty.

Numerical Stability
-------------------

The probabilities are clipped before applying the logarithm:

    np.clip(y_pred, 1e-7, 1 - 1e-7)

This prevents:

    log(0)

which would produce:

    -inf

and break training.

Role in Neural Networks
-----------------------

Typical classification pipeline:

    Input
      ↓
    MLP
      ↓
    Softmax
      ↓
    Class Probabilities
      ↓
    Cross-Entropy Loss
      ↓
    Backpropagation
      ↓
    Weight Updates

The loss tells the network how wrong its predictions are.
Backpropagation then uses this information to adjust the weights and improve future predictions.
"""