# Neural network from scratch

One script: `nn_scratch.py`. It follows Lecture 3 and Lab 3 task 4:
one hidden layer of tanh units, a sigmoid output, cross entropy, and the backpropagation lines.

## What it does
1. Standardising the features with the training mean only.
2. Writing the forward pass and the backward pass ourselves.
3. Checking backpropagation with finite differences and asserts.
4. Training with gradient descent and keeping the step with the best validation cost.
5. Searching hidden units, purchase weights, ridge strength and thresholds on validation data.
6. Testing the selected models once and comparing with the scikit-learn MLPClassifier.

## How to run
From the project root (it takes about 2 minutes):

    python -m neural_network.nn_scratch
    PAGE_VALUES=1 python -m neural_network.nn_scratch

## What we select
Same three models as for logistic regression: plain, best F1, and max precision with recall of at least 0.5.

Results are saved in `test_result/neural_network/`.
