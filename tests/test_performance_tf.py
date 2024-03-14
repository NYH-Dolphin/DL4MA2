import numpy as np

def test_performance_tf():
    # Generate random predictions and targets
    n_samples = 100
    predictions = np.load('results/predictions_tl.npy')
    targets = np.load('data/targets.npy', allow_pickle=True)

    # Calculate accuracy
    accuracy = np.mean(predictions.round() == targets)

    # Check that accuracy is above threshold
    threshold = 0.94
    assert accuracy > threshold, f"Accuracy {accuracy} is below threshold {threshold}"

    # Check that arrays are not identical
    assert not np.array_equal(predictions, targets), "Predictions and targets arrays are identical"
