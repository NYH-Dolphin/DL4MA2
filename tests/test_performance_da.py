import numpy as np

def test_performance_da():
    # Generate random predictions and targets
    predictions = np.load('results/predictions_da.npy')
    targets = np.load('data/targets.npy', allow_pickle=True)

    # Calculate accuracy
    accuracy = np.mean(predictions.round() == targets)

    # Check that accuracy is above threshold
    threshold = 0.75
    assert accuracy > threshold, f"Accuracy {accuracy} is below threshold {threshold}"

    # Check that arrays are not identical
    assert not np.array_equal(predictions, targets), "Predictions and targets arrays are identical"
