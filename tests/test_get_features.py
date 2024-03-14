import pytest
import numpy as np
import tensorflow as tf
from unittest.mock import Mock
from utils import get_features_and_labels


def test_get_features_and_labels():
    # Create a mock dataset
    images = np.random.rand(10, 224, 224, 3).astype(np.float32)
    labels = np.random.randint(0, 10, size=(10,), dtype=np.int64)
    dataset = tf.data.Dataset.from_tensor_slices((images, labels)).batch(2)

    # Create a mock convolutional base
    conv_base = Mock()

    # Configure the mock to return a fixed set of features for each batch of images
    features_shape = (2, 7, 7, 512)
    conv_base.predict.return_value = np.random.rand(*features_shape).astype(np.float32)

    # Compute features and labels using the function
    features, labels = get_features_and_labels(dataset, conv_base)

    # Check that the shapes of the returned arrays are correct
    assert features.shape == (10, 7, 7, 512)
    assert labels.shape == (10,)

    # Check that the features and labels have the correct type
    assert features.dtype == np.float32
    assert labels.dtype == np.int64

    # Check that the convolutional base was called once for each batch of images
    assert conv_base.predict.call_count == 5  # 5 = ceil(10 / 2)

# run the tests
if __name__ == '__main__':
    test_get_features_and_labels()
