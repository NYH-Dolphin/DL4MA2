import tensorflow as tf
import keras
from models import build_baseline


def test_build_baseline():
    input_shape = (180, 180, 3)
    model = build_baseline(input_shape)

    # Test object type 
    assert isinstance(model, keras.models.Model)

    # Test input and output shapes
    assert model.input_shape == (None,) + input_shape
    assert model.output_shape == (None, 1)

    # Test number of layers
    assert len(model.layers) == 11

    # Test Block 1
    assert isinstance(model.layers[0], keras.layers.Conv2D)
    assert model.layers[0].filters == 32
    assert model.layers[0].kernel_size == (3, 3)
    assert model.layers[0].activation == keras.activations.relu

    assert isinstance(model.layers[1], keras.layers.MaxPooling2D)
    assert model.layers[1].pool_size == (2, 2)

    # Test Block 2
    assert isinstance(model.layers[2], keras.layers.Conv2D)
    assert model.layers[2].filters == 64
    assert model.layers[2].kernel_size == (3, 3)
    assert model.layers[2].activation == keras.activations.relu

    assert isinstance(model.layers[3], keras.layers.MaxPooling2D)
    assert model.layers[3].pool_size == (2, 2)

    # Test Block 3
    assert isinstance(model.layers[4], keras.layers.Conv2D)
    assert model.layers[4].filters == 128
    assert model.layers[4].kernel_size == (3, 3)
    assert model.layers[4].activation == keras.activations.relu

    assert isinstance(model.layers[5], keras.layers.MaxPooling2D)
    assert model.layers[5].pool_size == (2, 2)

    # Test Block 4
    assert isinstance(model.layers[6], keras.layers.Conv2D)
    assert model.layers[6].filters == 256
    assert model.layers[6].kernel_size == (3, 3)
    assert model.layers[6].activation == keras.activations.relu

    assert isinstance(model.layers[7], keras.layers.MaxPooling2D)
    assert model.layers[7].pool_size == (2, 2)

    # Test convolutional layer
    assert isinstance(model.layers[8], keras.layers.Conv2D)
    assert model.layers[8].filters == 256
    assert model.layers[8].kernel_size == (3, 3)
    assert model.layers[8].activation == keras.activations.relu

    # Test remaining layers
    assert isinstance(model.layers[9], keras.layers.Flatten)
    assert isinstance(model.layers[10], keras.layers.Dense)
    assert model.layers[10].units == 1
    assert model.layers[10].activation == keras.activations.sigmoid

    # Test model properties
    assert model.loss == 'binary_crossentropy'
    assert (isinstance(model.optimizer, keras.optimizers.Adam)
        or isinstance(model.optimizer, keras.optimizers.legacy.Adam))
