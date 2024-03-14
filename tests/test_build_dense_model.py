import tensorflow as tf
import keras
from models import build_dense_model


def test_build_dense_model():
    input_shape = (224, 224, 3)
    model = build_dense_model(input_shape)

    # Test object type
    assert isinstance(model, keras.models.Model)

    # Test input and output shapes
    assert model.input_shape == (None,) + input_shape
    assert model.output_shape == (None, 1)

    # Test layer types
    assert isinstance(model.layers[0], keras.layers.Flatten)
    assert isinstance(model.layers[1], keras.layers.Dense)
    assert isinstance(model.layers[2], keras.layers.Dropout)
    assert isinstance(model.layers[3], keras.layers.Dense)

    # Test layer properties
    assert model.layers[0].input_shape == (None,) + input_shape
    assert model.layers[0].output_shape == (None, 150528)
    assert model.layers[1].units == 256
    assert model.layers[3].units == 1
    assert model.layers[3].activation == keras.activations.sigmoid

    # Test model properties
    assert model.loss == 'binary_crossentropy'
    assert (isinstance(model.optimizer, keras.optimizers.Adam)
        or isinstance(model.optimizer, keras.optimizers.legacy.Adam))
