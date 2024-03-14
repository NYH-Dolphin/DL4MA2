import tensorflow as tf
import keras


def build_dense_model(input_shape):
    """
    Build and compile a dense neural network model with a 
    sigmoid output layer for binary classification.

    Parameters
    ----------
    input_shape : tuple
        The shape of the input tensor, excluding the batch size.

    Returns
    -------
    tf.keras.Model
        The compiled dense neural network model.

    Notes
    -----
    The dense neural network model consists of a flatten layer,
    a dense layer with 256 units, a dropout layer with 0.5 dropout rate,
    and a sigmoid output layer.
    The binary cross-entropy loss function is used for training,
    and the Adam optimizer is used for optimization.
    The model is evaluated based on the accuracy metric during training.

    Examples
    --------
    >>> model = build_dense_model(input_shape=(28, 28, 1))
    >>> model.summary()

    """
    # YOUR CODE HERE
    pass


def load_conv_model(weights='imagenet', include_top=False, input_shape=(180, 180, 3)):
    """
    Loads the VGG16 convolutional base model.

    Parameters
    ----------
    weights : str or None, optional
        Specifies which weights to load for the model. It can be either 'imagenet'
        (pre-training on ImageNet) or None (random initialization).
        The default value is 'imagenet'.
    include_top : bool, optional
        Whether to include the fully connected layers at the top of the network.
        The default value is False, which means the last fully connected layers
        are excluded.
    input_shape : tuple of int, optional
        The shape of the input tensor to the model.
        The default shape is (180, 180, 3).

    Returns
    -------
    A Keras model object.

    """
    # Hint: use keras.applications
    # YOUR CODE HERE
    pass


def build_baseline(input_shape):
    """
    Parameters
    ----------
    input_shape : tuple of int
        The shape of the input tensor, e.g. (height, width, channels).

    Returns
    -------
    model : keras.Model
        The compiled baseline convolutional neural network model.

    Notes
    -----
    The model architecture consists of four convolutional blocks consisting
    of one convolutional and one max pooling layer:
    * Block 1: 32 filters, kernel size 3, activation function ReLU, pool size 2.
    * Block 2: 64 filters, kernel size 3, activation function ReLU, pool size 2.
    * Block 3: 128 filters, kernel size 3, activation function ReLU, pool size 2.
    * Block 4: 256 filters, kernel size 3, activation function ReLU, pool size 2.

    The output of the fourth convolutional block is followed by a convolutional network
    with 256 filters, kernel size 3, and activation function ReLU.
    The output of this convolutional layer is then flattened and connected
    to a dense layer with one neuron, which is activated by the sigmoid function.
    The model is compiled with binary crossentropy loss, the Adam optimizer,
    and the accuracy metric.

    """
    # YOUR CODE HERE
    pass


def build_reg_model(input_shape):
    """
    Parameters
    ----------
    input_shape : tuple of int
        The shape of the input tensor, e.g. (height, width, channels).

    Returns
    -------
    model : keras.Model
        The compiled baseline convolutional neural network model.

    Notes
    -----
    The model architecture is identical to the baseline but it has a 
    Dropout layer with a 0.5 dropout rate between the dense and flatten layers.

    """
    # YOUR CODE HERE
    pass
