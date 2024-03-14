from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
import numpy as np
import random
import cv2
import os


def flip_image(image):
    """
    Flip the given image horizontally using OpenCV.

    Parameters:
    -----------
    image : np.ndarray
        Input image array.

    Returns:
    --------
    np.ndarray
        Flipped image array.

    """
    # YOUR CODE HERE
    pass


def rotate_image(image, angle):
    """
    Rotate an image by a given angle in degrees.

    Parameters
    ----------
    image : numpy.ndarray
        The image to be rotated.
    angle : float
        The angle in degrees to rotate the image by.

    Returns
    -------
    numpy.ndarray
        The rotated image.

    Notes
    -----
    This function rotates the input image by the specified angle using
    OpenCV's `getRotationMatrix2D` and `warpAffine` functions.
    The border mode is set to `cv2.BORDER_REPLICATE`.

    """
    # YOUR CODE HERE
    pass


def augment_image(image):
    """
    Augments an image by randomly flipping it horizontally and rotating it
    by a random angle between -10 and 10 degrees.

    Parameters:
    -----------
    image : np.ndarray
        The input image to augment.

    Returns:
    --------
    np.ndarray
        The augmented image.

    """
    # Randomly flip the image horizontally
    if np.random.rand() > 0.5:
        image = flip_image(image)
    
    # Randomly rotate the image
    angle = np.random.uniform(-10, 10)
    image = rotate_image(image, angle)
    
    return image


def get_features_and_labels(dataset, conv_base):
    """
    Extracts features from a pre-trained convolutional base and 
    returns them along with their corresponding labels.

    Parameters
    ----------
    dataset : tf.data.Dataset
        The dataset containing the images and their corresponding labels.
    conv_base : keras.Model
        The pre-trained convolutional base used for feature extraction.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        A tuple containing two numpy arrays - the concatenated features and
        labels respectively.

    Notes
    -----
    This function expects that the input images have already been preprocessed
    according to the requirements of the pre-trained convolutional base.
    Specifically, it uses the `preprocess_input` function from the 
    VGG16 module to preprocess the images. 

    """
    # YOUR CODE HERE
    # Hint: you can get help from the course textbook, Chapter 8
    pass


# ============= Data Generation Functions: DO NOT CHANGE =============

def load_data(data_path):
    """
    Load image files and their corresponding labels from a given directory.

    Parameters
    ----------
    data_path : str
        Path to the directory containing the image files.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        A tuple of two numpy arrays.
        The first array contains the file paths of the image files,
        and the second array contains their corresponding labels
        (0 for cat and 1 for dog).

    """
    image_files = []
    labels = []
    for folder in os.listdir(data_path):
        folder_path = os.path.join(data_path, folder)
        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_name)
            image_files.append(file_path)
            if type(folder) is not str:
                folder = folder.decode('utf8')
            labels.append(1 if folder == 'dog' else 0)

    return image_files, labels


def data_generator(data_path, img_shape, augment, normalize, shuffle):
    """
    A generator function that yields batches of preprocessed images and
    their corresponding labels from a directory of images.

    Parameters:
    -----------
    data_path : str
        Path to the directory containing images.
    img_shape : tuple
        Shape to which the images will be resized to.
    augment : bool
        Whether to perform data augmentation on the images or not.
    normalize : bool
        Whether to normalize the pixel values of the images or not.
    shuffle : bool
        Whether to shuffle the data or not.

    Yields:
    -------
    Tuple[np.ndarray, np.ndarray]
        A tuple containing the preprocessed image and its corresponding label.

    Example:
    --------
    data_gen = data_generator('/path/to/images', (224, 224), True, True)
    images, labels = next(data_gen)

    """
    # Get list of image file names and their corresponding labels
    image_files, labels = load_data(data_path)
    
    # Convert labels to numpy array
    labels = np.array(labels)

    # Shuffle images and labels
    if shuffle:
        idxs = np.random.permutation(len(labels))
        image_files = [image_files[i] for i in idxs]
        labels = labels[idxs]

    for idx in range(len(image_files)):

        # Load image and label
        label = labels[idx]
        file_path = image_files[idx]
        img = cv2.imread(file_path.decode('utf-8'))

        # Correct BGR to RGB color
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Resize image to expected size
        img = cv2.resize(img, img_shape) 

        if augment:
          # Augment image
          img = augment_image(img)

        if normalize:
          # Normalize image to within [0,1]
          img = img / 255.

        yield img, label


def create_dataset(data_path, batch_size, img_shape, augment=False, normalize=True, shuffle=True):
    """
    Creates a TensorFlow dataset from image files in a directory.

    Parameters
    ----------
    data_path : str
        Path to directory containing the image files.
    batch_size : int
        Batch size for the returned dataset.
    img_shape : tuple
        Tuple of integers representing the desired image shape, e.g. (224, 224).
    augment : bool, optional
        Whether to apply image augmentation to the dataset.
        Default is False.
    normalize : bool, optional
        Whether to normalize the pixel values of the images to the range [0, 1].
        Default is True.
    shuffle : bool
        Whether to shuffle the data or not.
        Default is True.

    Returns
    -------
    dataset : tf.data.Dataset
        A TensorFlow dataset containing the images and their labels.

    """
    output_size = img_shape + (3,)
    dataset = tf.data.Dataset.from_generator(
        data_generator,
        args=[data_path, img_shape, augment, normalize, shuffle],
        output_signature=(
            tf.TensorSpec(shape=output_size, dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.uint8)))

    # Add augmented images
    if augment:
        dataset_aug = tf.data.Dataset.from_generator(
            data_generator,
            args=[data_path, img_shape, augment, normalize, shuffle],
            output_signature=(
                tf.TensorSpec(shape=output_size, dtype=tf.float32),
                tf.TensorSpec(shape=(), dtype=tf.uint8)
                )
            )
        dataset = dataset.concatenate(dataset_aug)

    dataset = dataset.batch(batch_size)

    return dataset


# ============= Auxiliary Functions: DO NOT CHANGE =============

def explore_data(train_ds, data_home, class_names):
    """
    Plots the distribution of classes in the training, validation, and test sets, and displays a sample of images from the
    training set.

    Parameters
    ----------
    train_ds : tf.data.Dataset
      A dataset object for the training set.
    data_home : str
      The directory path to the dataset.
    class_names : List[str]
      A list of class names.

    Returns
    -------
    None

    """
    # Plot the distribution of classes in the training, validation, and test sets
    fig, ax = plt.subplots(1, 3, figsize=(10, 5))

    # Plot the distribution of classes in the training set
    image_files, labels = load_data(os.path.join(data_home, 'train'))
    train_class_counts = labels
    ax[0].bar(range(len(class_names)), np.bincount(train_class_counts))
    ax[0].set_xticks(range(len(class_names)))
    ax[0].set_xticklabels(class_names, rotation=45)
    ax[0].set_title('Training set')

    # Plot the distribution of classes in the validation set
    image_files, labels = load_data(os.path.join(data_home, 'validation'))
    val_class_counts = labels
    ax[1].bar(range(len(class_names)), np.bincount(val_class_counts))
    ax[1].set_xticks(range(len(class_names)))
    ax[1].set_xticklabels(class_names, rotation=45)
    ax[1].set_title('Validation set')

    # Plot the distribution of classes in the test set
    image_files, labels = load_data(os.path.join(data_home, 'test'))
    test_class_counts = labels
    ax[2].bar(range(len(class_names)), np.bincount(test_class_counts))
    ax[2].set_xticks(range(len(class_names)))
    ax[2].set_xticklabels(class_names, rotation=45)
    ax[2].set_title('Test set')

    plt.show()


def plot_loss(history):
    """
    Plot the training and validation loss and accuracy.

    Parameters
    ----------
    history : keras.callbacks.History
      The history object returned by the `fit` method of a Keras model.

    Returns
    -------
    None

    """
    # Get metrics
    accuracy = history.history["accuracy"]
    val_accuracy = history.history["val_accuracy"]
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    
    # Get number of epochs for x-axis
    epochs = range(1, len(accuracy) + 1)

    # Plot
    plt.plot(epochs, accuracy, "bo", label="Training accuracy")
    plt.plot(epochs, val_accuracy, "b", label="Validation accuracy")
    plt.title("Training and validation accuracy")
    plt.legend()
    plt.figure()
    plt.plot(epochs, loss, "bo", label="Training loss")
    plt.plot(epochs, val_loss, "b", label="Validation loss")
    plt.title("Training and validation loss")
    plt.legend()
    plt.show()
