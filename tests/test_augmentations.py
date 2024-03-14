import pytest
import numpy as np
from utils import flip_image, rotate_image


def test_flip_image():
    # create a flipped version of the test image
    img = np.load('data/original_image.npy')
    expected_image = np.load('data/flipped_image.npy')
    flipped_image = flip_image(img)

    # check that the flipped image is correct
    assert np.allclose(flipped_image, expected_image)


def test_rotate_image():
    # create a rotated version of the test image
    img = np.load('data/original_image.npy')
    expected_image = np.load('data/rotated_image.npy')
    rotated_image = rotate_image(img, 45)

    # check that the rotated image is correct
    assert np.allclose(rotated_image, expected_image)


# run the tests
if __name__ == '__main__':
    test_flip_image()
    test_rotate_image()
