import torch as t


def periodic_padding(image: t.Tensor, padding=1):
    """
    Create a periodic padding (wrap) around an image stack, to emulate periodic boundary conditions
    Adapted from https://github.com/tensorflow/tensorflow/issues/956

    If the image is 3-dimensional (like an image batch), padding occurs along the last two axes
    """
    if len(image.shape) == 2:
        upper_pad = image[-padding:, :]
        lower_pad = image[:padding, :]

        partial_image = t.cat([upper_pad, image, lower_pad], dim=0)

        left_pad = partial_image[:, -padding:]
        right_pad = partial_image[:, :padding]

        padded_image = t.cat([left_pad, partial_image, right_pad], dim=1)

    elif len(image.shape) == 3:
        upper_pad = image[:, -padding:, :]
        lower_pad = image[:, :padding, :]

        partial_image = t.cat([upper_pad, image, lower_pad], dim=1)

        left_pad = partial_image[:, :, -padding:]
        right_pad = partial_image[:, :, :padding]

        padded_image = t.cat([left_pad, partial_image, right_pad], axis=2)

    else:
        assert True, "Input data shape not understood."

    return padded_image


if __name__ == "__main__":
    print("testing the periodic padding function...")
    size = 9
    batch_size = 2

    batch = t.zeros((batch_size, size, size))

    batch[0, 0, 0] = 1
    batch[0, 0, 1] = 1
    batch[0, 0, 2] = 1
    batch[0, 1, 0] = 1
    batch[0, 1, 2] = 1
    batch[0, 2, 0] = 1
    batch[0, 2, 1] = 1
    batch[0, 2, 2] = 1

    padded_batch = periodic_padding(batch)

    # Torus wrapping around the right edge providing active padding
    assert padded_batch[0, 1, -1] == 1.0
    assert padded_batch[0, 2, -1] == 1.0
    assert padded_batch[0, 3, -1] == 1.0
    # Below and above should not be active
    assert padded_batch[0, 0, -1] == 0.0
    assert padded_batch[0, 4, -1] == 0.0
    # Torus wrapping around the bottom providing active cells
    assert padded_batch[0, -1, 1] == 1.0
    assert padded_batch[0, -1, 2] == 1.0
    assert padded_batch[0, -1, 1] == 1.0
    # Left and right should not be active
    assert padded_batch[0, -1, 0] == 0.0
    assert padded_batch[0, -1, 4] == 0.0
    # Torus wrapping around the left edge should be inactive
    assert padded_batch[0, 0, 0] == 0.0
    assert padded_batch[0, 1, 0] == 0.0
    assert padded_batch[0, 2, 0] == 0.0
    assert padded_batch[0, 3, 0] == 0.0
    assert padded_batch[0, 4, 0] == 0.0
    # Bottom right corner should be active
    assert padded_batch[0, -1, -1] == 1.0

    print("all tests passed successfully")
