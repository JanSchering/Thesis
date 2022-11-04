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
