import torch as t
import torch.nn.functional as F
from torchvision import transforms
import torchmetrics

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from PIL import Image
from tqdm import tqdm

import os
from os import path, getcwd, listdir, mkdir

import sys

args = sys.argv
print(args)
load_trace = args[1] == "load" if len(sys.argv) > 1 else False
print(f"loading trace? {load_trace}")
subj = args[2] if len(sys.argv) > 2 else "seadragon"
img_filetype = args[3] if len(sys.argv) > 3 else "jpg"
labels_filetype = args[4] if len(sys.argv) > 4 else "png"


device = t.device("cuda") if t.cuda.is_available() else t.device("cpu")


class STEFunction(t.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        input[input > 0] = 1
        return input.float()

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


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


def unfold_conv(batch: t.Tensor) -> t.Tensor:
    unfold_transform = t.nn.Unfold(kernel_size=3)
    # provide a periodic torus padding to the grid
    padded_batch = periodic_padding(batch)
    # we need to add a channel dimension because unfold expects vectors of shape (N,C,H,W)
    padded_batch = padded_batch.unsqueeze(1)
    # apply the unfolding operator on the padded grid, this provides all convolution blocks
    unfolded_batch = unfold_transform(padded_batch)
    # turn each convolution block into a row
    batch_reshaped = unfolded_batch.permute(0, 2, 1)
    return batch_reshaped


def bg_contacts(conv_mat: t.Tensor) -> t.Tensor:
    num_convs = conv_mat.shape[1]
    return t.sum((1 - conv_mat) * conv_mat[:, :, 4].T.expand(num_convs, 9))


def fg_contacts(conv_mat):
    num_convs = conv_mat.shape[1]
    return t.sum(conv_mat * (1 - conv_mat[:, :, 4]).T.expand(num_convs, 9))


def grid2gsc(grid, scale_min, scale_max, scale_mean, scale_std):
    mean_sampler = t.distributions.Normal(scale_mean, scale_std)
    reconstruction = t.zeros_like(grid)
    reconstruction += grid * mean_sampler.sample(reconstruction.shape)
    reconstruction = reconstruction.clamp(0, 1)
    return reconstruction


def reconstruction_loss(ref_pic, rec_pic):
    return t.log(t.sum((ref_pic - rec_pic) ** 2))


def loss_fn(
    grid, ref_pic, lambda_bg=0.05, lambda_rec=1, lambda_reg=0.01, lambda_fg=0.05
):
    rec_pic = grid2gsc(
        grid, t.min(ref_pic), t.max(ref_pic), t.mean(ref_pic), t.std(ref_pic)
    )
    lattice = STEFunction.apply(grid)

    conv_mat = unfold_conv(lattice)
    return (
        bg_contacts(conv_mat) * lambda_bg
        + reconstruction_loss(ref_pic, rec_pic) * lambda_rec
        + t.sum(grid) * lambda_reg
        + fg_contacts(conv_mat) * lambda_fg
    )


data_path = path.join(getcwd(), "data")
image_path = path.join(data_path, subj, f"img.{img_filetype}")
res_path = path.join(getcwd(), "results", subj)
img = Image.open(image_path)

labels_path = path.join(data_path, subj, f"labels.{labels_filetype}")
labels = np.asarray(Image.open(labels_path)) / 255

img_gsc = img.convert("L")
test_img = np.asarray(img_gsc) / 255
grid = t.zeros_like(t.tensor(test_img))
grid = grid.unsqueeze(0)
ref = t.from_numpy(test_img)
np.save(path.join(data_path, subj, "grayscale.npy"), ref.cpu().numpy())

ref = ref.unsqueeze(0)
max_intensity = t.max(ref)
min_intensity = t.min(ref)
mean_intensity = t.mean(ref)
std_intensity = t.std(ref)

centering = transforms.Normalize(mean_intensity, std_intensity)
ref_centered = centering(ref)
center2 = (ref_centered - t.min(ref_centered)) / (
    t.max(ref_centered) - (t.min(ref_centered))
)
np.save(path.join(data_path, subj, "centered_normed.npy"), center2.cpu().numpy())

ref_centered = ref_centered.to(device).type(t.float)
dice = torchmetrics.Dice(average="micro").to(device)
jaccard = torchmetrics.JaccardIndex(task="binary").to(device)

if load_trace:
    losses = np.load(path.join(res_path, "losses.npy")).tolist()
    dice_scores = np.load(path.join(res_path, "dice_scores.npy")).tolist()
    jaccard_scores = np.load(path.join(res_path, "jaccard_scores.npy")).tolist()
    center2 = (
        t.tensor(np.load(path.join(res_path, "grid.npy"))).to(device).type(t.float)
    )
else:
    losses = []
    dice_scores = []
    jaccard_scores = []
    center2 = (ref_centered - t.min(ref_centered)) / (
        t.max(ref_centered) - (t.min(ref_centered))
    ).type(t.float)


center2.requires_grad_()
for i in tqdm(range(101)):
    loss = loss_fn(
        center2,
        ref_centered,
        lambda_bg=0.01,
        lambda_rec=5,
        lambda_reg=0.01,
        lambda_fg=0.1,
    )
    losses.append(loss.detach().cpu().numpy())
    grads = t.autograd.grad(loss, center2)[0]
    center2 = t.clamp(center2 - 0.01 * grads, min=0, max=1)
    with t.no_grad():
        np.save(
            path.join(res_path, "grid_states", f"grid_{i}.npy"),
            center2.detach().cpu().numpy(),
        )
    if i % 20 == 0:
        with t.no_grad():
            thres_img = center2.detach().clone()
            thres_img[center2 > 0] = 1
            jaccard_scores.append(
                jaccard(thres_img[0], t.tensor(labels).to(device))
                .detach()
                .cpu()
                .numpy()
            )
            dice_scores.append(
                dice(thres_img[0], t.tensor(labels).type(t.int).to(device))
                .detach()
                .cpu()
                .numpy()
            )

np.save(path.join(res_path, "losses.npy"), np.array(losses))
np.save(path.join(res_path, "dice_scores.npy"), np.array(dice_scores))
np.save(path.join(res_path, "jaccard_scores.npy"), np.array(jaccard_scores))
np.save(path.join(res_path, "grid.npy"), center2.detach().cpu().numpy())
