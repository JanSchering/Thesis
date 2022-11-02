from typing import Callable, List, Tuple
import torch as t
import numpy as np
import matplotlib.pyplot as plt
import random
from IPython.display import Image
from IPython import display
from tqdm import tqdm
import time


def visualize_sequence(sequence: t.Tensor, framerate: float) -> None:
    """
    Visualize an infection spread sequence as an animated CA.

    sequence:   the sequence of grid states. Expected shape (num_steps, grid_height, grid_width).
    framerate:  the framerate of the animation.
    """
    for _, state in enumerate(sequence):
        plt.imshow(state, cmap="Greys", interpolation="nearest", vmin=0, vmax=1)
        plt.pause(framerate)


def visualize_transition(transition: t.Tensor) -> None:
    """
    Visualize a transition from one grid state to another.

    transition (t.Tensor): numpy matrix of shape (2, grid_height, grid_width).
    """
    fig, axs = plt.subplots(1, 2, figsize=(3, 3))
    for stateIdx, state in enumerate(transition):
        axs[stateIdx].imshow(
            state, cmap="Greys", interpolation="nearest", vmin=0, vmax=1
        )
        axs[stateIdx].get_xaxis().set_visible(False)
        axs[stateIdx].get_yaxis().set_visible(False)
    return fig


def visualize_batch(grids: t.Tensor) -> None:
    """
    Produce a plot of all current grid states.
    """
    fig, axs = plt.subplots(1, grids.shape[0], figsize=(3, 3))
    for gridIdx, grid in enumerate(grids):
        axs[gridIdx].imshow(grid, cmap="Greys", interpolation="nearest", vmin=0, vmax=1)
        axs[gridIdx].get_xaxis().set_visible(False)
        axs[gridIdx].get_yaxis().set_visible(False)
    plt.show()
