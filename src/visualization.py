import json
from pathlib import Path
from typing import Any, Dict, List
from matplotlib import pyplot as plt
import numpy as np
import torch
import random
from matplotlib import colors
from matplotlib.figure import Figure


def max_label_slice(label_tensor: torch.Tensor, label_filter: int = None) -> int:
    """Finds axial slice with highest proportion of labels.

    Good for comparing predictions and ground truth.

    Parameters
    ----------
    label_tensor: torch.Tensor (shape: (R, A, S))
        3D tensor of labels
    label_filter: int
        Numerical encoding of specific label. Use if you want to only look at highest proportion of only one label.
        Otherwise, all labels are treated equally

    Returns
    -------
    int
        Slice # of axial slice with highest proportion of labels
    """
    num_slices = label_tensor.shape[-1]
    label_pixels = np.zeros(num_slices)
    for i in range(num_slices):
        slice = label_tensor[0, 0, :, :, i]
        if label_filter is not None:
            num_label_pixels = torch.sum(torch.eq(slice, label_filter)).item()
        else:
            num_label_pixels = len(np.nonzero(slice))
        label_pixels[i] = num_label_pixels
    return int(np.argmax(label_pixels))


def plot_training_output(checkpoint: Path) -> Figure:
    """Convenience method for visualizing training output from a checkpoint.

    Extracts relevant information from checkpoint and calls plot_training_val_error.

    Parameters
    ----------
    checkpoint: Path
        Checkpoint where training data, config file, etc. is saved
    """
    training_outputs_path = checkpoint / "training_outputs.json"
    with training_outputs_path.open('r') as jf:
        training_outputs = json.load(jf)
    config_path = checkpoint / "config.json"
    with config_path.open('r') as jf:
        config = json.load(jf)
    epoch_loss_values = training_outputs['epoch_loss_values']
    val_metric_values = training_outputs['val_metric_values']
    label_subset = config['data_params']['label_subset']
    val_interval = config['training_params']['val_interval']
    return plot_training_val_error(
        epoch_loss_values, val_metric_values, label_subset, val_interval
    )


def plot_training_val_error(
    epoch_loss_values: List,
    metric_values: Dict[str, List],
    label_subset: Dict[str, int],
    val_interval: int = 2
) -> Figure:
    """ Plots training error and validation metrics over course of training

    Validation metrics are divided on a class basis.

    Parameters
    ----------
    epoch_loss_values: List
        Training loss per epoch
    metric_values: Dict [str, List] (List shape: [epochs/$val_interval])
        Dictionary mapping classes to their validation metric values during training.
    label_subset: Dict[str, int]
        Subset of labels that were trained on, mapped to their desired numerical encoding. Should be same size as
        2nd dimension of $metric_values.
    val_interval: int
        How often validation metrics are computed (in epochs)
    """
    fig = plt.figure("train", (12, 6))
    plt.subplot(1, 2, 1)
    plt.title("Epoch Average Loss")
    x = [i + 1 for i in range(len(epoch_loss_values))]
    y = epoch_loss_values
    plt.xlabel("epoch")
    plt.plot(x, y)
    plt.subplot(1, 2, 2)
    plt.title("Val Mean Dice")
    x = [val_interval * (i + 1) for i in range(len(metric_values))]
    plt.xlabel("epoch")
    for label, y in metric_values.items():
        plt.plot(x, y, label=label)
    plt.legend()
    return fig


def plot_slices(slices: Dict[str, Any], label_filter: int = None) -> None:
    """Plots each slice in $slices side by side.

    If label slice is available, max_label_slice is called to find the slice with max
    proportion of labels. Otherwise, center slice is chosen.

    "Label" and "output" items are shown on a color scale. All others are shown on gray scale.

    All tensor/array values are automatically squeezed to have 3D shape.

    Parameters
    ----------
    slices: Dict[str, Any]
        Dictionary mapping slice type to tensors or Arrays.
        Example: {"image": torch.Tensor(shape: [1, 1, 128, 128, 96]),
                  "label": np.ndarray(shape: [1, 128, 128, 96])}
    label_filter: int
        Numerical encoding of label that will exclusively be shown if passed in as argument.

    """
    colors_ = ['k', '#00c8cf', 'y']
    cmap = colors.ListedColormap(colors_)

    if 'label' in slices:
        label_slice = max_label_slice(slices['label'], label_filter)
    else:
        slice = random.choice(slices.values())  # gets one of the slices to get the shape
        label_slice = slice.shape[-1] // 2  # last dimension is superior-inferior

    n_slices = len(slices)
    fig = plt.figure("slices", (n_slices * 6, 6))
    for i, (slice_name, slice) in enumerate(slices.items()):
        ax = plt.subplot(1, n_slices, i + 1)
        plt.title(f"{slice_name}")
        slice = torch.squeeze(slice)[:, :, label_slice].T

        if slice_name in {"label", "output"}:
            # filter label/output based on label_filter, if applicable
            if label_filter is not None:
                zeros = torch.zeros(slice.shape)
                slice = torch.where(slice == label_filter, slice, zeros)
            plt.imshow(slice, origin='lower', cmap=cmap, vmin=0, vmax=len(colors_))
        else:
            plt.imshow(slice, cmap="gray", origin='lower')
        ax.invert_xaxis()

    plt.show()
    return fig