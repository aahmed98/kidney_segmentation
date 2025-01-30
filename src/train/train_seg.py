
import os
from typing import Any, Dict, List, Tuple

from src.preprocessing import get_kits_dataset, get_visceral_dataset
from PIL.ImagePath import Path

from monai.inferers.utils import sliding_window_inference
from monai.losses.dice import DiceLoss
from monai.metrics.meandice import compute_meandice
from monai.networks.layers.factories import Norm
from monai.networks.nets import UNet
from monai.transforms.post.array import AsDiscrete
import torch
from torch.nn import Module


def train_segmentation(
        config: Dict[str, Any],
        output_dir: Path) -> Tuple[Module, List, List]:
    """Trains segmentation model.

    Parameters
    ----------
    config: Dict[str, Any]
        Dictionary that stores data, model, and training parameters.
    output_dir: Path
        Path to directory where config, model, and job_record are saved.

    """
    label_subset = config['data_params']['label_subset']

    # load data
    dataset = config['dataset']
    if dataset == 'kits':
        train_ds, train_loader, train_transforms, val_ds, val_loader, val_transforms = get_kits_dataset(
            **config['data_params'])
    elif dataset == 'visceral':
        train_ds, train_loader, train_transforms, val_ds, val_loader, val_transforms = get_visceral_dataset(
            **config['data_params'])
    else:
        raise NotImplementedError(f"No dataset implementation found for {dataset}!")

    # load model
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model_type = config["model"]
    if model_type == "unet":
        model = UNet(
            out_channels=len(label_subset) + 1,
            norm=Norm.BATCH,
            **config['model_params']
        )
        model = torch.nn.DataParallel(model)
        model.to(device)
    else:
        raise NotImplementedError(f"No model implementation found for {model_type}!")

    # set up training params
    loss_fn = config['training_params']['loss_fn']
    if loss_fn == "dice":
        loss_function = DiceLoss(to_onehot_y=True, softmax=True)
    else:
        raise NotImplementedError(f"No loss implementation found for {loss_fn}!")

    optimizer = torch.optim.Adam(model.parameters(), 1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', verbose=True)

    max_epochs = config['training_params']['max_epochs']
    val_interval = config['training_params']['val_interval']
    best_metric = -1
    best_metric_epoch = -1
    epoch_loss_values = []
    metric_values = []
    # post_pred and post_label one-hot encode segmentations. only used to calculate dice
    post_pred = AsDiscrete(argmax=True, to_onehot=True, n_classes=len(label_subset) + 1)
    post_label = AsDiscrete(to_onehot=True, n_classes=len(label_subset) + 1)

    for epoch in range(max_epochs):
        print("-" * 10)
        print(f"epoch {epoch + 1}/{max_epochs}")
        model.train()
        epoch_loss = 0
        step = 0
        for batch_data in train_loader:
            step += 1
            inputs, labels = (
                batch_data["image"].to(device),
                batch_data["label"].to(device),
            )
            optimizer.zero_grad()
            outputs = model(inputs)  # [batch_sz * num_samples, num_classes, R, A, S]
            if epoch_loss == 0:  # prints at beginning of epoch
                print("Output shape:", outputs.shape)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            print(
                f"{step}/{len(train_ds) // train_loader.batch_size}, "
                f"train_loss: {loss.item():.4f}")
        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

        if (epoch + 1) % val_interval == 0:
            model.eval()
            with torch.no_grad():
                metric_sum = torch.zeros(len(label_subset))
                metric_count = 0
                for val_data in val_loader:
                    val_inputs, val_labels = (
                        val_data["image"].to(device),
                        val_data["label"].to(device),
                    )
                    roi_size = (160, 160, 160)
                    sw_batch_size = 4
                    val_outputs = sliding_window_inference(
                        val_inputs, roi_size, sw_batch_size, model)
                    val_outputs = post_pred(val_outputs)
                    val_labels = post_label(val_labels)
                    value = compute_meandice(
                        y_pred=val_outputs,
                        y=val_labels,
                        include_background=False,
                    )  # (batch_sz, num_classes)
                    metric_count += len(value)
                    metric_sum += value.sum(axis=0).cpu()  # (num_classes,)
                metric = metric_sum / metric_count  # (num_classes,)
                combined_metric = torch.mean(metric)  # best metric is average across classes
                metric_values.append(metric.tolist())
                if combined_metric > best_metric:
                    best_metric = combined_metric
                    best_metric_epoch = epoch + 1
                    torch.save(model.state_dict(), os.path.join(output_dir, "model.pth"))
                    print("saved new best metric model")
                print(
                    f"current epoch: {epoch + 1} current mean dice: {combined_metric:.4f}"
                    f"\nbest mean dice: {best_metric:.4f} "
                    f"at epoch: {best_metric_epoch}"
                )
            scheduler.step(combined_metric)

    # each sublist is val scores for one class
    metrics_per_class = [[x[i] for x in metric_values] for i in range(len(label_subset))]
    metrics_per_class = {key: metrics_per_class[i] for i, key in enumerate(
        dict(sorted(label_subset.items(), key=lambda item: item[1])))}

    return model, epoch_loss_values, metrics_per_class