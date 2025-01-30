from typing import Any, Dict
from pathlib import Path

from src.preprocessing import get_kits_dataset, get_visceral_dataset
from src.records import save_job_record
from src.visualization import plot_slices, plot_training_output
from monai.data import NiftiSaver
from monai.inferers.utils import sliding_window_inference
from monai.metrics.meandice import compute_meandice
from monai.networks.layers.factories import Norm
from monai.networks.nets import unet
from monai.transforms.compose import Compose
from monai.transforms.post.array import AsDiscrete
from monai.transforms.utils import allow_missing_keys_mode
import torch
from monai.data.inverse_batch_transform import BatchInverseTransform


def infer_segmentation(
    infer_config: Dict[str, Any],
    training_config: Dict[str, Any],
    training_checkpoint: Path,
    output_dir: Path
) -> None:
    """Performs inference using model trained on segmentation task.

    Parameters
    ----------
    infer_config: Dict[str, Any]
        Dictionary that stores parameters for inference.
    training_config: Dict[str, Any]
        Dictionary that stores data, model, and training parameters.
    training_checkpoint: Path
        Path to directory that stores saved model and training outputs.
    output_dir: Path
        Path to directory where config, model, and job_record are saved.
    """
    train_label_subset = training_config['data_params']['label_subset']  # for loading model
    infer_label_subset = infer_config['data_params']['label_subset']  # for viz/evaluation

    # load data
    dataset = infer_config['dataset']
    if dataset == 'kits':
        _, _, _, _, val_loader, val_transforms = get_kits_dataset(
            **infer_config['data_params'])
    elif dataset == 'visceral':
        _, _, _, _, val_loader, val_transforms = get_visceral_dataset(
            **infer_config['data_params'])
    else:
        raise NotImplementedError(f"No dataset implementation found for {dataset}!")

    # load model
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model_type = training_config["model"]
    if model_type == "unet":
        model = unet(
            out_channels=len(train_label_subset) + 1,
            norm=Norm.BATCH,
            **training_config['model_params']
        )
        model = torch.nn.DataParallel(model)
        model.to(device)
    else:
        raise NotImplementedError(f"No model implementation found for {model_type}!")

    model.load_state_dict(torch.load(training_checkpoint / "model.pth"))
    model.eval()

    if infer_config["visualize_training_output"]:
        fig = plot_training_output(training_checkpoint)
        img_path = output_dir / "train_val_error.png"
        fig.savefig(img_path)

    if infer_config["evaluate_val_predictions"]:
        val_output_dir = output_dir / "val_outputs"
        val_output_dir.mkdir(parents=True)
        saver = NiftiSaver(output_dir=val_output_dir, output_postfix='seg')

        # post_pred and post_label one-hot encode segmentations. only used to calculate dice
        post_pred = AsDiscrete(argmax=True, to_onehot=True, n_classes=len(infer_label_subset) + 1)
        post_label = AsDiscrete(to_onehot=True, n_classes=len(infer_label_subset) + 1)

        with torch.no_grad():
            metric_sum = torch.zeros(len(infer_label_subset))
            metric_count = 0
            for _, val_data in enumerate(val_loader):
                # run model on val input
                val_input, val_label = (
                    val_data["image"].to(device),
                    val_data["label"].to(device),
                )
                roi_size = (160, 160, 160)
                sw_batch_size = 4
                val_output = sliding_window_inference(
                    val_input, roi_size, sw_batch_size, model
                )
                # [batch_sz, R, A, S]
                segmentation = torch.argmax(val_output, dim=1).detach().cpu().type(torch.FloatTensor)
                val_output = post_pred(val_output)
                val_label = post_label(val_label)
                value = compute_meandice(
                    y_pred=val_output,
                    y=val_label,
                    include_background=False,
                )  # (batch_sz, num_classes)
                metric_count += len(value)
                metric_sum += value.sum(axis=0).cpu()  # (num_classes,)

                # save segmentation
                file_name = val_data["image_meta_dict"]['filename_or_obj'][0]
                case_num = file_name.split("/")[-2]  # based on enforced structure of data
                saver.save(segmentation, {'filename_or_obj': case_num})

                # side by side of image, label, and output
                fig = plot_slices({
                    "image": val_data["image"],
                    "label": val_data["label"],
                    "output": segmentation
                }, label_filter=infer_label_subset["kidney"])

                # save side by side plot
                img_path = val_output_dir / case_num / "side_by_side.png"
                fig.savefig(img_path)

            # save mean dice scores
            dice_per_class = metric_sum / metric_count  # (num_classes,), mean dice per class
            dice_per_class = dice_per_class.tolist()
            dice_per_class = {key: dice_per_class[i] for i, key in enumerate(
                dict(sorted(infer_label_subset.items(), key=lambda item: item[1])))}  # sort label subset by value
            dice_per_class = {
                "dice_per_class": dice_per_class
            }
            save_job_record(output_dir, "val_metrics.json", dice_per_class)


# TODO: fix resampling and saving
def resample_and_save_seg(
    segmentation: torch.Tensor,
    val_data: Dict[str, Any],
    saver: NiftiSaver,
    inverter: BatchInverseTransform,
    val_transforms: Compose
) -> None:
    data_fwd = {"label": segmentation, "label_transforms": val_data['label_transforms']}
    with allow_missing_keys_mode(val_transforms):
        data_fwd_inv = inverter(data_fwd)
    print("inverse seg batch shape: ", data_fwd_inv[0]["label"].shape)
    metadata = data_fwd_inv[0]['label_meta_dict']
    affine_metadata = {
        'original_affine': metadata['original_affine'],
        'affine': metadata['affine'],
        'spatial_shape': torch.squeeze(metadata['spatial_shape'])
    }
    return affine_metadata