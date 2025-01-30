from pathlib import Path
from typing import List, Dict, Mapping, Hashable, Tuple
import src.constants as constants
import src.locations as locations
from monai.transforms import (
    AddChanneld,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandCropByPosNegLabeld,
    ScaleIntensityRanged,
    Spacingd,
    ToTensord,
    RandAffined,
    RandShiftIntensityd
)
from monai.transforms.transform import MapTransform
from monai.config import KeysCollection
from monai.data import CacheDataset, DataLoader, Dataset
import numpy as np
import torch
import os
import glob
import math


class LabelFilterd(MapTransform):
    """Dictionary-based transform that selects subset of original labels.

    This transform assumes the ``data`` dictionary has a key for the input.
    data's metadata and contains `affine` field.  The key is formed by ``key_{meta_key_postfix}``.

    Basically perform a torch.where() operation that only selects user-specified labels
    (e.g. {'kidney': 1.0}) and discards the rest``.

    Example1:
        old_labels_dict = {'kidney': 1, 'ureter': 2, 'artery': 3}
        label_subset = {'kidney': 1, 'ureter': 2}
        new_labels_dict = {'kidney': 'kidney', 'ureter': 'ureter'}

        result: output label tensor only has kidney and ureter labels mapped to 1 and 2, respectively.
    Example2:
        old_labels_dict = {'left_kidney': 9, 'right_kidney': 10, 'liver':11}
        label_subset = {'kidney': 1}
        new_labels_dict = {'left_kidney': 'kidney', 'right_kidney': 'kidney'}

        result: output label tensor has single kidney label (aggregation of left_kidney + right_kidney) mapped to 1.

    Parameters
    ----------
    old_labels_dict: Dict[str, int]
        Mapping from anatomical structures to numbers used in labels.
    label_subset: Dict[str, int]
        Subset of new labels that will be trained on mapped to their desired numerical encoding.
        Note that if keys do not match old_labels_dict, new_labels_dict must specify mapping from old labels
        to new labels.
    new_labels_dict: Dict[str, str]
        Mapping from keys of old_labels_dict to keys of label_subset. Defaults to old_labels_dict if None
        is passed. Example of when this is necessary is the visceral dataset:
        old_label_dict = {"left_kidney": 9, "right_kidney": 10}
        label_subset = {"kidney": 1}
        new_labels_dict = {"left_kidney": "kidney", "right_kidney": "kidney"}
    """

    def __init__(
        self,
        keys: KeysCollection,
        old_labels_dict: Dict,
        label_subset: List,
        new_labels_dict: Dict = None,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        if new_labels_dict is None:  # assume keys of label_subset match old_labels_dict
            new_labels_dict = {label: label for label in label_subset}

        assert set(new_labels_dict.keys()).issubset(set(old_labels_dict.keys()))
        assert set(label_subset) == set(new_labels_dict.values())

        relevant_labels = new_labels_dict.keys()  # labels from old dictionary that will be used
        self.filtered_labels = [old_labels_dict[label] for label in relevant_labels]  # nums
        self.old_to_new_mapping = {
            old_labels_dict[label]: label_subset[new_labels_dict[label]]
            for label in relevant_labels}  # num -> num
        self.old_to_new_mapping[0.0] = 0  # background

    def __call__(self, data: Mapping[Hashable, np.ndarray]) -> Dict[Hashable, np.ndarray]:
        d = dict(data)
        for key in self.key_iterator(d):
            tensor = d[key]  # label tensor
            zeros = torch.zeros(tensor.shape)
            # bool_tensor[i,j] == True if tensor[i,j] in self.filtered_labels, False o/w
            bool_tensor = tensor == self.filtered_labels[0]
            for label in self.filtered_labels[1:]:
                bool_tensor = torch.logical_or(bool_tensor, tensor == label)
            filtered_tensor = torch.where(bool_tensor, tensor, zeros)
            new_tensor = filtered_tensor.apply_(lambda x: self.old_to_new_mapping[x])
            d[key] = new_tensor
        return d


def extract_segmentation_data(
    data_path: Path,
    image_name: str,
    label_name: str,
    val_size: float = 0.1
) -> Tuple[List[Dict[str, Path]], List[Dict[str, Path]]]:
    """Extracts segmentation data from $data_path -> train/val files.

    Images and labels must be in .nii.gz format.

    Parameters
    ----------
    data_path: Path
        Path that contains the image and label files used for training/validations.
        Directory should have the following structure:
        - case_1
            - image_name (.nii.gz)
            - label_name (.nii.gz)
            - ...
        - case_2
            - image_name (.nii.gz)
            - label_name (.nii.gz)
            - ...
        ...
    image_name: str
        Name of the .nii.gz file referencing the image (see data_path)
    label_name: str
        Name of the .nii.gz file referencing the label (see data_path)
    val_size: float
            Proportion of available data to be used for validation.

    Returns
    -------

    """

    train_images = []
    train_labels = []

    for case in sorted(os.listdir(data_path)):
        try:
            x = glob.glob(os.path.join(data_path, case, image_name))[0]
            y = glob.glob(os.path.join(data_path, case, label_name))[0]
            train_images.append(x), train_labels.append(y)
        except IndexError:  # couldn't find either image or label
            continue

    data_dicts = [
        {"image": image_name, "label": label_name}
        for image_name, label_name in zip(train_images, train_labels)
    ]

    num_val_files = math.floor(val_size * len(data_dicts))
    train_files, val_files = data_dicts[:-num_val_files], data_dicts[-num_val_files:]
    print("# of Training Files: ", len(train_files))
    print("# of Validation Files: ", len(val_files))

    return train_files, val_files


def train_val_transforms_segmentation(
    old_labels_dict: Dict[str, int],
    label_subset: Dict[str, int],
    new_labels_dict: Dict[str, str],
    level: int = 50,
    window: int = 400
) -> Tuple[Compose, Compose]:
    """Constructs predefind train/val transforms for segmentation task.

    Includes label encoding/filtering, which refers to selecting a subset of the original labels that came with the
    dataset and specifiying what number they map to.

    Parameters
    ----------
    old_labels_dict: Dict[str, int]
        Mapping from anatomical structures to numbers used in labels.
    label_subset: Dict[str, int]
        Subset of new labels that will be trained on mapped to their desired numerical encoding.
        Note that if keys do not match old_labels_dict, new_labels_dict must specify mapping from old labels
        to new labels.
    new_labels_dict: Dict[str, str]
        Mapping from keys of old_labels_dict to keys of label_subset. Defaults to old_labels_dict if None
        is passed. Example of when this is necessary is the visceral dataset:
        old_label_dict = {"left_kidney": 9, "right_kidney": 10}
        label_subset = {"kidney": 1}
        new_labels_dict = {"left_kidney": "kidney", "right_kidney": "kidney"}
    level: int
        Center of intensity scaling
    window: int
        Range of intensity scaling

    Returns
    -------
    train_transforms: Compose
        Composition of transforms for training set
    val_transforms: Compose
        Composition of transforms for validation set
    """
    train_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            AddChanneld(keys=["image", "label"]),
            Spacingd(keys=["image", "label"], pixdim=(
                1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            RandShiftIntensityd(keys=["image"], offsets=10, prob=0.5),
            ScaleIntensityRanged(
                keys=["image"], a_min=level - window / 2, a_max=level + window / 2,
                b_min=0.0, b_max=1.0, clip=True,
            ),
            CropForegroundd(keys=["image", "label"], source_key="image"),
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=(128, 128, 96),
                pos=1,
                neg=1,
                num_samples=4,
                image_key="image",
                image_threshold=0,
            ),
            RandAffined(
                keys=["image", "label"],
                mode=('bilinear', 'nearest'),
                prob=0.5,
                rotate_range=(np.pi / 12, np.pi / 12, np.pi / 12),  # rotates around dimension axis
                scale_range=(0.05, 0.05, 0.05),  # streches in dimension
                translate_range=(10, 10, 10),  # translates in dimension
                shear_range=(0.05, 0.05, 0.05)  # shears in dimension
            ),
            ToTensord(keys=["image", "label"]),
            LabelFilterd(
                keys=["label"],
                old_labels_dict=old_labels_dict,
                label_subset=label_subset,
                new_labels_dict=new_labels_dict
            )
        ]
    )
    val_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            AddChanneld(keys=["image", "label"]),
            Spacingd(keys=["image", "label"], pixdim=(
                1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            ScaleIntensityRanged(
                keys=["image"], a_min=level - window / 2, a_max=level + window / 2,
                b_min=0.0, b_max=1.0, clip=True,
            ),
            CropForegroundd(keys=["image", "label"], source_key="image"),
            ToTensord(keys=["image", "label"]),
            LabelFilterd(
                keys=["label"],
                old_labels_dict=old_labels_dict,
                label_subset=label_subset,
                new_labels_dict=new_labels_dict
            )
        ]
    )

    return train_transforms, val_transforms


def construct_dataset(
    files: List[Dict[str, Path]],
    transforms: Compose,
    batch_sz: int = 2,
    cache_dataset: bool = True,
    shuffle: bool = True,
) -> Tuple[Dataset, DataLoader]:
    """Creates a Dataset and Dataloader.

    Separately call this function for training and validation sets.

    Parameters
    ----------
    files: List[Dict[str, Path]]
        List of dictionaries of the form {"image":img_path, "label":label_path}
    transforms: Compose
        Composition of transforms to be applied to $files.
    batch_sz: int = 2
        # of samples per gradient update. Note that the RandCrop transforms generates four crops
        per batch item. Therefore, the 'true' batch size is 4*batch_sz
    cache_dataset: bool = True
        If True, CacheDataset. Else, Dataset. MONAI claims that CacheDataset -> 10x faster training.
    shuffle: bool = True
        Whether DataLoader shuffles files each iteration.

    Returns
    -------
    ds: Dataset
        MONAI Dataset (either Cache or normal)
    loader: DataLoader
        MONAI DataLoader
    """

    if cache_dataset:
        ds = CacheDataset(data=files, transform=transforms, cache_rate=1.0, num_workers=4)
    else:
        ds = Dataset(data=files, transform=transforms)

    loader = DataLoader(ds, batch_size=batch_sz, shuffle=shuffle, num_workers=4)
    return ds, loader


def full_segmentation_preprocess(
    data_path: Path,
    image_name: str,
    label_name: str,
    old_labels_dict: Dict[str, int],
    label_subset: Dict[str, int],
    new_labels_dict: Dict[str, str],
    level: int = 50,
    window: int = 400,
    batch_sz: int = 2,
    val_size: float = 0.1,
    cache_dataset: bool = True,
) -> Tuple[Dataset, DataLoader, Compose, Dataset, DataLoader, Compose]:
    """Executes entire preprocessing pipeline.

    Combines all steps of preprocessing into single function:
    1. Extract segmentation data from some data directory -> train/val files
    2. Construct train/val transforms, including label encoding/filtering.
    3. Create MONAI Datasets/Dataloaders for train/val files using train/val transforms.

    Useful for predefined datasets.

    Parameters
    ----------
    data_path: Path
        Path that contains the image and label files used for training/validations.
        Directory should have the following structure:
        - case_1
            - image_name (.nii.gz)
            - label_name (.nii.gz)
            - ...
        - case_2
            - image_name (.nii.gz)
            - label_name (.nii.gz)
            - ...
        ...
    image_name: str
        Name of the .nii.gz file referencing the image (see data_path)
    label_name: str
        Name of the .nii.gz file referencing the label (see data_path)
    old_labels_dict: Dict[str, int]
        Mapping from anatomical structures to numbers used in labels.
    label_subset: Dict[str, int]
        Subset of new labels that will be trained on mapped to their desired numerical encoding.
        Note that if keys do not match old_labels_dict, new_labels_dict must specify mapping from old labels
        to new labels.
    new_labels_dict: Dict[str, str]
        Mapping from keys of old_labels_dict to keys of label_subset. Defaults to old_labels_dict if None
        is passed. Example of when this is necessary is the visceral dataset:
        old_label_dict = {"left_kidney": 9, "right_kidney": 10}
        label_subset = {"kidney": 1}
        new_labels_dict = {"left_kidney": "kidney", "right_kidney": "kidney"}
    level: int
        Center of intensity scaling
    window: int
        Range of intensity scaling
    batch_sz: int
        # of samples per gradient update. Note that the RandCrop transforms generates four crops
        per batch item. Therefore, the 'true' batch size is 4*batch_sz
    val_size: float
        Proportion of available data to be used for validation.
    cache_dataset: bool
        If True, CacheDataset. Else, Dataset. MONAI claims that CacheDataset -> 10x faster training.

    Returns
    -------
    train_ds: Dataset
        MONAI Dataset (either Cache or normal) for training set.
    train_loader: DataLoader
        MONAI DataLoader for training set.
    train_transforms: Compose
        Composition of transforms used for training data.
    val_ds: Dataset
        MONAI Dataset (either Cache or normal) for validation set.
    val_loader: DataLoader
        MONAI DataLoader for validation set.
    val_transforms: Compose
        Composition of transforms used for validation data.

    """

    # crawl data folder to extract images and labels
    train_files, val_files = extract_segmentation_data(
        data_path=data_path,
        image_name=image_name,
        label_name=label_name,
        val_size=val_size
    )
    # compose transforms (including label filter)
    train_transforms, val_transforms = train_val_transforms_segmentation(
        old_labels_dict=old_labels_dict,
        label_subset=label_subset,
        new_labels_dict=new_labels_dict,
        level=level,
        window=window
    )
    # create monai Dataset and DataLoader for train/val datasets
    train_ds, train_loader = construct_dataset(
        files=train_files,
        transforms=train_transforms,
        cache_dataset=cache_dataset,
        batch_sz=batch_sz,
        shuffle=True
    ) if len(train_files) > 0 else (None, None)

    val_ds, val_loader = construct_dataset(
        files=val_files,
        transforms=val_transforms,
        cache_dataset=cache_dataset,
        batch_sz=1,
        shuffle=False
    ) if len(val_files) > 0 else (None, None)

    return train_ds, train_loader, train_transforms, val_ds, val_loader, val_transforms


def get_kits_dataset(
    label_subset: Dict[str, int] = {"kidney": 1, "ureter": 2},
    batch_sz: int = 2,
    val_size: float = 0.1,
    cache_dataset: bool = True,
) -> Tuple[Dataset, DataLoader, Compose, Dataset, DataLoader, Compose]:
    """
    Utility function to quickly retrieve kits dataset.
    """

    return full_segmentation_preprocess(
        data_path=locations.kits21_data_dir,
        image_name="imaging.nii.gz",
        label_name="aggregated_MAJ_seg.nii.gz",
        old_labels_dict=constants.kits21_labels,
        label_subset=label_subset,
        new_labels_dict=None,
        cache_dataset=cache_dataset,
        val_size=val_size,
        batch_sz=batch_sz
    )


def get_visceral_dataset(
    label_subset: Dict[str, int] = {"kidney": 1},
    batch_sz: int = 2,
    val_size: float = 1,
    cache_dataset: bool = True
) -> Tuple[Dataset, DataLoader, Compose, Dataset, DataLoader, Compose]:
    """
    Utility function to quickly retrieve visceral dataset.
    """

    return full_segmentation_preprocess(
        data_path=locations.visceral_data_dir,
        image_name="CTce.nii.gz",
        label_name="combined-label.nii.gz",
        old_labels_dict=constants.visceral_labels,
        label_subset=label_subset,
        new_labels_dict=constants.visceral_mapping,
        cache_dataset=cache_dataset,
        val_size=val_size,
        batch_sz=batch_sz
    )