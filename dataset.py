from typing import Any, Callable, Optional, Tuple, Union

import os.path
import pickle
from pathlib import Path
from PIL import Image

import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms.functional as F
from torchvision.datasets.utils import check_integrity, download_and_extract_archive
from torchvision.datasets.vision import VisionDataset


class CINIC10(VisionDataset):
    """`CINIC10 <https://github.com/BayesWatch/cinic-10>`_ Dataset.

    Args:
        root (str or ``pathlib.Path``): Root directory of dataset where directory
            ``cinic-10-batches-py`` exists or will be saved to if download is set to True.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that takes in a PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.

    """

    base_folder = "cinic-10-batches-py"
    url = "https://huggingface.co/datasets/alexey-zhavoronkin/CINIC10/resolve/main/cinic-10-python.tar.gz?download=true"
    filename = "cinic-10-python.tar.gz"
    tgz_md5 = None
    train_list = [
        ["data_batch_1", None],
        ["data_batch_2", None],
        ["data_batch_3", None],
        ["data_batch_4", None],
        ["data_batch_5", None],
        ["data_batch_6", None],
        ["data_batch_7", None],
        ["data_batch_8", None],
        ["data_batch_9", None],
        ["data_batch_10", None],
        ["data_batch_11", None],
        ["data_batch_12", None],
        ["data_batch_13", None],
        ["data_batch_14", None],


    ]

    test_list = [
        ["test_batch_1", None],
        ["test_batch_2", None],
        ["test_batch_3", None],
        ["test_batch_4", None],
        ["test_batch_5", None],
        ["test_batch_6", None],
        ["test_batch_7", None],


    ]
    meta = {
        "filename": "batches.meta",
        "key": "label_names",
        "md5": None,
    }

    def __init__(
        self,
        root: Union[str, Path],
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:

        super().__init__(root, transform=transform, target_transform=target_transform)

        self.train = train  # training set or test set

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError("Dataset not found or corrupted. You can use download=True to download it")

        if self.train:
            downloaded_list = self.train_list
        else:
            downloaded_list = self.test_list

        self.data: Any = []
        self.targets = []

        # now load the picked numpy arrays
        for file_name, checksum in downloaded_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            with open(file_path, "rb") as f:
                entry = pickle.load(f, encoding="latin1")
                self.data.append(entry["data"])
                if "labels" in entry:
                    self.targets.extend(entry["labels"])
                else:
                    self.targets.extend(entry["fine_labels"])

        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC

        self._load_meta()

    def _load_meta(self) -> None:
        path = os.path.join(self.root, self.base_folder, self.meta["filename"])
        if not check_integrity(path, self.meta["md5"]):
            raise RuntimeError("Dataset metadata file not found or corrupted. You can use download=True to download it")
        with open(path, "rb") as infile:
            data = pickle.load(infile, encoding="latin1")
            self.classes = data[self.meta["key"]]
        self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self) -> int:
        return len(self.data)

    def _check_integrity(self) -> bool:
        for filename, md5 in self.train_list + self.test_list:
            fpath = os.path.join(self.root, self.base_folder, filename)
            if not check_integrity(fpath, md5):
                return False
        return True

    def download(self) -> None:
        if self._check_integrity():
            print("Files already downloaded and verified")
            return
        download_and_extract_archive(self.url, self.root, filename=self.filename, md5=self.tgz_md5)

    def extra_repr(self) -> str:
        split = "Train" if self.train is True else "Test"
        return f"Split: {split}"



class NpyDataset(Dataset):
    """Loads pre-computed iteration of dataset
      collected using `inference_utils.prepare_testset`
    """

    def __init__(
        self,
        root: str,
        idx: int
    ):
        super().__init__()

        self.data = np.load(os.path.join(root, f'data_batch_{idx}.npy'))
        self.targets = np.load(os.path.join(root, 'task_labels.npy'))[:, 0].tolist()
        
    def __getitem__(self, index) -> Any:
        img, target = self.data[index], self.targets[index]
        img = F.to_tensor(img)

        return img, target

    def __len__(self) -> int:
        return len(self.data)
