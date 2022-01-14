""" Quick n Simple Image Folder, Tarfile based DataSet

Hacked together by / Copyright 2020 Ross Wightman
"""
import torch.utils.data as data
import os
import logging
import math
import collections
import tqdm
import cv2
import torch
import pandas as pd

from glob import glob
from PIL import Image

from .parsers import create_parser
from torchvision import datasets as torch_datasets
from torchvision import transforms
from torchvision.utils import save_image
from torch.utils.data import Dataset

from torchvision.utils import make_grid

# create logger
_logger = logging.getLogger(__name__)

# create console handler and set level to debug
ch = logging.StreamHandler()

# create formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# add formatter to ch
ch.setFormatter(formatter)

# add ch to logger
_logger.addHandler(ch)
_ERROR_RETRY = 50


class ImageDataset(data.Dataset):

    def __init__(
            self,
            root,
            parser=None,
            class_map=None,
            load_bytes=False,
            transform=None,
            target_transform=None,
    ):
        if parser is None or isinstance(parser, str):
            parser = create_parser(parser or '', root=root, class_map=class_map)
        self.parser = parser
        self.load_bytes = load_bytes
        self.transform = transform
        self.target_transform = target_transform
        self._consecutive_errors = 0

    def __getitem__(self, index):
        img, target = self.parser[index]
        try:
            img = img.read() if self.load_bytes else Image.open(img).convert('RGB')
        except Exception as e:
            _logger.warning(f'Skipped sample (index {index}, file {self.parser.filename(index)}). {str(e)}')
            self._consecutive_errors += 1
            if self._consecutive_errors < _ERROR_RETRY:
                return self.__getitem__((index + 1) % len(self.parser))
            else:
                raise e
        self._consecutive_errors = 0
        if self.transform is not None:
            img = self.transform(img)
        if target is None:
            target = -1
        elif self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

    def __len__(self):
        return len(self.parser)

    def filename(self, index, basename=False, absolute=False):
        return self.parser.filename(index, basename, absolute)

    def filenames(self, basename=False, absolute=False):
        return self.parser.filenames(basename, absolute)


class IterableImageDataset(data.IterableDataset):

    def __init__(
            self,
            root,
            parser=None,
            split='train',
            is_training=False,
            batch_size=None,
            repeats=0,
            download=False,
            transform=None,
            target_transform=None,
    ):
        assert parser is not None
        if isinstance(parser, str):
            self.parser = create_parser(
                parser, root=root, split=split, is_training=is_training,
                batch_size=batch_size, repeats=repeats, download=download)
        else:
            self.parser = parser
        self.transform = transform
        self.target_transform = target_transform
        self._consecutive_errors = 0

    def __iter__(self):
        for img, target in self.parser:
            if self.transform is not None:
                img = self.transform(img)
            if self.target_transform is not None:
                target = self.target_transform(target)
            yield img, target

    def __len__(self):
        if hasattr(self.parser, '__len__'):
            return len(self.parser)
        else:
            return 0

    def filename(self, index, basename=False, absolute=False):
        assert False, 'Filename lookup by index not supported, use filenames().'

    def filenames(self, basename=False, absolute=False):
        return self.parser.filenames(basename, absolute)


class AugMixDataset(torch.utils.data.Dataset):
    """Dataset wrapper to perform AugMix or other clean/augmentation mixes"""

    def __init__(self, dataset, num_splits=2):
        self.augmentation = None
        self.normalize = None
        self.dataset = dataset
        if self.dataset.transform is not None:
            self._set_transforms(self.dataset.transform)
        self.num_splits = num_splits

    def _set_transforms(self, x):
        assert isinstance(x, (list, tuple)) and len(x) == 3, 'Expecting a tuple/list of 3 transforms'
        self.dataset.transform = x[0]
        self.augmentation = x[1]
        self.normalize = x[2]

    @property
    def transform(self):
        return self.dataset.transform

    @transform.setter
    def transform(self, x):
        self._set_transforms(x)

    def _normalize(self, x):
        return x if self.normalize is None else self.normalize(x)

    def __getitem__(self, i):
        x, y = self.dataset[i]  # all splits share the same dataset base transform
        x_list = [self._normalize(x)]  # first split only normalizes (this is the 'clean' split)
        # run the full augmentation on the remaining splits
        for _ in range(self.num_splits - 1):
            x_list.append(self._normalize(self.augmentation(x)))
        return tuple(x_list), y

    def __len__(self):
        return len(self.dataset)


class EventMNISTDataset(data.Dataset):
    def __init__(self, root, split='train', transform=None, target_transform=None, number_of_frames=9,
                 img_prefix="img_", sample=False, sample_times=9):
        self.dataset_root = root
        self.train_dir = os.path.join(self.dataset_root, "train")
        self.val_dir = os.path.join(self.dataset_root, "val")
        self.test_dir = os.path.join(self.dataset_root, "test")
        self.sample = sample
        self.sample_times = sample_times
        self.dir_dict = {
            "train": self.train_dir,
            "val": self.val_dir,
            "valid": self.val_dir,
            "validation": self.val_dir,
            "test": self.test_dir,

        }

        self.transform = transform
        self.target_transform = target_transform
        accepted_frames = [4, 9, 16, 25, 36, 49, 64]
        if number_of_frames not in accepted_frames:
            raise Exception("The number of frames should be a value between {}")
        self.number_of_frames = number_of_frames
        self.frames_per_row = int(math.sqrt(number_of_frames))
        self.frames_per_col = self.frames_per_row

        self.img_dir = self.dir_dict[split]
        self.labels_file = os.path.join(self.img_dir, "labels.csv")
        self.csv_data = {'fname': [], 'label': []}
        self.raw_data_loader = None
        self.generated_img_id = 0
        if not os.path.exists(self.labels_file):
            if not os.path.exists(self.img_dir):
                os.makedirs(self.img_dir)
            if not os.path.exists(os.path.join(self.dataset_root, "raw")):
                self.__download_raw_data(split)
            else:
                self.__load_raw_data(split)
            self.__create_summation_training_data(img_prefix)
            #self.__create_event_training_data(split, img_prefix)
            #self.__create_no_event_training_data(split, img_prefix)
            self.__save_annotations()
        self.img_labels = pd.read_csv(self.labels_file)

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = Image.open(img_path).convert('RGB')
        image = image.convert('L')  # Reading grayscale
        # image = read_image(img_path, mode=ImageReadMode.GRAY)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

    def __download_raw_data(self, split):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        if split in ["train"]:
            dataset1 = torch_datasets.MNIST(f"{os.path.join(self.dataset_root, 'raw')}", train=True, download=True,
                                            transform=transform)
        elif split in ["val", "valid", "validation"]:
            dataset1 = torch_datasets.MNIST(f"{os.path.join(self.dataset_root, 'raw')}", train=False, download=True,
                                            transform=transform)
        # If true the data loaded for each batch will be sampled multiple times
        if self.sample:
            train_kwargs = {'batch_size': self.sample_times*self.number_of_frames}
        else:
            train_kwargs = {'batch_size': self.number_of_frames}

        self.raw_data_loader = torch.utils.data.DataLoader(dataset1, **train_kwargs)

    def __load_raw_data(self, split):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        if split in ["train"]:
            dataset1 = torch_datasets.MNIST(f"{os.path.join(self.dataset_root, 'raw')}", train=True, download=False,
                                            transform=transform)
        elif split in ["val", "valid", "validation"]:
            dataset1 = torch_datasets.MNIST(f"{os.path.join(self.dataset_root, 'raw')}", train=False, download=False,
                                            transform=transform)

        if self.sample:
            train_kwargs = {'batch_size': self.sample_times*self.number_of_frames}
        else:
            train_kwargs = {'batch_size': self.number_of_frames}

        self.raw_data_loader = torch.utils.data.DataLoader(dataset1, **train_kwargs)

    def __create_summation_training_data(self, img_prefix="img_"):
        """
        A function to create an experimental MNist dataset. The dataset is used to test/verify if transformer can
        compute additions of 9 mnist numbers patched together in an image. That will assume that the mnist ciphers
        will be distributed differently in the image space. Will the transformers understand that?

        """
        no_cuda = True
        use_cuda = not no_cuda and torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")

        for batch_idx, (data, target) in enumerate(tqdm.tqdm(self.raw_data_loader, desc="Creating the event dataset")):
            data, target = data.to(device), target.to(device)
            if data.shape[0] != self.number_of_frames:
                continue

            data = data.reshape(self.frames_per_row, data.shape[1], data.shape[2] * self.frames_per_row, data.shape[3])
            data = data.transpose(2, 3)
            data = data.reshape(1, data.shape[1], data.shape[2] * self.frames_per_row, data.shape[3])
            data = data.transpose(2, 3)
            image_data = data[0]
            img_id = f'{self.generated_img_id}'.zfill(9)
            img_name = f'{img_prefix}{img_id}.png'
            img_path = os.path.join(self.img_dir, img_name)
            save_image(image_data, f'{img_path}')
            self.csv_data["fname"].append(img_name)
            self.csv_data["label"].append(int(target.sum().cpu().numpy()))
            self.generated_img_id += 1

    def __create_event_training_data(self, split, img_prefix="img_"):

        no_cuda = True
        use_cuda = not no_cuda and torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")

        event = 1
        for batch_idx, (data, target) in enumerate(tqdm.tqdm(self.raw_data_loader, desc="Creating the event dataset")):
            data, target = data.to(device), target.to(device)
            if data.shape[0] != self.number_of_frames:
                continue

            data = data.reshape(self.frames_per_row, data.shape[1], data.shape[2] * self.frames_per_row, data.shape[3])
            data = data.transpose(2, 3)
            data = data.reshape(1, data.shape[1], data.shape[2] * self.frames_per_row, data.shape[3])
            data = data.transpose(2, 3)
            image_data = data[0]
            img_id = f'{self.generated_img_id}'.zfill(9)
            img_name = f'{img_prefix}{img_id}.png'
            img_path = os.path.join(self.img_dir, img_name)
            save_image(image_data, f'{img_path}')
            self.csv_data["fname"].append(img_name)
            self.csv_data["label"].append(event)
            self.generated_img_id += 1

    def __create_no_event_training_data(self, split, img_prefix="img_"):

        no_cuda = True
        use_cuda = not no_cuda and torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")
        event = 0

        label_to_tensors = {}
        for batch_idx, (data, target) in enumerate(
                tqdm.tqdm(self.raw_data_loader, desc="Creating the no event dataset")):
            data, target = data.to(device), target.to(device)
            for sample_idx in range(data.shape[0]):
                if target[sample_idx].item() not in label_to_tensors:
                    label_to_tensors[target[sample_idx].item()] = [data[sample_idx]]
                else:
                    label_to_tensors[target[sample_idx].item()].append(data[sample_idx])

        for label, value in label_to_tensors.items():
            for batch in range(0, len(value), self.number_of_frames):
                data = value[batch:batch + self.number_of_frames]
                if len(data) != 9:
                    continue
                data = torch.stack(data)
                data = data.reshape(self.frames_per_row, data.shape[1], data.shape[2] * self.frames_per_row,
                                    data.shape[3])
                data = data.transpose(2, 3)
                data = data.reshape(1, data.shape[1], data.shape[2] * self.frames_per_row, data.shape[3])
                data = data.transpose(2, 3)
                image_data = data[0]
                img_id = f'{self.generated_img_id}'.zfill(9)
                img_name = f'{img_prefix}{img_id}.png'
                img_path = os.path.join(self.img_dir, img_name)
                save_image(image_data, f'{img_path}')

                self.csv_data["fname"].append(img_name)
                self.csv_data["label"].append(event)
                self.generated_img_id += 1

    def __save_annotations(self):
        pd.DataFrame(self.csv_data).to_csv(self.labels_file, index=False)


class VideoEventDataset(Dataset):
    def __init__(self, root, input_data,
                 split='train',
                 transform=None,
                 target_transform=None,
                 number_of_frames=9,
                 crop=None,
                 update=False,  # If the dataset creation should be re-run
                 img_prefix="img_",
                 video_ext="mp4"):
        """
        The class is used to create the Dataset for detecting events in a sequence of images that are taken in
        with a sampling rate that can variate from 1 frame per 15 seconds to 1 frame per 5 minutes

        This class is being created to be able to model relationships between patches of the same image.
        The inspiration is based on the transformer architecture.
        Since transformers try to create attention the image on a global scale, by creating queries about the different
        patches. The idea is that by patching different images in a new one, the temporal dimension will be converted
        into a spatial dimension. Thus reducing the need to perform 4D convolutional operation.
        Another inspiration is to understand if transformers/attention will be able to learn features about the
        """
        _logger.info("Creating the VideoEventDataset for {split}")
        self.dataset_root = root
        # Video should be stored in the following directory structure
        # root/event/video.ext, where event is an integer mapping the labeled event for that video
        self.video_ext = video_ext

        # Fix automatic validation and training dataset creation
        self.video_list = glob(f"{input_data}/{split}/*/*.{self.video_ext}")

        _logger.info(f"The following {split} videos were found in the input directory {self.video_list }")

        self.train_dir = os.path.join(self.dataset_root, "train")
        self.val_dir = os.path.join(self.dataset_root, "val")
        self.test_dir = os.path.join(self.dataset_root, "test")
        self.dir_dict = {
            "train": self.train_dir,
            "val": self.val_dir,
            "valid": self.val_dir,
            "validation": self.val_dir,
            "test": self.test_dir,

        }
        self.update = update
        self.transform = transform
        self.crop = crop  # (y, h, x, w)

        self.target_transform = target_transform
        accepted_frames = [4, 9, 16, 25, 36, 49, 64]
        if number_of_frames not in accepted_frames:
            raise Exception("The number of frames should be a value between {}")
        self.number_of_frames = number_of_frames
        self.frames_per_row = int(math.sqrt(number_of_frames))
        self.frames_per_col = self.frames_per_row

        self.img_dir = self.dir_dict[split]
        self.labels_file = os.path.join(self.img_dir, "labels.csv")
        self.csv_data = {'fname': [], 'label': []}
        self.raw_data_loader = None
        self.generated_img_id = 0

        if not os.path.exists(self.labels_file) or self.update:
            _logger.info(f"The labels file does not exist. Creating dataset from scratch for the following videos {self.video_list}")
            if not os.path.exists(self.img_dir):
                os.makedirs(self.img_dir)
            self.__create_event_data_split(img_prefix)
            self.__save_annotations()
            self.img_labels = pd.read_csv(self.labels_file)

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = Image.open(img_path).convert('RGB')
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

    def __create_event_data_split(self, img_prefix):
        """
        @param img_prefix: the prefix to add to the ImageId when saving.
        """
        transform = transforms.ToTensor()

        for idx, video in enumerate(tqdm.tqdm(self.video_list)):
            event = video.split("/")[1]
            vidcap = cv2.VideoCapture(video)
            video_length = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
            success, image = vidcap.read()
            if success is False:
                print("Video file could not be read")
                raise("VideoFormatError")
            image_buffer = collections.deque(maxlen=self.number_of_frames)
            with tqdm.tqdm(total=video_length, desc=f"Processing video file {video}") as pbar:
                while success:
                    success, image = vidcap.read()
                    if success:
                        if self.crop is not None:
                            image = image[self.crop[0]:self.crop[0] + self.crop[1],
                                self.crop[2]:self.crop[2] + self.crop[3]]
                        # Resize the shape of the original image to have it fit inside the frame
                        w, h = int(image.shape[1] / self.frames_per_row), int(image.shape[0] / self.frames_per_col)
                        image = cv2.resize(image, (w, h))

                        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                        tensor = transform(image)
                        image_buffer.append(tensor)
                        pbar.update(1)
                        if len(image_buffer) != image_buffer.maxlen:
                            continue

                        image_data = make_grid(list(image_buffer), nrow=self.frames_per_row)
                        img_id = f'{self.generated_img_id}'.zfill(9)
                        img_name = f'{img_prefix}{img_id}.png'
                        img_path = os.path.join(self.img_dir, img_name)
                        save_image(image_data, f'{img_path}')
                        self.csv_data["fname"].append(img_name)
                        self.csv_data["label"].append(event)
                        self.generated_img_id += 1
                    else:
                        break

    def __save_annotations(self):
        pd.DataFrame(self.csv_data).to_csv(self.labels_file, index=False)
