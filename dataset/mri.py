import torch
import numpy as np
import os
import random 
from torch.utils.data import Dataset
from natsort import natsorted
# from torchvision.datasets import ImageFolder
from torchvision.datasets.folder import make_dataset

import nibabel as nib

## read MRI, normalize and padded to MRI
import numpy as np
import nibabel as nib
import torch
import random

def read_MRI(path, resolution, crop=(128,128,128), permute_axes=True):
    # Load MRI data
    img = nib.load(path)
    data = img.get_fdata()

    # Normalize the MRI data (clipping based on percentiles)
    max_value = np.percentile(data, 95)
    min_value = np.percentile(data, 5)
    data = np.where(data <= max_value, data, max_value)
    data = np.where(data <= min_value, 0., data)

    # Pad the data to the specified resolution
    padded_img = np.zeros(resolution)

    # Check for a bad case (image shape is 256 in one dimension)
    if data.shape[1] == 256:
        print(path)
    # Place the MRI data into the padded image
    padded_img[3:3+138, 8:8+176, 3:3+138] = data

    # Normalize to range [-1, 1]
    padded_img = (padded_img / (max_value + 0.00000001)) * 2 - 1
    
    # Perform random permutation of axes if permute_axes is True
    if permute_axes:
        # Randomly shuffle the axes (0, 1, 2)
        permuted_axes = random.sample([0, 1, 2], 3)  # Get a random permutation of (0, 1, 2)
        padded_img = np.transpose(padded_img, permuted_axes)  # Apply the permutation to the image
      
    # Perform random crop if crop is not None
    if crop is not None:
        crop_x, crop_y, crop_z = crop
        if padded_img.shape[0] > crop_x:
            x_start = random.randint(0, padded_img.shape[0] - crop_x)
        else:
            x_start = 0
        #y dim
        if padded_img.shape[1] > crop_y:
            y_start = random.randint(0, padded_img.shape[1] - crop_y)
        else:
            y_start = 0
        #z dim
        if padded_img.shape[2] > crop_z:
            z_start = random.randint(0, padded_img.shape[2] - crop_z)
        else:
            z_start = 0

        # Crop the subvolume from the padded image
        cropped_img = padded_img[x_start:x_start+crop_x, y_start:y_start+crop_y, z_start:z_start+crop_z]

        # Convert the numpy array to a torch tensor and add batch dimension
        data_tensor = torch.from_numpy(cropped_img[None, :, :, :]).float()
    else:
        # If no crop, return the entire padded image
        data_tensor = torch.from_numpy(padded_img[None, :, :, :]).float()

    return data_tensor


class MRIFolderDataset(Dataset):
    def __init__(self, root: str, crop, rotate, train = True, resolution = (144, 192, 144), seed = 42, ret_class_idx = False, **super_kwargs,):

        video_root = os.path.join(os.path.join(root))
        name = video_root.split('/')[-1]
        print(f'There are {len(name)} names in the file')
        self.name = name
        self.train = train
        self.resolution = resolution
        self.crop = crop
        self.rotate= rotate

        self.classes = list(
            natsorted(p for p in os.listdir(video_root) if os.path.isdir(os.path.join(video_root, p)))
            )
        
        self.class_to_idx = {self.classes[i]: i for i in range(len(self.classes))}

        self.samples = make_dataset(video_root, self.class_to_idx, ('nii.gz',), is_valid_file=None)
        video_list = [x[0] for x in self.samples]

        self.video_list = video_list
        self.size = len(self.video_list)
        random.seed(seed)
        self.shuffle_indices = [i for i in range(self.size)]
        random.shuffle(self.shuffle_indices)
        self.ret_class_idx = ret_class_idx


    def __len__(self):
        return self.size

    def _preprocess(self, video: torch.Tensor) -> torch.Tensor:
        video = read_MRI(video, self.resolution, self.crop, self.rotate)
        return video

    def __getitem__(self, idx: int) -> torch.Tensor:
        ## MRI
        idx = self.shuffle_indices[idx]
        pixel_values = self._preprocess(self.video_list[idx])

        ##Class 
        cls_name = self.video_list[idx].split('/')[-2]
        class_idx = self.class_to_idx[cls_name]
        return pixel_values, class_idx #torch.from_numpy(class_idx) 


def build_mri(args, transform):
    return MRIFolderDataset(args.data_path, args.crop, args.rotate)

    # return ImageFolder(args.data_path, transform=transform)

def build_imagenet_code(args):
    feature_dir = f"{args.code_path}/imagenet{args.image_size}_codes"
    label_dir = f"{args.code_path}/imagenet{args.image_size}_labels"
    assert os.path.exists(feature_dir) and os.path.exists(label_dir), \
        f"please first run: bash scripts/autoregressive/extract_codes_c2i.sh ..."
    return CustomDataset(feature_dir, label_dir)
