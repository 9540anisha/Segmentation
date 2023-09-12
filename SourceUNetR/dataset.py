from random import random
import torch
from torch.utils.data import Dataset
import numpy as np
import nibabel as nib

from utils import (generate_ex_list, gen_mask, correct_dims, resize_img, center_crop, find_and_crop_lesions, random_crop)


class MRIDataset(Dataset):
    def __init__(self,
                 directory,
                 size,
                 sampling_mode="random",
                 deterministic=False):
        assert sampling_mode in ['resize', 'center', 'center_val', 'random']
        self.directory = directory
        self.inputs, self.labels = generate_ex_list(self.directory)
        #print(self.labels)
        self.size = size
        self.sampling_mode = sampling_mode
        self.deterministic = deterministic
        self.current_item_path = None  # use this to find the sample's path for debugging and visual inspection

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        self.current_item_path = self.inputs[idx]
        input_img = correct_dims(nib.load(self.inputs[idx]).get_fdata())
        label_img = gen_mask(self.labels[idx])
        print(input_img.shape)
    

        # Resize to input image and label to size (self.size x self.size x self.size)
        if self.sampling_mode == "resize":
            ex, label = resize_img(input_img, label_img, self.size)

        # Constant center-crop sample of size (self.size x self.size x self.size)
        elif self.sampling_mode == 'center':
            ex, label = center_crop(input_img, label_img, self.size)

        # Find centers of lesion masks and crop image to include them
        # to measure consistent validation performance with small crops
        elif self.sampling_mode == "center_val":
            ex, label = find_and_crop_lesions(input_img, label_img, self.size,
                                              self.deterministic)

        # Randomly crop sample of size (self.size x self.size x self.size)
        elif self.sampling_mode == "random":
            ex, label = random_crop(input_img, label_img, self.size)

        else:
            print("Invalid sampling mode.")
            exit()

        ex = np.divide(ex, 255.0)
        print(type(ex))
        print(len(ex))
        print(len(ex[0]))
        print(len(ex[0][0]))
        # ex2 = [ex.copy() for _ in range(4)]
        # label2 = [label.copy() for _ in range(4)]
        # ex = np.stack(ex2, axis=0)
        # label = np.stack(label2, axis=0)
        # print(ex.shape)
        # print(label.shape)
        
        label = np.array([(label > 0).astype(int)]).squeeze()
        # (experimental) APPLY RANDOM FLIPPING ALONG EACH AXIS
        if not self.deterministic:
            for i in range(3):
                if random() > 0.5:
                    ex = np.flip(ex, i)
                    label = np.flip(label, i)
        inputs = torch.from_numpy(ex.copy()).type(
            torch.FloatTensor).unsqueeze(0)
        print("I am here")
        print(len(inputs))
        print(type(inputs))
        labels = torch.from_numpy(label.copy()).type(
            torch.FloatTensor).unsqueeze(0)
        # print(inputs)
        #new = np.stack((inputs,)*4, axis=0)
        #mask = np.expand_dims(labels, axis=4)
        # print(new)
        #inputs = new
        #labels = mask
        return inputs, labels

    def _load_full_label(self, label_paths):
        """
        Loading full-size label for evaluation.
        """
        label_full = gen_mask(label_paths)
        label_full = np.array([(label_full > 0).astype(int)]).squeeze()
        return torch.Tensor(label_full)

    def _project_full_label(self, input_path, preds):
        """
        Projecting a predicted subvolume to full-size volume for evaluation.
        Only supposed to work with center sampling_mode.
        """
        size = nib.load(input_path).get_fdata().shape
        preds_full = np.zeros(size)
        coords = [0]*3
        for i in range(3):
            coords[i] = int((size[i]-self.size[i])//2)
        x, y, z = coords
        preds_full[x:x+self.size[0], y:y+self.size[1], z:z+self.size[2]] = preds
        return torch.Tensor(preds_full)
