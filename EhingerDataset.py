from __future__ import print_function, division
import os
import torch
import csv
from torch.utils.data import Dataset
import utils
import pickle
import numpy as np

from torchtext.data import BucketIterator

def process_example(fix):

    fix_x = float(fix["fix_x"])
    fix_y = float(fix["fix_y"])
    fix_dur = float(fix["fix_duration"])
    return np.array([fix_x,fix_y,fix_dur])

class EhingerDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_file, root_dir, batch_size=32, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a image.
        """
        self.transform = transform
        self.root_dir = root_dir
        self.csv_file = os.path.join(self.root_dir,csv_file)
        self.landmarks_frame = csv.DictReader(open(self.csv_file))
        self.dictionary = {}
        for line in self.landmarks_frame:
            key = (line["img"],line["subject"])
            self.dictionary.setdefault(key, []).append(process_example(line))

        #create a list of keys
        self.keys = list(self.dictionary.keys())

        #computing features for all images
        self.feature_filename = os.path.join(self.root_dir, csv_file+".pkl")
        if not os.path.exists(self.feature_filename):
            utils.extract_features(root_dir,
                                  keys=self.keys,
                                  layer="layer3",
                                  transform=self.transform,
                                  output_file=self.feature_filename)

        self.features = pickle.load(open(self.feature_filename, 'rb'))

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        #get fixations
        landmarks = self.dictionary[self.keys[idx]]
        landmarks = np.array(landmarks)
        #get features
        img, _ = self.keys[idx]
        features = self.features[img]

        return {'features': torch.from_numpy(features).type(torch.float32),
                'landmarks':  torch.from_numpy(landmarks).type(torch.float32),
                }