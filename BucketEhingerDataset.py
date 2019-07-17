from EhingerDataset import EhingerDataset
import random
import numpy as np
import torch

class BucketEhingerDataset(EhingerDataset):

    def __init__(self, csv_file, root_dir, batch_size, transform=None):

        super().__init__(csv_file, root_dir, transform=transform)

        self.batch_size = batch_size

        #create buckets
        self.buckets = {}
        for key, value in self.dictionary.items():
            len_seq = len(value)
            self.buckets.setdefault(len_seq, []).append(key)

        self.bucket_keys = list(self.buckets.keys())
        self.bucket_dict = dict(zip(range(len(self.bucket_keys)), self.bucket_keys))

        print(len(self.bucket_dict), "buckets created")

    def get_bucket_weights(self):
        weights=[]
        for key, value in self.bucket_dict.items():
            weights.append(len(self.buckets[value]))
        return np.array(weights)

    def __getitem__(self, idx):

        #get random keys from bucket of number idx
        keys = self.buckets[self.bucket_dict[idx]]
        k = min(self.batch_size, len(keys))
        random_keys = random.sample(population=keys, k=k)
        #from keys, get features and sequence
        features  = np.array([self.features[k[0]] for k in random_keys])
        landmarks = np.array([self.dictionary[k] for k in random_keys])
        # yield batch
        return {'features':   torch.from_numpy(features).type(torch.float32),
                'landmarks':  torch.from_numpy(landmarks).type(torch.float32),
                'batch_size': k
                }

