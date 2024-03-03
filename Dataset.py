import os
import cv2
from torch.utils.data import Dataset as BaseDataset
import random


class Dataset(BaseDataset):
    def __init__(
            self,
            images_dir,
            folder_list,
            max_files=10,
            augmentation=None,
            preprocessing=None,
    ):
        self.dir = images_dir
        self.augmentation = augmentation
        self.preprocessing = preprocessing

        triplets = []
        folders = list(folder_list.keys())

        for folder in folders:
            path = os.path.join(images_dir, folder)
            files = list(os.listdir(path))[:max_files]
            num_files = len(files)

            for i in range(num_files - 1):
                for j in range(i + 1, num_files):
                    anchor = (folder, f"{i}.jpg")
                    positive = (folder, f"{j}.jpg")

                    neg_folder = folder
                    while neg_folder == folder:
                        neg_folder = random.choice(folders)
                    neg_file = random.randint(0, folder_list[neg_folder] - 1)
                    negative = (neg_folder, f"{neg_file}.jpg")

                    triplets.append((anchor, positive, negative))

        random.shuffle(triplets)
        self.triplet_list = triplets

    def __getitem__(self, i):

        # read data
        a, p, n = self.triplet_list[i]

        path = os.path.join(self.dir, a[0], a[1])
        anchor = cv2.imread(path)
        anchor = cv2.cvtColor(anchor, cv2.COLOR_BGR2RGB)

        path = os.path.join(self.dir, p[0], p[1])
        positive = cv2.imread(path)
        positive = cv2.cvtColor(positive, cv2.COLOR_BGR2RGB)

        path = os.path.join(self.dir, n[0], n[1])
        negative = cv2.imread(path)
        negative = cv2.cvtColor(negative, cv2.COLOR_BGR2RGB)

        if self.preprocessing:
            anchor = self.preprocessing(anchor)
            positive = self.preprocessing(positive)
            negative = self.preprocessing(negative)

        return anchor, positive, negative

    def __len__(self):
        return len(self.triplet_list)
