import os
import random
import torchvision.transforms as T


def split_dataset(directory, split=0.9):
    folders = os.listdir(os.path.join(os.getcwd(), directory))
    num_train = int(len(folders) * split)

    random.shuffle(folders)

    train_list, test_list = {}, {}

    # Creating Train-list
    for folder in folders[:num_train]:
        num_files = len(os.listdir(os.path.join(directory, folder)))
        train_list[folder] = num_files

    # Creating Test-list
    for folder in folders[num_train:]:
        num_files = len(os.listdir(os.path.join(directory, folder)))
        test_list[folder] = num_files

    return train_list, test_list


def get_preprocessing(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    transform = T.Compose([T.ToTensor(), T.Normalize(mean=mean, std=std)])
    return transform


def calc_euclidean(x1, x2):
    return (x1 - x2).pow(2).sum(1)
