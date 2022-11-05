import os
from typing import Callable, Optional, Tuple, Any
import torch 
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder




def filter_by_num_images(data_dir : str, min_examples: int) -> 'list[list[str, int]]':
    '''
    Used to filter dataset so that only classes with at least min_examples
    datapoints are used.
    takes as input the path to the root data directory and returns
    a list with the names and number of images for each subdirectory with >= min_examples.
    Expects a dataset with the following structure:
    Root_Dir
        -- Class 1
            --ex 1
            --ex 2
            .
            .
            .
            --ex n_1
        -- Class 2
            --ex 1
            .
            .
            .
            --ex n_2
        .
        .
        .
        --- Class k
            -- ex 1
            .
            .
            .
            --ex n_k
    '''
    data_dir = data_dir + '/' if data_dir[-1] != '/' else data_dir
    all_class_directories = next(os.walk(data_dir))[1]
    suitable_directories = []
    for dir in all_class_directories:
        num_files = len(next(os.walk(data_dir + dir))[2])
        if num_files >= min_examples:
            suitable_directories.append([dir, num_files])
    return suitable_directories




class GeoGuessr_Screenshots_Dataset(ImageFolder):
    def __init__(self, root: str, min_examples_per_class: int, *args, **kwargs):
        '''
        for *args and **kwargs options see http://pytorch.org/vision/main/generated/torchvision.datasets.ImageFolder.html
        main ones are root (mandatory)
        and transform
        '''
        self.min_examples_per_class = min_examples_per_class
        super().__init__(root, *args, **kwargs)


    def find_classes(self, directory: str):
        '''
        overrides find_classes to allow ignoring classes with few examples.
        Also calculates weights for each class at the same time.
        '''
        valid_subdirectories = filter_by_num_images(directory, self.min_examples_per_class)
        all_classes = []
        class_to_index = dict()
        class_index = 0
        for sub_dir in valid_subdirectories:
            all_classes.append(sub_dir[0])
            class_to_index[sub_dir[0]] = class_index
            class_index += 1
        self.num_classes = class_index
        self.index_to_class = dict()
        for key, val in class_to_index.items():
            self.index_to_class[val] = key
        return (all_classes, class_to_index)

    def get_num_classes(self) -> int:
        return self.num_classes

    def get_class_from_index(self, index) -> str:
        return self.index_to_class[index]

    def get_index_to_class_map(self) -> dict[int, str]:
        return self.index_to_class




def calculate_class_weights(dataset) -> torch.tensor:
    '''
    calculates weights for each class 
    according to method from https://scikit-learn.org/stable/modules/generated/sklearn.utils.class_weight.compute_class_weight.html
    '''
    loader = DataLoader(dataset, batch_size = 128, drop_last=False)
    samples_per_class = dict()
    for _, targets in loader:
        classes_in_sample = torch.unique(targets)
        for class_idx in classes_in_sample:
            occurrences = torch.sum(targets==class_idx)
            if samples_per_class.get(int(class_idx)) == None:
                samples_per_class[int(class_idx)] = occurrences
            else:
                samples_per_class[int(class_idx)] += occurrences
    n_classes = len(samples_per_class)
    n_samples = len(dataset)
    class_weights = torch.empty(n_classes)
    for i in samples_per_class.keys():
        class_weights[i] = n_samples / (n_classes *samples_per_class[i])
    return class_weights




def main():
    test = filter_by_num_images("./Data/", 100)
    for dir in test:
        print(dir)
    dataset = GeoGuessr_Screenshots_Dataset('./Data/', 1000)
    print(len(dataset), dataset.get_num_classes())




if __name__ == "__main__":
    main()