import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from typing import Optional
import matplotlib.pyplot as plt



def accuracy_per_class(model, dataset, batch_size: int = 128, device: 'torch.cuda.device' = 'cpu') -> tuple[float, dict[int, float]]:
    '''
    calculates overall accuracy of model on dataset as well as the accuracy on each class.
    Return value is (overall_accuracy, dict(class_index : class_accuracy)
    '''
    was_training = model.training
    model.eval()
    loader = DataLoader(dataset, batch_size = batch_size, drop_last=False)
    accuracy_per_class = dict()
    occurrences_per_class = dict()
    b_idx = 1
    total_accuracy = 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        output = model(inputs)
        preds = torch.argmax(output, dim=-1)
        correct = torch.where(preds == targets, 1, 0)
        total_accuracy += torch.sum(correct)
        for class_idx in torch.unique(targets):
            class_idx = int(class_idx)
            class_correct = torch.sum(torch.where(targets == class_idx, correct, 0))
            if occurrences_per_class.get(class_idx) == None:
                occurrences_per_class[class_idx] = int(torch.sum(targets == class_idx))
            else:
                occurrences_per_class[class_idx] += int(torch.sum(targets == class_idx))
            if accuracy_per_class.get(class_idx) == None:
                accuracy_per_class[class_idx] = class_correct
            else:
                accuracy_per_class[class_idx] += class_correct
        b_idx += 1
    for class_idx in accuracy_per_class.keys():
        accuracy_per_class[class_idx] = accuracy_per_class[class_idx] / occurrences_per_class[class_idx]
    total_accuracy = total_accuracy / len(dataset)
    if was_training:
        model.train()
    return total_accuracy, accuracy_per_class


def compute_confusion_matrix(model: nn.Module, dataset: Dataset, n_classes: int, batch_size: int = 128, device: torch.cuda.device = 'cpu') -> torch.tensor:
    '''
    computes the confusion matrix for model on dataset. 
    See https://en.wikipedia.org/wiki/Confusion_matrix for definition. 
    '''
    was_training = model.training
    model.eval()
    with torch.no_grad():
        model.to(device)
        matrix = torch.zeros((n_classes, n_classes), dtype=torch.int, device=device)
        loader = DataLoader(dataset, batch_size=batch_size)
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            preds = model(inputs)
            preds = torch.argmax(preds, dim=-1)
            for i in range(n_classes):
                for j in range(n_classes):
                    matrix[i, j] += torch.sum(torch.where(torch.logical_and(targets==i, preds==j), 1, 0))
    if was_training:
        model.train()
    return matrix


def display_classwise_accuracy(model, dataset, 
                                dataset_label: str, 
                                batch_size: int = 128, 
                                device: torch.cuda.device = 'cpu', 
                                index_to_class_map: Optional[dict[int, str]] = None, 
                                epoch: Optional[int] = None) -> None:
    '''
    wrapper function for computing classwise and overall accuracy then pretty printing.
    '''
    with torch.no_grad():
        if epoch == None:
            print(f"\nStatistics on {dataset_label} Data:")
        else:
            print(f"\nStatistics on {dataset_label} Data after epoch {epoch}:")
        total_acc, class_accuracies = accuracy_per_class(model, dataset, batch_size, device)
        print(f"Overall accuracy = {100*total_acc:.3f}%")
        for class_idx in class_accuracies.keys():
            class_label = class_idx if index_to_class_map == None else index_to_class_map[class_idx]
            print(f"Class: {class_label} Accuracy: {100*class_accuracies[class_idx]:.3f}%")


def display_confusion_matrix(confusion_matrix, title, index_to_label) -> None:
    '''
    creates and saves a figure with the confusion matrix
    saved to file with same name as title and .png extension.
    '''
    confusion_matrix = confusion_matrix.to('cpu')
    n_classes = confusion_matrix.shape[0]
    ticks = []
    for i in range(n_classes):
        ticks.append(index_to_label[i])
    fig, ax = plt.subplots(figsize=(16, 16))
    ax.matshow(confusion_matrix, cmap='Reds')
    for i in range(n_classes):
        for j in range(n_classes):
            ax.text(x=j, y=i,s=int(confusion_matrix[i, j]), va='center', ha='center', size='xx-large')
    # plt.imshow(confusion_matrix)
    ax.tick_params(axis="x", bottom=True, top=False, labelbottom=True, labeltop=False)
    ax.set_xticks(torch.arange(n_classes), ticks, fontsize=14, rotation=45, ha='right')
    ax.set_yticks(torch.arange(n_classes), ticks, fontsize=14)

    plt.title(title, fontsize=24)
    plt.xlabel('Predictions', fontsize=18)
    plt.ylabel('Actual', fontsize=18)
    file_path = title.replace(' ', '_').lower()
    plt.savefig(file_path + '.png')