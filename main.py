import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights
from torch.utils.data import Dataset, DataLoader, random_split
# from data_preprocessing import GeoGuessr_Screenshots_Dataset, calculate_class_weights
import matplotlib.pyplot as plt
import sys
from Utils import GeoGuessr_Screenshots_Dataset, calculate_class_weights, accuracy_per_class, compute_confusion_matrix, display_classwise_accuracy, display_confusion_matrix





def train_wrn(device, min_examples_per_class, random_seed: int = 42, epochs: int = 25, checkpoint_path: str = None, save_path: str = None):
    if checkpoint_path != None:
        print(f"Loading checkpoint from {checkpoint_path}")
        state = torch.load(checkpoint_path)
        e = state['epoch']
        random_seed = state['random_seed']
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((256, 512)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    dataset = GeoGuessr_Screenshots_Dataset('./Data/', min_examples_per_class, transform=transform)
    weights = ResNet50_Weights.DEFAULT
    model = resnet50(weights=weights)
    model.fc = nn.Linear(2048, dataset.get_num_classes())
    if checkpoint_path != None:
        model.load_state_dict(state['model']) 
    model.to(device)

    to_optimize = []
    for name, param in model.named_parameters():
        if 'fc' not in name:
            param.requires_grad = False
        else:
            to_optimize.append(param)
    optim = torch.optim.Adam(to_optimize)
    if checkpoint_path != None:
        optim.load_state_dict(state['optim'])
        
    train, validate, test = random_split(dataset, [0.8, 0.1, 0.1], generator=torch.Generator().manual_seed(random_seed))
    class_weights = calculate_class_weights(train)
    class_weights = class_weights.to(device)
    loader = DataLoader(train, batch_size=128, shuffle=True)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    e = 0 if checkpoint_path == None else e
    while e < epochs:

        print(f"Starting epoch {e}")
        for input, target in loader:
            optim.zero_grad()
            input, target = input.to(device), target.to(device)
            # input = preprocess(input)
            output = model(input)
            loss = criterion(output, target)
            loss.backward()
            optim.step()
            break
        
        if save_path != None:
            state_dict = {
                'epoch' : e, 
                'optim' : optim.state_dict(), 
                'model' : model.state_dict(), 
                'random_seed' : random_seed
                }
            torch.save(state_dict, save_path + '_' + str(e) + '.pt')
        
        if e + 1 % 5 == 0:
            with torch.no_grad():
                display_classwise_accuracy(model, train, 'training', device=device, index_to_class_map=dataset.get_index_to_class_map(), epoch=e)
                display_classwise_accuracy(model, validate, 'validation', device=device, index_to_class_map=dataset.get_index_to_class_map(), epoch=e)
        e += 1


def test_saved_wrn(model_checkpoint_path: str, min_examples_per_class: int, device: torch.cuda.device = 'cpu') -> None:
    state = torch.load(model_checkpoint_path)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((256, 512)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    dataset = GeoGuessr_Screenshots_Dataset('./Data/', min_examples_per_class, transform=transform)
    n_classes = dataset.get_num_classes()
    model = resnet50()
    model.fc = nn.Linear(2048, dataset.get_num_classes())
    model.load_state_dict(state['model'])
    model.to(device)
    model.eval()
    random_seed = state['random_seed']
    train, validate, test = random_split(dataset, [0.8, 0.1, 0.1], generator=torch.Generator().manual_seed(random_seed))
    train_confusion_matrix = compute_confusion_matrix(model, train, n_classes, 128, device)
    test_confusion_matrix = compute_confusion_matrix(model, test, n_classes, 128, device)
    assert int(torch.sum(train_confusion_matrix)) == len(train)
    assert int(torch.sum(test_confusion_matrix)) == len(test)
    index_to_label = dataset.get_index_to_class_map()
    display_confusion_matrix(train_confusion_matrix, "Train Data Confusion Matrix for " + model_checkpoint_path[:model_checkpoint_path.find('.')], index_to_label)
    display_confusion_matrix(test_confusion_matrix, "Test Data Confusion Matrix for " + model_checkpoint_path[:model_checkpoint_path.find('.')], index_to_label)

    with torch.no_grad():
        display_classwise_accuracy(model, test, 'Test', device=device, index_to_class_map=index_to_label)
        display_classwise_accuracy(model, train, 'Train', device=device, index_to_class_map=index_to_label)
    return


def main():
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    if len(sys.argv) != 4 and len(sys.argv) != 5:
        raise ValueError("Usage: python main.py mode checkpoint_path min_examples_per_class [load_path]")
    min_examples_per_class = int(sys.argv[3])
    if sys.argv[1] == 'train':
        save_path = sys.argv[2]
        train_wrn(device, min_examples_per_class, epochs=100, save_path=save_path)
    if sys.argv[1] == 'resume-training':
        save_path = sys.argv[2]
        load_path = sys.argv[4]
        train_wrn(device, min_examples_per_class, epochs=100, save_path=save_path, checkpoint_path=load_path)
    if sys.argv[1] == 'test':
        checkpoint_path = sys.argv[2]
        test_saved_wrn(checkpoint_path, min_examples_per_class, device)
    else:
        raise ValueError("Usage: python main.py mode checkpoint_path min_examples_per_class")


if __name__ == "__main__":
    main()