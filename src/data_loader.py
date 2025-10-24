# src/data_loader.py

import torch
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import TensorDataset, DataLoader, Subset, random_split

def get_mnist_data_loaders(pair, error_type, error_rate, noise_source_class, batch_size, validation_split, num_workers):
    """Prepares MNIST data with specified label noise for a binary classification task."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # --- 1. Load full datasets ---
    train_dataset_full = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset_full = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    # --- 2. Filter for the specified pair ---
    def filter_pair(dataset, pair):
        indices = [i for i, (img, label) in enumerate(dataset) if label in pair]
        return Subset(dataset, indices)

    train_subset = filter_pair(train_dataset_full, pair)
    test_subset = filter_pair(test_dataset_full, pair)
    
    # --- 3. Extract data and remap labels to 0 and 1 ---
    def extract_and_remap(subset, pair):
        data = torch.stack([img for img, label in subset])
        labels = torch.tensor([0 if label == pair[0] else 1 for img, label in subset])
        return data, labels

    train_data, train_labels = extract_and_remap(train_subset, pair)
    test_data, test_labels = extract_and_remap(test_subset, pair)

    # --- 4. Inject Label Noise into the training set ---
    n_train = len(train_labels)
    n_noise = int(n_train * error_rate)
    
    if n_noise > 0:
        noise_indices = np.random.choice(n_train, n_noise, replace=False)
        
        if error_type == "mutual_mixing":
            # Flip the label (0 -> 1, 1 -> 0)
            train_labels[noise_indices] = 1 - train_labels[noise_indices]
        
        elif error_type == "external_noise":
            # Find indices of the noise source class
            noise_source_indices = [i for i, (img, label) in enumerate(train_dataset_full) if label == noise_source_class]
            # Take a sample of noise images
            noise_images_indices = np.random.choice(noise_source_indices, n_noise, replace=len(noise_source_indices) < n_noise)
            
            # Replace original image and label with external noise
            train_data[noise_indices] = torch.stack([train_dataset_full[i][0] for i in noise_images_indices])
            # Assign the noisy label randomly to one of the pair classes
            train_labels[noise_indices] = torch.randint(0, 2, (n_noise,))

    # --- 5. Create Train/Validation Split ---
    train_val_dataset = TensorDataset(train_data, train_labels)
    n_val = int(len(train_val_dataset) * validation_split)
    n_train = len(train_val_dataset) - n_val
    train_dataset, val_dataset = random_split(train_val_dataset, [n_train, n_val])
    
    test_dataset = TensorDataset(test_data, test_labels)

    # --- 6. Create DataLoaders ---
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    
    return train_loader, val_loader, test_loader