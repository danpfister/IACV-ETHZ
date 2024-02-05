from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch.utils.data import DataLoader

from evaluation import evaluate_model


def train_model(
    model,
    train_loader,
    val_loader, 
    num_epochs,
    optimizer,
    device,
    save_path=f"./ckpt/model.pt"
):
    """
    Feel free to change the arguments of this function - if necessary.

    Trains the model on the given dataset. Selects the best model based on the
    validation set and saves it to the given path. 
    Inputs: 
        model: The model to train [nn.Module]
        train_loader: The training data loader [DataLoader]
        val_loader: The validation data loader [DataLoader]
        num_epochs: The number of epochs to train for [int]
        optimizer: The optimizer [Any]
        best_of: The metric to use for validation [str: "loss" or "accuracy"]
        device: The device to train on [str: cpu, cuda, or mps]
        save_path: The path to save the model to [str]
    Output:
        Dictionary containing the training and validation losses and accuracies
        at each epoch. Also contains the epoch number of the best model.
    """

    criterion = nn.CrossEntropyLoss()

    results = {
    'train_loss': [],
    'val_loss': [],
    'train_accuracy': [],
    'val_accuracy': [],
    'best_epoch': 0
    }

    best_val_accuracy = 0.0

    for epoch in range(num_epochs):
        # Training
        model.train()  # Set the model to training mode
        running_loss = 0.0
        
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)  # Move data to GPU if available
            
            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()

                
            if i % 10 == 9:  # Print every 100 mini-batches
                print(f"Epoch {epoch+1}, Iteration {i+1}, Loss: {running_loss/100:.4f}")
                running_loss = 0.0
        
        # Calculate training accuracy
        correct_train = 0
        total_train = 0
        with torch.no_grad():
            for data in train_loader:
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                
                total_train += labels.size(0)
                correct_train += (predicted == labels).sum().item()
        
        train_accuracy = correct_train / total_train
        results['train_accuracy'].append(train_accuracy)
        
        # Calculate validation loss and accuracy
        model.eval()  # Set the model to evaluation mode
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        
        with torch.no_grad():
            for data in val_loader:
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()
        
        val_accuracy = correct_val / total_val
        results['val_accuracy'].append(val_accuracy)
        
        # Update best epoch based on validation accuracy
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            results['best_epoch'] = epoch + 1
        
        results['train_loss'].append(running_loss / len(train_loader))
        results['val_loss'].append(val_loss / len(val_loader))
        
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {results['train_loss'][-1]:.4f}, Val Loss: {results['val_loss'][-1]:.4f}, Train Acc: {train_accuracy:.4f}, Val Acc: {val_accuracy:.4f}")

    print("Training finished!")

    # Return the results dictionary
    return results

def plot_training_log(): pass