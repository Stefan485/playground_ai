import os
import logging
import warnings
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from time import time
from datetime import datetime
import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from datasets import load_dataset, Dataset
from dataclasses import dataclass
from PIL import Image

from model import ViT


class HFDataset(torch.utils.data.Dataset):
    def __init__(self, hf_dataset, transform=None):
        self.dataset = hf_dataset
        self.transform = transform

    def __getitem__(self, idx):
        # Ensure idx is an int (avoid tuples/lists)
        if isinstance(idx, (tuple, list, np.ndarray)):
            idx = int(idx[0]) if len(np.atleast_1d(idx)) > 0 else 0

        item = self.dataset[int(idx)]
        image, label = item["image"], item["label"]

        # Convert to PIL Image if needed
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image.astype(np.uint8))
        elif isinstance(image, list):
            image = Image.fromarray(np.array(image, dtype=np.uint8))

        if self.transform:
            image = self.transform(image)

        return image, label

    def __len__(self):
        return len(self.dataset)
    
class Trainer:
    """
    Trainer class for training the model

    Args:
    model (ViT): The model used for the experiment
    optimizer (optim): Optimizer used for training
    loss_func (nn): Loss function used for training
    exp_name (str): Name of the experiment
    device (str): Device to use for training
    scheduler (optim.lr_scheduler): Scheduler used for training
    """
    def __init__(self, model, optimizer, loss_func:str, device:str, scheduler):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.loss_func = loss_func
        self.device = device
        self.scheduler = scheduler


    def train(self, trainloader, valloader, epochs: int, early_stop_patience: int = 5):
        """
        Training function for the model

        Args:
        trainloader (DataLoader): DataLoader for training data
        valloader (DataLoader): DataLoader for validation data
        epochs (int): Number of epochs to train the model
        early_stop_patience (int, optional): Number of epochs to wait for improvement in validation loss before early stopping. Defaults to 5.

        Returns:
        None
        """
        train_losses, val_losses, accuracies = [], [], []
        best_val_loss = float('inf')
        patience_counter = 0

        try:
            outdir = os.path.join("experiments", datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))

            for epoch in range(epochs):
                start = time()
                train_loss = self.train_epoch(trainloader)
                val_accuracy, val_loss = self.test(valloader)

                train_losses.append(train_loss)
                val_losses.append(val_loss)
                accuracies.append(val_accuracy)

                print(f"Epoch: {epoch + 1}, Train loss: {train_loss:.4f}, Val loss: {val_loss:.4f}, Val accuracy: {val_accuracy:.4f}, Time: {time() - start:.4f}")

                # Check for early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    save_checkpoint(self.model, epoch + 1, outdir)
                    print("-------- Saved Best Model! --------")
                else:
                    patience_counter += 1
                    print("Early Stop Left: {}".format(early_stop_patience - patience_counter))

                if (early_stop_patience - patience_counter) == 0:
                    print("-------- Early Stop! --------")
                    break

            save_checkpoint(self.model, epochs, outdir)

        except KeyboardInterrupt:
            print("Keyboard interrupt detected. Saving the model...")
            save_checkpoint(self.model, epoch + 1, outdir)
            print("Model saved successfully.")

        return train_losses, val_losses, accuracies


    def train_epoch(self, trainloader):
        """
        Training function for one epoch

        Args:
        trainloader (DataLoader): DataLoader for training data

        Returns:
        train_loss (float): Training loss
        """
        self.model.train()
        total_loss = 0

        for batch in trainloader:
            batch = [t.to(self.device) for t in batch]
            images, labels = batch
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.loss_func(outputs, labels)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item() * len(images)
            self.scheduler.step(loss)
        
        return total_loss / len(trainloader.dataset)


    @torch.no_grad()
    def test(self, testloader):
        """
        Testing function for the model

        Args:
        testloader (DataLoader): DataLoader for testing data

        Returns:
        accuracy (float): Accuracy of the model
        avg_loss (float): Average loss of the model
        """
        self.model.eval()
        total_loss = 0
        correct = 0
        with torch.no_grad():
            for batch in testloader:
                batch = [t.to(self.device) for t in batch]
                images, labels = batch

                logits = self.model(images)

                loss = self.loss_func(logits, labels)
                total_loss += loss.item() * len(images)

                # Calculate the accuracy
                predictions = torch.argmax(logits, dim=1)
                correct += torch.sum(predictions == labels).item()
        accuracy = correct / len(testloader.dataset)
        avg_loss = total_loss / len(testloader.dataset)
        return accuracy, avg_loss


def plot_metrics(train_losses, val_losses, accuracies):
    """
    Plot the training and validation metrics

    Args:
    train_losses (list): List of training losses
    val_losses (list): List of validation losses

    Returns:
    None
    """
    epochs = range(1, len(train_losses) + 1)

    # Plot losses
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, val_losses, label='Validation Loss')
    plt.title('Training and Validation Losses')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Plot accuracies
    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracies, label='Validation Accuracy', color='green')
    plt.title('Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join("experiments", "metrics.png"))
    plt.show()


@torch.no_grad()
def test(model, device, testloader, loss_func):
    model.eval()
    total_loss = 0
    correct = 0
    for batch in testloader:
        batch = [t.to(device) for t in batch]
        images, labels = batch

        logits = model(images)
        loss = loss_func(logits, labels)

        predictions = torch.argmax(logits, dim=1)
        correct += torch.sum(predictions == labels).item()
        total_loss += loss.item()
    
    accuracy = correct / len(testloader.dataset)
    avg_loss = total_loss / len(testloader.dataset)
    return accuracy, avg_loss


def test_visualize(model, device, testloader, classes):
        """
        Visualize the predictions of the model
        """
        model.eval()
        with torch.no_grad():
            for batch in testloader:
                batch = [t.to(device) for t in batch]
                images, labels = batch

                logits = model(images)
                predictions = torch.argmax(logits, dim=1)

                for i in range(len(images)):
                    image = images[i]
                    label = labels[i]
                    prediction = predictions[i]

                    plt.imshow(image.permute(1, 2, 0).cpu())
                    plt.title(f"Label: {classes[label]}, Prediction: {classes[prediction]}")
                    plt.show()


def setup_seed(seed=3407):
	os.environ['PYTHONHASHSEED'] = str(seed)

	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)

	np.random.seed(seed)
	random.seed(seed)

	torch.backends.cudnn.deterministics = True
	torch.backends.cudnn.benchmarks = False
	torch.backends.cudnn.enabled = False

@dataclass
class Configuration:
    patch_size = 16
    hidden_size = 64
    num_hidden_layers = 2
    num_attention_heads = 3
    intermediate_size = 4 * hidden_size
    num_classes = 100
    hidden_dropout_prob = 0.04107314717204764
    attention_probs_dropout_prob = 0.12206211476886523
    image_size = 224
    num_channels = 3
    qkv_bias = True


    def as_dict(self):

        return {
            "patch_size": self.patch_size,
            "hidden_size": self.hidden_size,
            "num_hidden_layers": self.num_hidden_layers,
            "num_attention_heads": self.num_attention_heads,
            "intermediate_size": 4 * self.hidden_size,
            "num_classes": self.num_classes,
            "hidden_dropout_prob": self.hidden_dropout_prob,
            "attention_probs_dropout_prob": self.attention_probs_dropout_prob,
            "image_size": self.image_size,
            "num_channels": self.num_channels,
            "qkv_bias": self.qkv_bias
        }

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    early_stop_patience = 5
    num_workers = 4
    batch_size = 32
    epochs = 40
    learning_rate = 3e-4


    config = Configuration()

    trainloader, valloader, testloader = prepare_data(batch_size=batch_size, num_workers=num_workers)

    model = ViT(**config.as_dict()).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)
    loss_func = nn.CrossEntropyLoss()
    trainer = Trainer(model, optimizer, loss_func, device, scheduler)

    print("-------- Start Training! --------")
    train_losses, val_losses, accuracies = trainer.train(trainloader, valloader, epochs, early_stop_patience)


    print("-------- Start Testing! --------")
    accuracy, avg_loss = trainer.test(testloader)
    print(f"Test loss: {avg_loss:.4f}, Test accuracy: {accuracy:.4f}")
    print("-------- Testing Finished! --------\n\n")
    plot_metrics(train_losses, val_losses, accuracies)

    loss_func = nn.CrossEntropyLoss()
    testloader = prepare_test_data(batch_size=batch_size, num_workers=num_workers)

    # test_visualize(testloader, model, device)
    accuracy, avg_loss = test(model, device, testloader, loss_func)
    print(f"Test loss: {avg_loss:.4f}, Test accuracy: {accuracy:.4f}")


IMAGE_SIZE = 224
def prepare_data(batch_size: int = 128, num_workers: int = 12, shuffle: bool = True):

    train_transform = transforms.Compose([
        transforms.Lambda(lambda img: img.convert("RGB")),
        transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.8, 1.0), ratio=(0.75, 1.33), antialias=True),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)),
    ])

    transform = transforms.Compose([
        transforms.Lambda(lambda img: img.convert("RGB")),
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
    ])

    ds = load_dataset("timm/mini-imagenet", cache_dir="./data")

    train_dataset = HFDataset(ds["train"], train_transform)
    val_dataset = HFDataset(ds["validation"], transform)
    test_dataset = HFDataset(ds["test"], transform)


    trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)#, persistent_workers=True, pin_memory=True)
    valloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)#, persistent_workers=True, pin_memory=True)
    testloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)#, persistent_workers=True, pin_memory=True)

    return trainloader, valloader, testloader


def prepare_test_data(batch_size: int = 4, num_workers: int = 2):
    """
    Prepares the data for testing

    Args:
    root_dir (str): Path to the root directory of the dataset
    batch_size (int, optional): Batch size. Defaults to 4.
    num_workers (int, optional): Number of workers. Defaults to 2.

    Returns:
    testloader (DataLoader): DataLoader for testing data
    classes (list): List of classes in the dataset
    """

    # Define the transformations
    transform = transforms.Compose([
        transforms.Lambda(lambda img: img.convert("RGB")),
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
    ])

    ds = load_dataset("timm/mini-imagenet", cache_dir="./data")

    test_dataset = HFDataset(ds["test"], transform)

    testloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)#, persistent_workers=True, pin_memory=True)

    return testloader


def save_checkpoint(model, epoch:int, outdir:str):
    """
    Saves the model checkpoint

    Args:
    experiment_name (str): Name of the experiment
    model (ViT): The model used for the experiment
    epoch (int): Epoch number
    base_dir (str, optional): Base directory where the experiment is saved. Defaults to "experiments".
    
    Returns:
    None
    """
    os.makedirs(outdir, exist_ok=True)
    cpfile = os.path.join(outdir, f'model_{epoch:3d}.pt')
    torch.save(model.state_dict(), cpfile)


if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    setup_seed(42)

    main()