import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import random
from model import Configuration, VisionTransformer
from datasets import load_dataset
from tqdm.auto import tqdm
import math

class HFDataset(Dataset):
    def __init__(self, hf_dataset, transform=None):
        self.dataset = hf_dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image, label = item["image"], item["label"]
        if self.transform:
            image = self.transform(image)
        return image, label


max_lr = 3e-3
min_lr = max_lr * 0.01
warmup_ep = 10
max_ep = 50

def get_lr(it):

    if it < warmup_ep:
        return max_lr * (it + 1) / warmup_ep

    if it > max_ep:
        return min_lr

    decay_ratio = (it - warmup_ep) / (max_ep - warmup_ep)
    coeff = 0.5* (1.0 + math.cos(math.pi + decay_ratio))
    return min_lr * coeff * (max_ep - warmup_ep) 

def train_step(model, loader, optimizer, criterion, device, epoch):
    model.train()
    total_loss, correct = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        lr = get_lr(epoch)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        correct += (out.argmax(1) == y).sum().item()
    return total_loss / len(loader.dataset), correct / len(loader.dataset)

def evaluate(model, loader, device):
    model.eval()
    correct = 0
    with torch.inference_mode():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            correct += (out.argmax(1) == y).sum().item()
    return correct / len(loader.dataset)

def train(model, train_loader, valid_loader, optimizer, criterion, epochs, device):
    train_accuracies, valid_accuracies = [], []
    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step(model, train_loader, optimizer, criterion, device, epoch)
        valid_acc = evaluate(model, valid_loader, device)
        train_accuracies.append(train_acc)
        valid_accuracies.append(valid_acc)
        print(f"Epoch {epoch+1}/{epochs} | Train loss={train_loss:.4f}, Train acc={train_acc:.4f}, Valid acc={valid_acc:.4f}")
    return train_accuracies, valid_accuracies

# -------------------
# TRANSFORMS
# -------------------
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.Lambda(lambda img: img.convert("RGB")),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    torch.manual_seed(32)
    torch.cuda.manual_seed(32)
    random.seed(32)

    config = Configuration()
    model = VisionTransformer(**config.as_dict()).to(device)
    torch.set_float32_matmul_precision('high')

    # Optional: torch.compile for speed
    # model = torch.compile(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=3e-4, betas=(0.9, 0.95))
    epochs = 50

    # ds = load_dataset("benjamin-paine/imagenet-1k-64x64", cache_dir="./data")
    ds = load_dataset("timm/mini-imagenet", cache_dir="./data")
    train_dataset = HFDataset(ds["train"], transform)
    valid_dataset = HFDataset(ds["validation"], transform)
    test_dataset = HFDataset(ds["test"], transform)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)


    train_acc, valid_acc = train(model, train_loader, valid_loader, optimizer, criterion, epochs, device)
    test_acc = evaluate(model, test_loader, device=device)
    print(f"Model test ACC: {test_acc}")
    torch.save(model.state_dict(), "./models/mini_imagenet_model.pt")