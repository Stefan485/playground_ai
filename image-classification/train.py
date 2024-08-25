import torch
import torch.optim as optim
from datasets import load_dataset


#Parameters
device = 'cuda' if torch.cuda.is_available() else 'cpu'
batch_size = 64
shuffle = True
num_workers = 4
kernel_size = 4
stride = 2


#Load data
