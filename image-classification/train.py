import torch
import torch.optim as optim
from datasets import load_dataset
from transformers import AutoImageProcessor
from res_net import ResNetForImageClassification

#Parameters
device = 'cuda' if torch.cuda.is_available() else 'cpu'
batch_size = 64
shuffle = True

#Load data
dataset = load_dataset("huggingface/cats-image", trust_remote_code=True)
image = dataset["test"]["image"][0]

proccessor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")

#Downloaded files from huggingface.co https://huggingface.co/microsoft/resnet-50
model = ResNetForImageClassification("res_net_files")

inputs = proccessor(images=image, return_tensors="pt")
inputs = inputs.to(device)

model.to(device)

model.eval()

logits = model(**inputs)["logits"]

predicted_label = logits.argmax(-1).item()
print(model.config.id2label[str(predicted_label)])