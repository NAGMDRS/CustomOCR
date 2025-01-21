import os
import random
import string
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from torch.nn.utils.rnn import pad_sequence

# Character list
char_list = string.digits + string.ascii_letters + string.punctuation
num_classes = len(char_list) + 1  # +1 for the CTC blank token

# GPU configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

def collate_fn(batch):
    imgs, labels = zip(*batch)
    imgs = torch.stack(imgs)  # Stack all images (assumes they are the same size after preprocessing)
    labels = pad_sequence(labels, batch_first=True, padding_value=0)  # Pad labels to the same length
    return imgs, labels

# Encode text labels into indices
def encode_to_labels(txt):
    try:
        return [char_list.index(char) for char in txt]
    except ValueError as e:
        print(f"Character not in list: {e}")
        return []

# Decode indices to text
def decode_predictions(predictions):
    decoded_texts = []
    for pred in predictions:
        text = ""
        previous_char = None
        for idx in pred:
            if idx != previous_char and idx < len(char_list):  # Ignore repeated characters and blanks
                text += char_list[idx]
            previous_char = idx
        decoded_texts.append(text)
    return decoded_texts

# Preprocessing function
def preprocess_img(img, img_size):
    if img is None:
        img = np.zeros([img_size[1], img_size[0]], dtype=np.uint8)
        print("Image None!")

    img = cv2.resize(img, img_size, interpolation=cv2.INTER_CUBIC)
    img = img / 255.0  # Normalize to [0, 1]
    img = img[np.newaxis, :, :]  # Add channel dimension
    return torch.tensor(img, dtype=torch.float32)

# Dataset class
class OCRDataset(Dataset):
    def __init__(self, image_paths, texts, img_size):
        self.image_paths = image_paths
        self.texts = texts
        self.img_size = img_size

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = cv2.imread(self.image_paths[idx], cv2.IMREAD_GRAYSCALE)
        img = preprocess_img(img, self.img_size)
        label = encode_to_labels(self.texts[idx])
        return img, torch.tensor(label, dtype=torch.long)

# Compute output lengths
def compute_output_lengths(input_width, pooling_layers):
    output_width = input_width
    for kernel_size, stride in pooling_layers:
        output_width = (output_width - kernel_size) // stride + 1
    return output_width

# CRNN Model (unchanged)
class CRNN(nn.Module):
    def __init__(self, num_classes):
        super(CRNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1)),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1)),
            nn.Conv2d(512, 512, kernel_size=2, stride=1),
            nn.ReLU(),
        )
        self.rnn1 = nn.LSTM(input_size=512, hidden_size=128, bidirectional=True, batch_first=True)
        self.rnn2 = nn.LSTM(input_size=256, hidden_size=128, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.conv_layers(x)
        batch_size, channels, height, width = x.size()
        x = x.permute(0, 3, 1, 2).contiguous()
        x = x.view(batch_size, width, -1)
        x, _ = self.rnn1(x)
        x, _ = self.rnn2(x)
        x = self.fc(x)
        return x

# Evaluation function for accuracy
def evaluate_model(model, dataloader):
    model.eval()
    pooling_layers = [(2, 2), (2, 2), (2, 1), (2, 1)]  # Pooling configuration
    sequence_length = compute_output_lengths(img_size[0], pooling_layers)

    total_chars = 0
    correct_chars = 0
    with torch.no_grad():
        for imgs, labels in dataloader:
            imgs, labels = imgs.to(device), labels.to(device)

            # Forward pass
            logits = model(imgs)
            log_probs = logits.log_softmax(2).permute(1, 0, 2)
            preds = torch.argmax(log_probs, dim=2).permute(1, 0)

            # Decode predictions
            pred_texts = decode_predictions(preds.cpu().numpy())
            label_texts = decode_predictions(labels.cpu().numpy())

            for pred_text, label_text in zip(pred_texts, label_texts):
                total_chars += len(label_text)
                correct_chars += sum(p == l for p, l in zip(pred_text, label_text))

    accuracy = correct_chars / total_chars * 100
    print(f"Character Accuracy: {accuracy:.2f}%")

# Paths and constants
annotation_file = "/home/btech10154.22/OCRData/annotation.txt"
image_dir = "/home/btech10154.22/OCRData/images/"
img_size = (128, 32)
batch_size = 256

# Load data
with open(annotation_file, 'r') as f:
    annotations = [line.strip().split(',') for line in f.readlines()]

data = [(os.path.join(image_dir, name), text) for name, text in annotations]
random.shuffle(data)
train_data = data[:int(len(data) * 0.9)]
valid_data = data[int(len(data) * 0.9):]

train_dataset = OCRDataset([x[0] for x in train_data], [x[1] for x in train_data], img_size)
valid_dataset = OCRDataset([x[0] for x in valid_data], [x[1] for x in valid_data], img_size)

train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True, num_workers=4)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=False, num_workers=4)

# Initialize model and load weights
model = CRNN(num_classes).to(device)
model_path = "OCRData/Models/ocr_model1gpu.pkl"
model.load_state_dict(torch.load(model_path, map_location=device))

# Evaluate model
evaluate_model(model, valid_loader)
