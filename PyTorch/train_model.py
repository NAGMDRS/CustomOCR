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

def collate_fn(batch):
    imgs, labels = zip(*batch)
    imgs = torch.stack(imgs)  # Stack all images (assumes they are the same size after preprocessing)
    labels = pad_sequence(labels, batch_first=True, padding_value=0)  # Pad labels to the same length
    return imgs, labels

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# Character list
char_list = string.digits + string.ascii_letters + string.punctuation
num_classes = len(char_list) + 1  # +1 for the CTC blank token

# Encode text labels into indices
def encode_to_labels(txt):
    try:
        return [char_list.index(char) for char in txt]
    except ValueError as e:
        print(f"Character not in list: {e}")
        return []

# Preprocessing function
def preprocess_img(img, img_size):
    if img is None:
        img = np.zeros([img_size[1], img_size[0]], dtype=np.uint8)
        print("Image None!")

    img = cv2.resize(img, img_size, interpolation=cv2.INTER_CUBIC)
    img = img / 255.0  # Normalize to [0, 1]
    img = img[np.newaxis, :, :]  # Add channel dimension
    return torch.tensor(img, dtype=torch.float32)

# Dataset class(Can use @Decorators in th future)
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

def compute_output_lengths(input_width, pooling_layers):
    """
    Compute the output width after passing through convolutional layers.
    Args:
        input_width (int): The input image width.
        pooling_layers (list of tuples): Kernel size and stride for each pooling layer.
    Returns:
        int: The output sequence length.
    """
    output_width = input_width
    for kernel_size, stride in pooling_layers:
        output_width = (output_width - kernel_size) // stride + 1
    return output_width

# CRNN Model
class CRNN(nn.Module):
    def __init__(self, num_classes):
        super(CRNN, self).__init__()
        # Define convolutional layers
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),  # Conv1
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),                 
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),  # Conv2
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),                 
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),  # Conv3
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),  # Conv4
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1)),                      
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),  # Conv5
            nn.BatchNorm2d(512),                                   # BatchNorm
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),  # Conv6
            nn.BatchNorm2d(512),                                    #BatchNorm
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1)),                      
            nn.Conv2d(512, 512, kernel_size=2, stride=1),          # Conv7
            nn.ReLU(),
        )

        # Define RNN layers
        self.rnn1 = nn.LSTM(input_size=512, hidden_size=128, bidirectional=True, batch_first=True)
        self.rnn2 = nn.LSTM(input_size=256, hidden_size=128, bidirectional=True, batch_first=True)

        # Fully connected layer for output
        self.fc = nn.Linear(256, num_classes)
        
    def forward(self, x):
        # Convolutional layers
        x = self.conv_layers(x)  # Output shape: (batch, channels, height, width)

        # Reshape for RNN input
        batch_size, channels, height, width = x.size()
        x = x.permute(0, 3, 1, 2).contiguous()  # (batch, width, channels, height)
        x = x.view(batch_size, width, -1)  # (batch, width, features)

        # RNN layers
        x, _ = self.rnn1(x)  # First LSTM
        x, _ = self.rnn2(x)  # Second LSTM

        # Fully connected layers
        x = self.fc(x)

        return x

# CTC Loss Wrapper
ctc_loss = nn.CTCLoss(blank=num_classes - 1, zero_infinity=True)
def train_model(model, dataloader, optimizer, num_epochs):
    model.train()
    pooling_layers = [(2, 2), (2, 2), (2, 1), (2, 1)]  # Pooling configuration for the CNN
    sequence_length = compute_output_lengths(img_size[0], pooling_layers)  # Compute sequence length

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for imgs, labels in dataloader:
            imgs, labels = imgs.to(device), labels.to(device)

            optimizer.zero_grad()

            # Forward pass
            logits = model(imgs)  # Shape: (batch_size, sequence_length, num_classes) Hopefully

            # Transpose logits for CTC loss
            log_probs = logits.log_softmax(2).permute(1, 0, 2)  # Shape: (sequence_length, batch_size, num_classes)

            # Define input_lengths dynamically
            input_lengths = torch.full(
                size=(log_probs.size(1),),  # Batch size
                fill_value=sequence_length,  # Sequence length after pooling
                dtype=torch.long,
                device=device
            )
            label_lengths = torch.sum(labels != 0, dim=1).to(device)  # Non-padded label lengths

            # Validate batch shapes
            if log_probs.size(1) != labels.size(0):
                print(f"Batch size mismatch! log_probs: {log_probs.size(1)}, labels: {labels.size(0)}")
                continue

            # Compute CTC loss
            try:
                loss = ctc_loss(log_probs, labels, input_lengths, label_lengths)
            except RuntimeError as e:
                print(f"CTC Loss error: {e}")
                print(f"Logits shape: {logits.shape}")
                print(f"Labels shape: {labels.shape}")
                print(f"Input lengths: {input_lengths.shape}, Label lengths: {label_lengths.shape}")
                continue  # Skip this batch if an error occurs

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            epoch_loss += loss.item()

        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss / len(dataloader):.4f}")

def validate_model(model, dataloader):
    model.eval()
    pooling_layers = [(2, 2), (2, 2), (2, 1), (2, 1)]  # Pooling configuration for the CNN
    sequence_length = compute_output_lengths(img_size[0], pooling_layers)  # Compute sequence length

    total_loss = 0.0
    with torch.no_grad():
        for imgs, labels in dataloader:
            imgs, labels = imgs.to(device), labels.to(device)
            
            # Forward pass
            logits = model(imgs)  # Shape: (batch_size, sequence_length, num_classes)
            batch_size = logits.size(0)  # Dynamically determine batch size
            
            # Transpose logits for CTC loss
            log_probs = logits.log_softmax(2).permute(1, 0, 2)  # Shape: (sequence_length, batch_size, num_classes)
            
            # Input lengths: all equal to sequence_length
            input_lengths = torch.full(
                size=(batch_size,),
                fill_value=sequence_length,
                dtype=torch.long,
                device=device
            )

            # Label lengths: sum of non-padded values
            label_lengths = torch.sum(labels != 0, dim=1).to(device)  # Shape: (batch_size,)

            # Validate batch shapes
            if log_probs.size(1) != labels.size(0):
                print(f"Batch size mismatch! log_probs: {log_probs.size(1)}, labels: {labels.size(0)}")
                continue

            # Calculate validation loss
            try:
                loss = ctc_loss(log_probs, labels, input_lengths, label_lengths)
            except RuntimeError as e:
                print(f"CTC Loss error: {e}")
                print(f"Logits shape: {logits.shape}")
                print(f"Labels shape: {labels.shape}")
                print(f"Input lengths: {input_lengths.shape}, Label lengths: {label_lengths.shape}")
                continue  # Skip this batch if an error occurs

            total_loss += loss.item()

    print(f"Validation Loss: {total_loss / len(dataloader):.4f}")

# Paths and constants
annotation_file = "/home/btech10154.22/OCRData/annotation.txt"
image_dir = "/home/btech10154.22/OCRData/images/"
img_size = (128, 32)
batch_size = 256
num_epochs = 30

# Load data
with open(annotation_file, 'r') as f:
    annotations = [line.strip().split(',') for line in f.readlines()]

data = [(os.path.join(image_dir, name), text) for name, text in annotations]
random.shuffle(data)
train_data = data[:int(len(data) * 0.9)]
valid_data = data[int(len(data) * 0.9):]

train_dataset = OCRDataset([x[0] for x in train_data], [x[1] for x in train_data], img_size)
valid_dataset = OCRDataset([x[0] for x in valid_data], [x[1] for x in valid_data], img_size)

train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_fn ,shuffle=True, num_workers=4)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, collate_fn=collate_fn ,shuffle=False, num_workers=4)

# Initialize model, optimizer
model = CRNN(num_classes).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
train_model(model, train_loader, optimizer, num_epochs)

# Save the model weights
torch.save(model.state_dict(), "OCRData/Models/ocr_model1gpu.pkl")

# Validate the model
validate_model(model, valid_loader)
