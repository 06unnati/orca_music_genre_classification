import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os

# ==================== 1. MFCC Dataset ====================
class MFCCDataset(Dataset):
    def __init__(self, mfcc_dir, target_size=(1, 20, 1300)):
        self.mfcc_dir = mfcc_dir
        self.target_size = target_size
        self.genres = os.listdir(mfcc_dir)
        self.files = []

        for genre in self.genres:
            genre_path = os.path.join(mfcc_dir, genre)
            for file_name in os.listdir(genre_path):
                if file_name.endswith('.npy'):
                    self.files.append((os.path.join(genre_path, file_name), genre))

    def __len__(self):
        return len(self.files)

    def resize_or_pad(self, spectrogram):
        current_size = spectrogram.shape[2]
        if current_size < self.target_size[2]:
            padding = self.target_size[2] - current_size
            spectrogram = np.pad(spectrogram, ((0, 0), (0, 0), (0, padding)), mode='constant')
        elif current_size > self.target_size[2]:
            spectrogram = spectrogram[:, :, :self.target_size[2]]
        return spectrogram

    def __getitem__(self, idx):
        file_path, genre = self.files[idx]
        spectrogram = np.load(file_path)
        spectrogram = np.expand_dims(spectrogram, axis=0)
        spectrogram = self.resize_or_pad(spectrogram)
        spectrogram = torch.tensor(spectrogram, dtype=torch.float32)
        label = self.genres.index(genre)
        return spectrogram, label

# ==================== 2. STFT Dataset ====================
class STFTDataset(Dataset):
    def __init__(self, stft_dir, target_size=(1, 1025, 1300)):
        self.stft_dir = stft_dir
        self.target_size = target_size
        self.genres = os.listdir(stft_dir)
        self.files = []

        for genre in self.genres:
            genre_path = os.path.join(stft_dir, genre)
            for file_name in os.listdir(genre_path):
                if file_name.endswith('.npy'):
                    self.files.append((os.path.join(genre_path, file_name), genre))

    def __len__(self):
        return len(self.files)

    def resize_or_pad(self, spectrogram):
        current_size = spectrogram.shape[2]
        if current_size < self.target_size[2]:
            padding = self.target_size[2] - current_size
            spectrogram = np.pad(spectrogram, ((0, 0), (0, 0), (0, padding)), mode='constant')
        elif current_size > self.target_size[2]:
            spectrogram = spectrogram[:, :, :self.target_size[2]]
        return spectrogram

    def __getitem__(self, idx):
        file_path, genre = self.files[idx]
        spectrogram = np.load(file_path)
        spectrogram = np.expand_dims(spectrogram, axis=0)
        spectrogram = self.resize_or_pad(spectrogram)
        spectrogram = torch.tensor(spectrogram, dtype=torch.float32)
        label = self.genres.index(genre)
        return spectrogram, label

# ==================== 3. Combined Dataset ====================
class CombinedAudioDataset(Dataset):
    def __init__(self, mfcc_dataset, stft_dataset):
        assert len(mfcc_dataset) == len(stft_dataset)
        self.mfcc_dataset = mfcc_dataset
        self.stft_dataset = stft_dataset

    def __len__(self):
        return len(self.mfcc_dataset)

    def __getitem__(self, idx):
        mfcc, label1 = self.mfcc_dataset[idx]
        stft, label2 = self.stft_dataset[idx]
        assert label1 == label2
        return mfcc, stft, label1

# ==================== 4. Dual-Branch CNN ====================
class DualBranchGenreClassifier(nn.Module):
    def __init__(self, mfcc_shape=(1, 20, 1300), stft_shape=(1, 1025, 1300)):
        super().__init__()

        self.mfcc_conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.stft_conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        # Compute flattened sizes
        self.mfcc_flattened = self._get_flattened_size(self.mfcc_conv, mfcc_shape)
        self.stft_flattened = self._get_flattened_size(self.stft_conv, stft_shape)

        combined_size = self.mfcc_flattened + self.stft_flattened

        self.fc = nn.Sequential(
            nn.Linear(combined_size, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 10)
        )

    def _get_flattened_size(self, conv, input_shape):
        dummy = torch.zeros(1, *input_shape)
        out = conv(dummy)
        return out.view(1, -1).shape[1]

    def forward(self, mfcc_input, stft_input):
        mfcc_out = self.mfcc_conv(mfcc_input)
        mfcc_out = mfcc_out.view(mfcc_out.size(0), -1)

        stft_out = self.stft_conv(stft_input)
        stft_out = stft_out.view(stft_out.size(0), -1)

        combined = torch.cat((mfcc_out, stft_out), dim=1)
        return self.fc(combined)

# ==================== 5. Accuracy Helper ====================
def calculate_accuracy(output, labels):
    _, predicted = torch.max(output, 1)
    correct = (predicted == labels).sum().item()
    return correct / labels.size(0)

# ==================== 6. Load Data ====================
mfcc_dir = 'C:/Users/14252/Desktop/CSES/mfccFeatures'
stft_dir = 'C:/Users/14252/Desktop/CSES/stftFeatures'

mfcc_dataset = MFCCDataset(mfcc_dir)
stft_dataset = STFTDataset(stft_dir)
combined_dataset = CombinedAudioDataset(mfcc_dataset, stft_dataset)

train_loader = DataLoader(combined_dataset, batch_size=32, shuffle=True)

# ==================== 7. Train Model ====================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = DualBranchGenreClassifier().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 10
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    running_acc = 0.0

    for mfcc, stft, label in train_loader:
        mfcc, stft, label = mfcc.to(device), stft.to(device), label.to(device)

        optimizer.zero_grad()
        output = model(mfcc, stft)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()

        acc = calculate_accuracy(output, label)
        running_loss += loss.item()
        running_acc += acc

    avg_loss = running_loss / len(train_loader)
    avg_acc = running_acc / len(train_loader)

    print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | Accuracy: {avg_acc*100:.2f}%")

# ==================== 8. Save Model ====================
torch.save(model.state_dict(), 'dual_branch_genre_classifier.pth')