import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import ast
import numpy as np
import re
from torch.utils.data import random_split

# 1. Helper function to clean and parse tonal_centroid
def clean_tonal_centroid(tonal_centroid_str):
    """Fix any missing commas or formatting issues in tonal_centroid string."""
    # Fix missing commas or incorrect formatting (add a comma between values if needed)
    tonal_centroid_str = re.sub(r'(\d)(\s+)(\d)', r'\1,\3', tonal_centroid_str)  # Example fix
    return tonal_centroid_str

def parse_tonal_centroid(tonal_centroid_str):

    tonal_centroid_str = tonal_centroid_str.replace('...', '').replace('\n', ' ').strip()

    tonal_centroid_str = re.sub(r'(?<=[\d])\s+(?=[+-]?\d)', ',', tonal_centroid_str)

    rows = re.findall(r'\[([^\[\]]+)\]', tonal_centroid_str)
    formatted_rows = ['[' + row.strip() + ']' for row in rows]

    final_str = '[' + ','.join(formatted_rows) + ']'

    try:
        tonal_centroid = np.array(eval(final_str))
    except Exception as e:
        tonal_centroid = np.array([])

    return tonal_centroid

# 2. Dataset class for CSV
class MusicDataset(Dataset):
    def __init__(self, csv_file):
        df = pd.read_csv(csv_file)

        # Convert chroma_features from string to list
        df["chroma_features"] = df["chroma_features"].apply(ast.literal_eval)

        # Apply the parse_tonal_centroid function to tonal_centroid column
        df["tonal_centroid"] = df["tonal_centroid"].apply(parse_tonal_centroid)

        # List to store the features
        self.X = []
        for _, row in df.iterrows():
            tonal_centroid = row["tonal_centroid"]
            if tonal_centroid.size == 0:
                tonal_centroid = np.zeros((6, 6))
            tonal_centroid = tonal_centroid.flatten()

            # Concatenate the features
            features = np.concatenate([  
                [row["spectral_centroid"]],
                [row["spectral_flux_mean"]],
                row["chroma_features"],
                tonal_centroid,
                [row["tempo"]],
                [row["hnr"]],
            ])
            self.X.append(features)

        # Convert to tensor
        self.X = torch.tensor(np.array(self.X), dtype=torch.float32)

        # Normalize the features: Z-score standardization
        self.X = (self.X - self.X.mean(dim=0)) / self.X.std(dim=0)

        # Genre as label
        genres = sorted(df["genre"].unique())
        self.genre_to_idx = {genre: idx for idx, genre in enumerate(genres)}
        self.y = torch.tensor(df["genre"].map(self.genre_to_idx).values, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# 3. DNN Model
class MusicGenreClassifier(nn.Module):
    def __init__(self, input_size, num_classes):
        super(MusicGenreClassifier, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(), 
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.net(x)

# 4. Training function
# 4. Training function without validation
def train_model(csv_path, epochs=30, batch_size=35, learning_rate=0.001):
    dataset = MusicDataset(csv_path)
    
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    input_size = dataset.X.shape[1]
    num_classes = len(dataset.genre_to_idx)

    model = MusicGenreClassifier(input_size, num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

    for epoch in range(epochs):
        # Train
        model.train()
        total_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Train accuracy
        model.eval()
        correct_train, total_train = 0, 0
        with torch.no_grad():
            for X_batch, y_batch in train_loader:
                outputs = model(X_batch)
                _, predicted = torch.max(outputs.data, 1)
                total_train += y_batch.size(0)
                correct_train += (predicted == y_batch).sum().item()
        train_accuracy = 100 * correct_train / total_train

        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}, "
              f"Train Acc: {train_accuracy:.2f}%")

    return model, dataset.genre_to_idx


# 5. Run training
model, genre_map = train_model("C:\\Users\\14252\\Desktop\\CSES\\feature_extraction.csv")
