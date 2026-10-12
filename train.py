import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

IMG_SIZE = 32
SEQ_LEN = 12
N = 1200

os.makedirs("models", exist_ok=True)
os.makedirs("data", exist_ok=True)

class FloodDataset(Dataset):
    def __init__(self, images, rainfall, labels):
        self.images = torch.tensor(images, dtype=torch.float32)
        self.rainfall = torch.tensor(rainfall, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return self.images[i], self.rainfall[i], self.labels[i]

def make_demo_data(n=N):
    # Synthetic satellite-like RGB images and 12-step rainfall sequences.
    # The label is intentionally generated from both spatial and temporal signals.
    images = np.zeros((n, 3, IMG_SIZE, IMG_SIZE), dtype=np.float32)
    rainfall = np.zeros((n, SEQ_LEN, 1), dtype=np.float32)
    labels = np.zeros(n, dtype=np.float32)

    for i in range(n):
        water = np.random.rand() < 0.45
        img = np.random.normal(0.25, 0.06, (3, IMG_SIZE, IMG_SIZE)).astype(np.float32)

        # Green vegetation / land
        img[1] += np.random.uniform(0.05, 0.20)
        # Blue water region for some samples
        if water:
            x1, x2 = np.random.randint(4, 14), np.random.randint(18, 28)
            y1, y2 = np.random.randint(4, 14), np.random.randint(18, 28)
            img[2, x1:x2, y1:y2] += 0.55
            img[0, x1:x2, y1:y2] *= 0.55
            img[1, x1:x2, y1:y2] *= 0.75

        rain = np.random.gamma(shape=2.0, scale=10.0, size=SEQ_LEN)
        if water:
            rain += np.random.uniform(2, 8)
        rainfall[i, :, 0] = rain / 100.0

        # Combined risk signal
        rain_score = rain[-6:].mean()
        spatial_score = 1.0 if water else 0.0
        risk_score = 0.55 * spatial_score + 0.45 * min(rain_score / 60.0, 1.0)
        labels[i] = 1.0 if risk_score > 0.50 else 0.0

        images[i] = np.clip(img, 0, 1)

    return images, rainfall, labels

class CNNLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.lstm = nn.LSTM(input_size=1, hidden_size=32, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(64 + 32, 32), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(32, 1)
        )

    def forward(self, image, rain):
        spatial = self.cnn(image).flatten(1)
        temporal, _ = self.lstm(rain)
        temporal = temporal[:, -1, :]
        return self.fc(torch.cat([spatial, temporal], dim=1)).squeeze(1)

images, rainfall, labels = make_demo_data()
idx_train, idx_test = train_test_split(
    np.arange(N), test_size=0.20, random_state=SEED, stratify=labels
)

train_ds = FloodDataset(images[idx_train], rainfall[idx_train], labels[idx_train])
test_ds = FloodDataset(images[idx_test], rainfall[idx_test], labels[idx_test])
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=64)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = CNNLSTM().to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

EPOCHS = 12
print("Device:", device)
for epoch in range(EPOCHS):
    model.train()
    total = 0
    for x_img, x_rain, y in train_loader:
        x_img, x_rain, y = x_img.to(device), x_rain.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x_img, x_rain)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total += loss.item()

    model.eval()
    preds, actual = [], []
    with torch.no_grad():
        for x_img, x_rain, y in test_loader:
            p = torch.sigmoid(model(x_img.to(device), x_rain.to(device))).cpu().numpy()
            preds.extend((p >= 0.5).astype(int))
            actual.extend(y.numpy().astype(int))
    acc = accuracy_score(actual, preds)
    print(f"Epoch {epoch+1:02d}/{EPOCHS}  Loss={total/len(train_loader):.4f}  Val_Accuracy={acc:.4f}")

precision = precision_score(actual, preds, zero_division=0)
recall = recall_score(actual, preds, zero_division=0)
f1 = f1_score(actual, preds, zero_division=0)
print("\nFinal evaluation")
print(f"Accuracy : {accuracy_score(actual, preds):.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1-score : {f1:.4f}")

torch.save({
    "model_state": model.state_dict(),
    "img_size": IMG_SIZE,
    "seq_len": SEQ_LEN
}, "models/flood_cnn_lstm.pt")

# Save a compact CSV for project documentation/demo.
rows = []
for i in range(N):
    rows.append({
        "sample_id": i,
        "rainfall_last_6_avg_mm": float(rainfall[i, -6:, 0].mean() * 100),
        "flood_label": int(labels[i])
    })
pd.DataFrame(rows).to_csv("data/flood_dataset.csv", index=False)

print("\nSaved: models/flood_cnn_lstm.pt")
print("Saved: data/flood_dataset.csv")
