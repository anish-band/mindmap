import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LABEL_ORDER = ["NEGATIVE", "NEUTRAL", "POSITIVE"]


# ─── Sklearn models ───────────────────────────────────────────────────────────

def train_svm(X_train, y_train):
    print("  Training SVM (RBF kernel)...")
    t0 = time.time()
    clf = SVC(kernel="rbf", C=10, gamma="scale", probability=True, random_state=42)
    clf.fit(X_train, y_train)
    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s")
    return clf, elapsed


def train_rf(X_train, y_train):
    print("  Training Random Forest (200 estimators)...")
    t0 = time.time()
    clf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)
    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s")
    return clf, elapsed


# ─── PyTorch CNN ──────────────────────────────────────────────────────────────

class CNN1D(nn.Module):
    def __init__(self, n_features: int, n_classes: int = 3):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(4),

            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(4),

            nn.Conv1d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(64, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, n_features] → [batch, 1, n_features]
        return self.head(self.encoder(x.unsqueeze(1)))


# ─── PyTorch LSTM ─────────────────────────────────────────────────────────────

class EEGLSTM(nn.Module):
    def __init__(self, input_size: int = 2, hidden_size: int = 128,
                 num_layers: int = 2, n_classes: int = 3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True, dropout=0.3
        )
        self.head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(hidden_size, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, seq_len, input_size]
        _, (h, _) = self.lstm(x)
        return self.head(h[-1])


# ─── Training loop ────────────────────────────────────────────────────────────

def _encode_labels(y_train, y_test):
    le = LabelEncoder()
    le.classes_ = np.array(LABEL_ORDER)
    y_tr = le.transform(y_train)
    y_te = le.transform(y_test)
    return y_tr, y_te, le


def _make_loaders(X_train, y_train, X_test, y_test, batch_size=64):
    X_tr = torch.tensor(X_train, dtype=torch.float32)
    X_te = torch.tensor(X_test,  dtype=torch.float32)
    y_tr = torch.tensor(y_train, dtype=torch.long)
    y_te = torch.tensor(y_test,  dtype=torch.long)
    train_dl = DataLoader(TensorDataset(X_tr, y_tr), batch_size=batch_size, shuffle=True)
    test_dl  = DataLoader(TensorDataset(X_te, y_te), batch_size=batch_size)
    return train_dl, test_dl


def _train_torch(model, train_dl, n_epochs=30, lr=1e-3):
    model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)
    criterion = nn.CrossEntropyLoss()
    t0 = time.time()
    for epoch in range(n_epochs):
        model.train()
        for xb, yb in train_dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            opt.step()
        sched.step()
        if (epoch + 1) % 10 == 0:
            print(f"    Epoch {epoch+1}/{n_epochs} — loss: {loss.item():.4f}")
    return time.time() - t0


def _predict_torch(model, test_dl):
    model.eval()
    preds, probs = [], []
    with torch.no_grad():
        for xb, _ in test_dl:
            logits = model(xb.to(DEVICE))
            p = torch.softmax(logits, dim=1).cpu().numpy()
            probs.append(p)
            preds.append(p.argmax(axis=1))
    return np.concatenate(preds), np.concatenate(probs, axis=0)


def train_cnn(X_train, y_train, X_test, y_test, n_epochs=30):
    print(f"  Training 1D CNN on {DEVICE}...")
    y_tr, y_te, le = _encode_labels(y_train, y_test)
    train_dl, test_dl = _make_loaders(X_train.values, y_tr, X_test.values, y_te)
    model = CNN1D(n_features=X_train.shape[1])
    elapsed = _train_torch(model, train_dl, n_epochs=n_epochs)
    print(f"  Done in {elapsed:.1f}s")
    return model, elapsed, test_dl, y_te, le


def train_lstm(X_train_fft, y_train, X_test_fft, y_test, n_epochs=30):
    """X_train_fft / X_test_fft: numpy arrays [n, seq_len, 2]"""
    print(f"  Training LSTM on {DEVICE}...")
    y_tr, y_te, le = _encode_labels(y_train, y_test)
    seq_len = X_train_fft.shape[1]
    train_dl, test_dl = _make_loaders(
        X_train_fft.reshape(len(X_train_fft), seq_len, 2), y_tr,
        X_test_fft.reshape(len(X_test_fft),  seq_len, 2), y_te,
    )
    model = EEGLSTM(input_size=2, hidden_size=128, num_layers=2)
    elapsed = _train_torch(model, train_dl, n_epochs=n_epochs)
    print(f"  Done in {elapsed:.1f}s")
    return model, elapsed, test_dl, y_te, le
