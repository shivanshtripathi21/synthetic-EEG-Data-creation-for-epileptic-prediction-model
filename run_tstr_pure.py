"""
TSTR Experiment 2 - STRICTLY as per Paper 1
Train = Synthetic Preictal + Real Interictal (0 real preictal anchor)
Test  = Real Preictal + Real Interictal
"""
import os, sys, json, torch
import numpy as np
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import cv2
from tqdm import tqdm

PROJECT_ROOT = r"C:\Users\SHIVA\Downloads\Paper1_Reproduction-20260629T171224Z-3-001"
SAVE_MODEL   = os.path.join(PROJECT_ROOT, "05_results", "models")
os.makedirs(SAVE_MODEL, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ── Load image paths ──────────────────────────────────────────────────────────
real_pre_paths   = sorted(Path(PROJECT_ROOT, "02_preprocessed", "preictal").rglob("*.png"))
real_inter_paths = sorted(Path(PROJECT_ROOT, "02_preprocessed", "interictal").rglob("*.png"))

accepted_list_path = os.path.join(PROJECT_ROOT, "05_results", "tables", "accepted_synthetic.json")
with open(accepted_list_path) as f:
    syn_paths_accepted = [Path(p) for p in json.load(f)]

print(f"Real preictal   : {len(real_pre_paths)}")
print(f"Real interictal : {len(real_inter_paths)}")
print(f"Synthetic       : {len(syn_paths_accepted)}")

# ── Dataset ───────────────────────────────────────────────────────────────────
class CESPDataset(Dataset):
    def __init__(self, preictal_paths, interictal_paths, synthetic_paths=None, img_size=128):
        self.items = (
            [(p, 1) for p in preictal_paths] +
            [(p, 0) for p in interictal_paths] +
            ([(p, 1) for p in synthetic_paths] if synthetic_paths else [])
        )
        self.img_size = img_size

    def __len__(self): return len(self.items)

    def __getitem__(self, idx):
        path, label = self.items[idx]
        img = cv2.imread(str(path))
        if img is None:
            img = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
        img = cv2.resize(img, (self.img_size, self.img_size))
        img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1) / 255.0
        return img, torch.tensor(label, dtype=torch.float32)

# ── Model ─────────────────────────────────────────────────────────────────────
class CESP(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(4)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128*4*4, 256), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(256, 1)
        )
    def forward(self, x): return self.classifier(self.features(x)).squeeze(1)

# ── Train / Eval ──────────────────────────────────────────────────────────────
def train_cesp(train_ds, val_ds, epochs=50, lr=1e-4, batch_size=32,
               threshold=0.35, pos_weight_scale=2.0):
    model = CESP().to(device)
    pos_w = torch.tensor([pos_weight_scale], dtype=torch.float32).to(device)
    crit  = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    opt   = torch.optim.Adam(model.parameters(), lr=lr)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    model.train()
    for ep in range(epochs):
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            opt.zero_grad()
            crit(model(X), y).backward()
            opt.step()

    # Evaluate
    model.eval()
    val_loader = DataLoader(val_ds, batch_size=64)
    all_preds, all_labels = [], []
    with torch.no_grad():
        for X, y in val_loader:
            logits = model(X.to(device)).cpu().numpy()
            probs  = 1 / (1 + np.exp(-logits))
            all_preds.extend((probs >= threshold).astype(int))
            all_labels.extend(y.numpy().astype(int))

    all_preds   = np.array(all_preds)
    all_labels  = np.array(all_labels)
    TP = ((all_preds == 1) & (all_labels == 1)).sum()
    TN = ((all_preds == 0) & (all_labels == 0)).sum()
    FP = ((all_preds == 1) & (all_labels == 0)).sum()
    FN = ((all_preds == 0) & (all_labels == 1)).sum()
    sen  = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    spec = TN / (TN + FP) if (TN + FP) > 0 else 0.0
    acc  = (TP + TN) / len(all_labels)

    return model, {"sensitivity": sen, "specificity": spec, "accuracy": acc}

# ── TSTR Pure (Exactly as per Paper 1) ───────────────────────────────────────
def train_cesp_tstr_pure(real_pre_paths, real_inter_paths, syn_paths,
                          epochs=50, lr=1e-4, batch_size=32, n_folds=10,
                          threshold=0.35, pos_weight_scale=2.0):
    """
    STRICTLY Paper 1 TSTR:
      Train = Synthetic Preictal + Real Interictal
      Test  = Real Preictal + Real Interictal
    """
    real_pre_arr   = np.array([str(p) for p in real_pre_paths])
    real_inter_arr = np.array([str(p) for p in real_inter_paths])
    syn_arr        = np.array([str(p) for p in syn_paths])

    all_real   = np.concatenate([real_pre_arr, real_inter_arr])
    all_labels = np.array([1]*len(real_pre_arr) + [0]*len(real_inter_arr))

    skf     = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    results = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(all_real, all_labels)):
        val_paths   = all_real[val_idx]
        val_labels  = all_labels[val_idx]
        val_pre     = val_paths[val_labels == 1]
        val_inter   = val_paths[val_labels == 0]

        tr_inter   = all_real[train_idx][all_labels[train_idx] == 0]

        # Balance synthetic to interictal count
        n_inter    = len(tr_inter)
        syn_subset = syn_arr[:n_inter] if len(syn_arr) >= n_inter else syn_arr

        # Train: 0 Real Preictal + Real Interictal + Synthetic Preictal (PURE)
        train_ds = CESPDataset([], list(tr_inter), list(syn_subset))
        val_ds   = CESPDataset(list(val_pre), list(val_inter))

        _, metrics = train_cesp(train_ds, val_ds, epochs=epochs, lr=lr,
                                batch_size=batch_size, threshold=threshold,
                                pos_weight_scale=pos_weight_scale)
        results.append(metrics)
        print(f"  Fold {fold+1:2d} | Sen: {metrics['sensitivity']:.4f} | "
              f"Spec: {metrics['specificity']:.4f} | Acc: {metrics['accuracy']:.4f} | "
              f"[0 real preictal — PURE SYNTHETIC]")
    return results

# ── RUN ───────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("EXPERIMENT 2: TSTR — Train Synthetic, Test Real (PAPER 1)")
print("="*60)
print("[INFO] Running 10-Fold TSTR (0 real preictal anchor)...")

tstr_results = train_cesp_tstr_pure(
    real_pre_paths, real_inter_paths, syn_paths_accepted,
    epochs=50, lr=1e-4, batch_size=32, n_folds=10,
    threshold=0.35, pos_weight_scale=2.0
)

tstr_sen  = np.mean([r["sensitivity"]  for r in tstr_results])
tstr_spec = np.mean([r["specificity"]  for r in tstr_results])
tstr_acc  = np.mean([r["accuracy"]     for r in tstr_results])

print(f"\n{'='*60}")
print(f"TSTR FINAL | Sen: {tstr_sen:.4f} | Spec: {tstr_spec:.4f} | Acc: {tstr_acc:.4f}")
print(f"Paper target: Sen ~0.8821 (88.21%)")
print(f"{'='*60}")

# Save metrics
save_path = os.path.join(PROJECT_ROOT, "05_results", "tables", "tstr_metrics.json")
with open(save_path, 'w') as f:
    json.dump({"sensitivity": float(tstr_sen), "specificity": float(tstr_spec), "accuracy": float(tstr_acc)}, f, indent=2)
print(f"\n✅ TSTR metrics saved to: {save_path}")
