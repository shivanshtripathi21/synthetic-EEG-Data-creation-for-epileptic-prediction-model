import os, json, torch
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import cv2
from tqdm import tqdm

PROJECT_ROOT = r"C:\Users\SHIVA\Downloads\Paper1_Reproduction-20260629T171224Z-3-001"
RESULTS_DIR = os.path.join(PROJECT_ROOT, "Paper1_Reproduction", "05_results", "tables")
os.makedirs(RESULTS_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ── Load image paths ──────────────────────────────────────────────────────────
preictal_dir = Path(PROJECT_ROOT, "Paper1_Reproduction", "02_preprocessed", "preictal")
interictal_dir = Path(PROJECT_ROOT, "Paper1_Reproduction", "02_preprocessed", "interictal")

real_pre_paths   = np.array(sorted(preictal_dir.rglob("*.png")))
real_inter_paths = np.array(sorted(interictal_dir.rglob("*.png")))

accepted_list_path = os.path.join(PROJECT_ROOT, "Paper1_Reproduction", "05_results", "tables", "accepted_synthetic.json")
with open(accepted_list_path) as f:
    syn_paths_accepted = np.array([Path(p) for p in json.load(f)])

print(f"Loaded Real preictal   : {len(real_pre_paths)}")
print(f"Loaded Real interictal : {len(real_inter_paths)}")
print(f"Loaded Synthetic       : {len(syn_paths_accepted)}")

# ── 80/20 Train-Test Split (Fixed Test Set) ──────────────────────────────────
pre_train, pre_test = train_test_split(real_pre_paths, test_size=0.2, random_state=42)
inter_train, inter_test = train_test_split(real_inter_paths, test_size=0.2, random_state=42)

print(f"\nFixed Train set: {len(pre_train)} preictal, {len(inter_train)} interictal")
print(f"Fixed Test set : {len(pre_test)} preictal, {len(inter_test)} interictal")

# ── Dataset & Model ───────────────────────────────────────────────────────────
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

def train_eval_cesp(train_ds, val_ds, epochs=50, lr=1e-4, batch_size=64,
                    threshold=0.45, pos_weight_scale=1.5):
    model = CESP().to(device)
    pos_w = torch.tensor([pos_weight_scale], dtype=torch.float32).to(device)
    crit  = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    opt   = torch.optim.Adam(model.parameters(), lr=lr)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=64)

    for ep in range(epochs):
        model.train()
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            opt.zero_grad()
            crit(model(X), y).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()

    # Evaluate
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for X, y in val_loader:
            logits = model(X.to(device)).cpu()
            probs  = torch.sigmoid(logits).numpy()
            all_preds.extend((probs >= threshold).astype(int))
            all_labels.extend(y.numpy().astype(int))

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)
    TP = ((all_preds == 1) & (all_labels == 1)).sum()
    TN = ((all_preds == 0) & (all_labels == 0)).sum()
    FP = ((all_preds == 1) & (all_labels == 0)).sum()
    FN = ((all_preds == 0) & (all_labels == 1)).sum()
    
    sen  = float(TP) / (TP + FN + 1e-8)
    spec = float(TN) / (TN + FP + 1e-8)
    acc  = float(TP + TN) / len(all_labels)

    return {"sensitivity": sen, "specificity": spec, "accuracy": acc}

# ── RUN EXPERIMENTS ───────────────────────────────────────────────────────────
N = len(pre_train)
experiments = [
    ("Baseline", 0.0),
    ("+25%", 0.25),
    ("+50%", 0.50),
    ("+100%", 1.00),
    ("+200%", 2.00)
]

print("\n" + "="*70)
print("RUNNING DATA AUGMENTATION EXPERIMENTS (Fixed Test Set)")
print("="*70)

val_ds = CESPDataset(pre_test.tolist(), inter_test.tolist())
results_list = []

for name, multiplier in experiments:
    num_synthetic = int(N * multiplier)
    print(f"\nExperiment: {name} (Real Pre: {N}, Syn Pre: {num_synthetic}, Real Inter: {len(inter_train)})")
    
    if num_synthetic > len(syn_paths_accepted):
        print(f"  [WARNING] Not enough synthetic images ({len(syn_paths_accepted)} < {num_synthetic}). Using all available.")
        syn_subset = syn_paths_accepted.tolist()
    else:
        syn_subset = syn_paths_accepted[:num_synthetic].tolist()
        
    train_ds = CESPDataset(pre_train.tolist(), inter_train.tolist(), syn_subset)
    
    # Calculate pos_weight_scale matching the paper's original run (ratio of neg to pos)
    n_pos = len(pre_train) + len(syn_subset)
    n_neg = len(inter_train)
    pos_weight = float(n_neg) / max(n_pos, 1)
    
    print(f"  Training for 50 epochs...")
    metrics = train_eval_cesp(
        train_ds, val_ds, epochs=50, lr=1e-4, batch_size=64, 
        threshold=0.45, pos_weight_scale=pos_weight
    )
    
    print(f"  Result -> Sen: {metrics['sensitivity']:.4f} | Spec: {metrics['specificity']:.4f} | Acc: {metrics['accuracy']:.4f}")
    
    results_list.append({
        "Experiment": name,
        "Real Preictal": N,
        "Synthetic Preictal": len(syn_subset),
        "Real Interictal": len(inter_train),
        "Sensitivity": metrics['sensitivity'],
        "Specificity": metrics['specificity'],
        "Accuracy": metrics['accuracy']
    })
    
    # Save iteratively
    df = pd.DataFrame(results_list)
    save_path = os.path.join(RESULTS_DIR, "augmentation_experiments.csv")
    df.to_csv(save_path, index=False)

print("\n" + "="*70)
print("FINAL RESULTS")
print("="*70)
print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
print(f"\nResults saved to: {save_path}")
