import os
import json
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.manifold import TSNE
from scipy.linalg import sqrtm
import matplotlib.pyplot as plt
from tqdm import tqdm

# Paths
PROJECT_ROOT = r"C:\Users\SHIVA\Downloads\Paper1_Reproduction-20260629T171224Z-3-001"
PREPROCESSED_DIR = os.path.join(PROJECT_ROOT, "Paper1_Reproduction", "02_preprocessed")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "Paper1_Reproduction", "05_results")
os.makedirs(RESULTS_DIR, exist_ok=True)

REAL_PRE_DIR = os.path.join(PREPROCESSED_DIR, "preictal")
SYN_PRE_DIR = os.path.join(PREPROCESSED_DIR, "synthetic_preictal")
ACCEPTED_JSON = os.path.join(RESULTS_DIR, "tables", "accepted_synthetic.json")

# Load file paths
real_pre_paths = [str(p) for p in Path(REAL_PRE_DIR).rglob("*.png") if p.is_file()]
with open(ACCEPTED_JSON, "r") as f:
    syn_paths_accepted = json.load(f)

print(f"Loaded Real Preictal: {len(real_pre_paths)}")
print(f"Loaded Synthetic Preictal: {len(syn_paths_accepted)}")

# Dataset Definition
class SpectrogramDataset(Dataset):
    def __init__(self, paths):
        self.paths = paths
        # InceptionV3 requires 299x299 inputs and normalization
        self.transform = transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img_path = self.paths[idx]
        img = Image.open(img_path).convert("RGB")
        return self.transform(img)

# Feature Extraction Function
def extract_features(model, dataloader, device):
    features = []
    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting"):
            batch = batch.to(device)
            out = model(batch)
            features.append(out.cpu().numpy())
    return np.vstack(features)

# MMD Calculation
def calculate_mmd(X, Y, gamma=None):
    # Using RBF Kernel MMD
    XX = rbf_kernel(X, X, gamma)
    YY = rbf_kernel(Y, Y, gamma)
    XY = rbf_kernel(X, Y, gamma)
    return XX.mean() + YY.mean() - 2 * XY.mean()

# FID Calculation
def calculate_fid(X, Y):
    mu1, sigma1 = X.mean(axis=0), np.cov(X, rowvar=False)
    mu2, sigma2 = Y.mean(axis=0), np.cov(Y, rowvar=False)
    
    ssdiff = np.sum((mu1 - mu2)**2.0)
    covmean = sqrtm(sigma1.dot(sigma2))
    
    if np.iscomplexobj(covmean):
        covmean = covmean.real
        
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load InceptionV3
    print("Loading pretrained InceptionV3...")
    model = models.inception_v3(pretrained=True)
    # Remove the final fully connected layer to get 2048-D features
    model.fc = torch.nn.Identity()
    model = model.to(device)
    
    # Freeze weights
    for param in model.parameters():
        param.requires_grad = False

    # 2. Extract Features
    real_ds = SpectrogramDataset(real_pre_paths)
    syn_ds = SpectrogramDataset(syn_paths_accepted)
    
    real_loader = DataLoader(real_ds, batch_size=32, shuffle=False, num_workers=0)
    syn_loader = DataLoader(syn_ds, batch_size=32, shuffle=False, num_workers=0)

    print("Extracting Real Features...")
    real_features = extract_features(model, real_loader, device)
    print("Extracting Synthetic Features...")
    syn_features = extract_features(model, syn_loader, device)

    # Subsample synthetic features if there are too many (for faster MMD/tSNE)
    # 4715 vs 1690. MMD is O(N^2), so we'll match sample sizes for fair comparison.
    n_samples = min(len(real_features), len(syn_features), 1500)
    np.random.seed(42)
    idx_real = np.random.choice(len(real_features), n_samples, replace=False)
    idx_syn = np.random.choice(len(syn_features), n_samples, replace=False)
    
    real_sub = real_features[idx_real]
    syn_sub = syn_features[idx_syn]

    # 3. Calculate Metrics
    print("Calculating MMD...")
    # Gamma default in sklearn is 1/(n_features * X.var()), which works well
    mmd_score = calculate_mmd(real_sub, syn_sub)
    print(f"MMD Score: {mmd_score:.4f}")

    print("Calculating FID...")
    fid_score = calculate_fid(real_features, syn_features) # FID uses full set since it calculates cov matrix
    print(f"FID Score: {fid_score:.4f}")

    # 4. t-SNE Visualization
    print("Running t-SNE...")
    X_combined = np.vstack([real_sub, syn_sub])
    labels = np.array(["Real"] * len(real_sub) + ["Synthetic"] * len(syn_sub))
    
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    X_2d = tsne.fit_transform(X_combined)

    plt.figure(figsize=(10, 8))
    plt.scatter(X_2d[labels == "Synthetic", 0], X_2d[labels == "Synthetic", 1], 
                c='orange', label='Synthetic Preictal', alpha=0.5, s=20)
    plt.scatter(X_2d[labels == "Real", 0], X_2d[labels == "Real", 1], 
                c='blue', label='Real Preictal', alpha=0.5, s=20)
    
    plt.title(f"t-SNE of InceptionV3 Features\nFID: {fid_score:.2f} | MMD: {mmd_score:.4f}")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    save_path = os.path.join(RESULTS_DIR, "tsne_inception_features.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved t-SNE plot to {save_path}")

    # Save metrics to JSON
    metrics_path = os.path.join(RESULTS_DIR, "tables", "inception_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump({"FID": float(fid_score), "MMD": float(mmd_score)}, f, indent=2)

if __name__ == "__main__":
    main()
