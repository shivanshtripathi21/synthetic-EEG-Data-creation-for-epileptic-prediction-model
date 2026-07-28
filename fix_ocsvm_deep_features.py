"""
fix_ocsvm_deep_features.py
Replaces OC-SVM cells (28-30) to use deep features from the trained
VAE-GAN Discriminator instead of flattened pixel features.
"""
import json, sys

with open('Paper1_Full_Pipeline_FINAL.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

cells = nb['cells']

# ============================================================
# CELL INDEX 37 → Cell 28: Deep Feature Extraction using Discriminator
# ============================================================
cells[37]["source"] = [
    "# ============================================================\n",
    "# CELL 28: Deep Feature Extraction using VAE-GAN Discriminator\n",
    "# Paper: Train OC-SVM on real preictal features, test on synthetic\n",
    "# FIX: Uses Discriminator's deep features (layer4: 512-dim)\n",
    "#      instead of flattened pixels for much better quality filtering\n",
    "# ============================================================\n",
    "\n",
    "# Load the best trained Discriminator\n",
    "disc_feat = Discriminator().to(device)\n",
    "disc_feat.load_state_dict(torch.load(\n",
    "    os.path.join(SAVE_MODEL, \"discriminator_best.pth\"), map_location=device\n",
    "))\n",
    "disc_feat.eval()\n",
    "print(\"Trained Discriminator loaded for feature extraction.\")\n",
    "\n",
    "# Transform for feature extraction (same as training)\n",
    "feat_transform = transforms.Compose([\n",
    "    transforms.ToPILImage(),\n",
    "    transforms.Resize((256, 256)),\n",
    "    transforms.ToTensor(),\n",
    "    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))\n",
    "])\n",
    "\n",
    "def extract_deep_features(image_paths, batch_size=64):\n",
    "    \"\"\"\n",
    "    Extract 512-dim deep features from Discriminator's layer4.\n",
    "    Much more meaningful than flattened pixels for OC-SVM.\n",
    "    \"\"\"\n",
    "    all_features = []\n",
    "    paths = list(image_paths)\n",
    "    \n",
    "    for i in tqdm(range(0, len(paths), batch_size), desc=\"Extracting deep features\"):\n",
    "        batch_paths = paths[i:i+batch_size]\n",
    "        batch_imgs = []\n",
    "        \n",
    "        for p in batch_paths:\n",
    "            img = cv2.imread(str(p))\n",
    "            if img is None:\n",
    "                continue\n",
    "            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)\n",
    "            img_tensor = feat_transform(img)\n",
    "            batch_imgs.append(img_tensor)\n",
    "        \n",
    "        if not batch_imgs:\n",
    "            continue\n",
    "        \n",
    "        batch_tensor = torch.stack(batch_imgs).to(device)\n",
    "        \n",
    "        with torch.no_grad():\n",
    "            # Get layer4 features (512-dim after global avg pool)\n",
    "            _, feat_maps = disc_feat(batch_tensor, return_features=True)\n",
    "            # Global average pool: [B, 512, 16, 16] → [B, 512]\n",
    "            features = torch.nn.functional.adaptive_avg_pool2d(feat_maps, 1)\n",
    "            features = features.view(features.size(0), -1)  # [B, 512]\n",
    "            all_features.append(features.cpu().numpy())\n",
    "    \n",
    "    return np.concatenate(all_features, axis=0)\n",
    "\n",
    "# Extract deep features from real preictal images\n",
    "real_paths = list(Path(PROJECT_ROOT, \"02_preprocessed\", \"preictal\").rglob(\"*.png\"))\n",
    "print(f\"Real preictal images: {len(real_paths)}\")\n",
    "\n",
    "X_real = extract_deep_features(real_paths)\n",
    "print(f\"Deep feature matrix: {X_real.shape}\")  # Should be (N, 512)\n",
]
cells[37]["outputs"] = []
cells[37]["execution_count"] = None

# ============================================================
# CELL INDEX 38 → Cell 29: Train OC-SVM on Deep Features
# ============================================================
cells[38]["source"] = [
    "# ============================================================\n",
    "# CELL 29: Train One-Class SVM on Deep Features\n",
    "# Paper: RBF kernel, trained on real preictal deep features\n",
    "# ============================================================\n",
    "scaler = StandardScaler()\n",
    "X_real_scaled = scaler.fit_transform(X_real)\n",
    "\n",
    "ocsvm = OneClassSVM(kernel=\"rbf\", nu=0.1, gamma=\"scale\")\n",
    "ocsvm.fit(X_real_scaled)\n",
    "print(\"One-Class SVM trained on real preictal deep features (512-dim).\")\n",
    "\n",
    "# Self-test on real data\n",
    "real_preds = ocsvm.predict(X_real_scaled)\n",
    "real_acceptance = (real_preds == 1).mean()\n",
    "print(f\"Real data acceptance rate: {real_acceptance:.3f} (should be ~0.90)\")\n",
]
cells[38]["outputs"] = []
cells[38]["execution_count"] = None

# ============================================================
# CELL INDEX 39 → Cell 30: Test OC-SVM on Synthetic Deep Features
# ============================================================
cells[39]["source"] = [
    "# ============================================================\n",
    "# CELL 30: Test One-Class SVM on Synthetic Data (Deep Features)\n",
    "# Paper: Accept synthetic images that OC-SVM classifies as inliers\n",
    "# ============================================================\n",
    "import json as json_lib\n",
    "\n",
    "syn_paths = sorted(Path(PROJECT_ROOT, '04_generated').glob('*.png'))\n",
    "print(f'Synthetic images: {len(syn_paths)}')\n",
    "\n",
    "# Extract deep features from synthetic images\n",
    "print('Extracting deep features from synthetic images...')\n",
    "X_syn = extract_deep_features(syn_paths)\n",
    "X_syn_scaled = scaler.transform(X_syn)\n",
    "\n",
    "# OC-SVM prediction\n",
    "syn_scores = ocsvm.decision_function(X_syn_scaled)\n",
    "syn_preds  = ocsvm.predict(X_syn_scaled)\n",
    "syn_acceptance = (syn_preds == 1).mean()\n",
    "print(f'OC-SVM acceptance rate: {syn_acceptance:.3f}')\n",
    "\n",
    "# Use strict SVM predictions if acceptance > 10%, else fallback to top-50%\n",
    "if syn_acceptance >= 0.10:\n",
    "    accepted_idx = np.where(syn_preds == 1)[0]\n",
    "    print(f'Using strict OC-SVM acceptance ({syn_acceptance:.1%})')\n",
    "else:\n",
    "    print(f'Low acceptance ({syn_acceptance:.1%}) — using top-50% by decision score')\n",
    "    threshold = np.percentile(syn_scores, 50)\n",
    "    accepted_idx = np.where(syn_scores >= threshold)[0]\n",
    "\n",
    "accepted_paths = [syn_paths[i] for i in accepted_idx]\n",
    "print(f'Accepted synthetic images: {len(accepted_paths)} / {len(syn_paths)}')\n",
    "print(f'Acceptance rate: {len(accepted_paths)/len(syn_paths):.1%}')\n",
    "\n",
    "# Save accepted paths list\n",
    "accepted_list = [str(p) for p in accepted_paths]\n",
    "with open(os.path.join(PROJECT_ROOT, '05_results/tables/accepted_synthetic.json'), 'w') as f:\n",
    "    json_lib.dump(accepted_list, f)\n",
    "print(f'Accepted list saved ({len(accepted_list)} images).')\n",
    "\n",
    "# Cleanup discriminator from GPU memory\n",
    "del disc_feat\n",
    "torch.cuda.empty_cache() if torch.cuda.is_available() else None\n",
    "print('Discriminator removed from memory.')\n",
]
cells[39]["outputs"] = []
cells[39]["execution_count"] = None

# Save
with open('Paper1_Full_Pipeline_FINAL.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("[DONE] OC-SVM cells updated to use deep features from Discriminator!")
print("  Cell 28: Deep feature extraction (512-dim from Disc layer4)")
print("  Cell 29: OC-SVM training on deep features")
print("  Cell 30: Synthetic testing with deep features + smart fallback")
