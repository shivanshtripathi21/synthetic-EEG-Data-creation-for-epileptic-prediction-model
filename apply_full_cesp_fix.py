import json

NOTEBOOK_PATH = r"c:\Users\SHIVA\Downloads\Paper1_Reproduction-20260629T171224Z-3-001\Paper1_Full_Pipeline_FINAL.ipynb"

with open(NOTEBOOK_PATH, "r", encoding="utf-8") as f:
    nb = json.load(f)

cells = nb["cells"]

# Update Cell 41 (CESP Architecture - Fast & Accurate)
cell_41_source = [
    "# ============================================================\n",
    "# CELL 31: CESP Architecture (Rasheed et al. 2021)\n",
    "# 3 Conv blocks (126, 64, 64 filters) + FC (65536 -> 32 -> 1)\n",
    "# FIX: ReLU in hidden layer instead of Sigmoid (prevents vanishing gradient)\n",
    "# ============================================================\n",
    "class CESPBlock(nn.Module):\n",
    "    def __init__(self, in_ch, out_ch):\n",
    "        super().__init__()\n",
    "        self.block = nn.Sequential(\n",
    "            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),\n",
    "            nn.BatchNorm2d(out_ch),\n",
    "            nn.ReLU(inplace=True),\n",
    "            nn.MaxPool2d(2, 2)\n",
    "        )\n",
    "    def forward(self, x):\n",
    "        return self.block(x)\n",
    "\n",
    "class CESP(nn.Module):\n",
    "    def __init__(self):\n",
    "        super().__init__()\n",
    "        self.features = nn.Sequential(\n",
    "            CESPBlock(3, 126),   # 256->128\n",
    "            CESPBlock(126, 64),  # 128->64\n",
    "            CESPBlock(64, 64),   # 64->32\n",
    "        )\n",
    "        # 64 * 32 * 32 = 65,536\n",
    "        self.classifier = nn.Sequential(\n",
    "            nn.Flatten(),\n",
    "            nn.Linear(64 * 32 * 32, 32),\n",
    "            nn.ReLU(inplace=True),\n",
    "            nn.Dropout(0.3),\n",
    "            nn.Linear(32, 1)\n",
    "        )\n",
    "\n",
    "    def forward(self, x):\n",
    "        x = self.features(x)\n",
    "        return self.classifier(x).squeeze()\n",
    "\n",
    "cesp_test = CESP().to(device)\n",
    "with torch.no_grad():\n",
    "    x_test = torch.randn(2, 3, 256, 256, device=device)\n",
    "    out = cesp_test(x_test)\n",
    "print('CESP output shape:', out.shape)\n",
    "print('CESP total params:', f'{sum(p.numel() for p in cesp_test.parameters()):,}')\n",
    "del cesp_test\n"
]

# Update Cell 44 (train_cesp - High performance, fast training)
cell_44_source = [
    "# ============================================================\n",
    "# CELL 34: CESP Training - Stratified K-Fold Cross Validation\n",
    "# Optimized: GPU acceleration + Batch Size 64 + Fast convergence\n",
    "# ============================================================\n",
    "from sklearn.model_selection import StratifiedKFold\n",
    "import torch.nn.functional as F\n",
    "\n",
    "cesp_device = device  # Uses CUDA GPU if available\n",
    "\n",
    "def train_cesp(train_dataset, val_dataset, epochs=10, lr=1e-3, batch_size=64, threshold=0.5):\n",
    "    cesp_model = CESP().to(cesp_device)\n",
    "    optimizer = optim.Adam(cesp_model.parameters(), lr=lr, weight_decay=1e-4)\n",
    "    \n",
    "    train_labels = np.array(train_dataset.labels)\n",
    "    n_pos = (train_labels == 1.0).sum()\n",
    "    n_neg = (train_labels == 0.0).sum()\n",
    "    pos_weight = torch.tensor([n_neg / max(1, n_pos)]).to(cesp_device)\n",
    "    criterion_cesp = nn.BCEWithLogitsLoss(pos_weight=pos_weight)\n",
    "\n",
    "    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,\n",
    "                              num_workers=0, pin_memory=True if torch.cuda.is_available() else False)\n",
    "    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False,\n",
    "                              num_workers=0, pin_memory=True if torch.cuda.is_available() else False)\n",
    "\n",
    "    for epoch in range(epochs):\n",
    "        cesp_model.train()\n",
    "        running_loss = 0.0\n",
    "        for imgs, labels in train_loader:\n",
    "            imgs   = imgs.float().to(cesp_device)\n",
    "            labels = labels.float().to(cesp_device)\n",
    "            optimizer.zero_grad()\n",
    "            logits = cesp_model(imgs)\n",
    "            loss   = criterion_cesp(logits, labels)\n",
    "            loss.backward()\n",
    "            optimizer.step()\n",
    "            running_loss += loss.item()\n",
    "        if (epoch + 1) % 5 == 0:\n",
    "            avg = running_loss / max(len(train_loader), 1)\n",
    "            print(f'    Epoch {epoch+1:2d}/{epochs} | Loss: {avg:.4f}')\n",
    "\n",
    "    cesp_model.eval()\n",
    "    all_preds, all_lbl = [], []\n",
    "    with torch.no_grad():\n",
    "        for imgs, labels in val_loader:\n",
    "            imgs   = imgs.float().to(cesp_device)\n",
    "            logits = cesp_model(imgs)\n",
    "            probs  = torch.sigmoid(logits).cpu()\n",
    "            preds  = (probs >= threshold).float()\n",
    "            all_preds.extend(preds.numpy())\n",
    "            all_lbl.extend(labels.numpy())\n",
    "\n",
    "    all_preds = np.array(all_preds)\n",
    "    all_lbl   = np.array(all_lbl)\n",
    "\n",
    "    tp = ((all_preds == 1) & (all_lbl == 1)).sum()\n",
    "    fn = ((all_preds == 0) & (all_lbl == 1)).sum()\n",
    "    fp = ((all_preds == 1) & (all_lbl == 0)).sum()\n",
    "    tn = ((all_preds == 0) & (all_lbl == 0)).sum()\n",
    "\n",
    "    sensitivity = tp / (tp + fn + 1e-8)\n",
    "    specificity = tn / (tn + fp + 1e-8)\n",
    "    accuracy    = (tp + tn) / len(all_lbl)\n",
    "\n",
    "    return cesp_model, {'sensitivity': sensitivity,\n",
    "                        'specificity': specificity,\n",
    "                        'accuracy': accuracy}\n",
    "\n",
    "print('train_cesp() defined on GPU with batch_size=64, epochs=10. Device:', cesp_device)\n"
]

# Update Cell 45 (TRTR Experiment)
cell_45_source = [
    "# ============================================================\n",
    "# CELL 35: Run CESP — Experiment 1: TRTR (Train Real, Test Real)\n",
    "# ============================================================\n",
    "print('='*60)\n",
    "print('EXPERIMENT 1: TRTR — Train on Real, Test on Real')\n",
    "print('='*60)\n",
    "\n",
    "all_paths  = real_pre_paths + real_inter_paths\n",
    "all_labels = [1]*len(real_pre_paths) + [0]*len(real_inter_paths)\n",
    "all_paths  = np.array(all_paths)\n",
    "all_labels_arr = np.array(all_labels)\n",
    "\n",
    "skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)\n",
    "trtr_results = []\n",
    "\n",
    "for fold, (train_idx, val_idx) in enumerate(skf.split(all_paths, all_labels_arr)):\n",
    "    train_pre   = all_paths[train_idx][all_labels_arr[train_idx] == 1]\n",
    "    train_inter = all_paths[train_idx][all_labels_arr[train_idx] == 0]\n",
    "    val_pre     = all_paths[val_idx][all_labels_arr[val_idx] == 1]\n",
    "    val_inter   = all_paths[val_idx][all_labels_arr[val_idx] == 0]\n",
    "\n",
    "    train_ds = CESPDataset(train_pre, train_inter)\n",
    "    val_ds   = CESPDataset(val_pre, val_inter)\n",
    "\n",
    "    _, metrics = train_cesp(train_ds, val_ds, epochs=10, lr=1e-3, batch_size=64, threshold=0.4)\n",
    "    trtr_results.append(metrics)\n",
    "    print(f'  Fold {fold+1:2d} | Sen: {metrics[\"sensitivity\"]:.4f} | Spec: {metrics[\"specificity\"]:.4f} | Acc: {metrics[\"accuracy\"]:.4f}')\n",
    "\n",
    "trtr_sen  = np.mean([r[\"sensitivity\"] for r in trtr_results])\n",
    "trtr_spec = np.mean([r[\"specificity\"] for r in trtr_results])\n",
    "trtr_acc  = np.mean([r[\"accuracy\"]    for r in trtr_results])\n",
    "print(f'\\nTRTR Average | Sen: {trtr_sen:.4f} | Spec: {trtr_spec:.4f} | Acc: {trtr_acc:.4f}')\n",
    "print(f'Paper target | Sen: ~0.89 (89.02%)')\n"
]

cells[41]["source"] = cell_41_source
cells[41]["outputs"] = []
cells[41]["execution_count"] = None

cells[44]["source"] = cell_44_source
cells[44]["outputs"] = []
cells[44]["execution_count"] = None

cells[45]["source"] = cell_45_source
cells[45]["outputs"] = []
cells[45]["execution_count"] = None

with open(NOTEBOOK_PATH, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print("Fast GPU optimizations applied to notebook successfully!")
