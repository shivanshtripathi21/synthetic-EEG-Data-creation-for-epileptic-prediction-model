"""
Patch Paper1 notebook: Fix ALL 4 CESP experiments to match Paper 1 (Rasheed 2021).
Paper params: Adam lr=1e-4, 50 epochs, BCE loss, 10-fold CV for all experiments.
TSTR keeps anchor strategy.
"""
import json
from pathlib import Path

NB_PATH = Path("Paper1_Full_Pipeline_FINAL.ipynb")

with open(NB_PATH, "r", encoding="utf-8") as f:
    nb = json.load(f)

changes = []

# ============================================================
# 1. FIX Cell 44: train_cesp() function
#    - lr: 1e-3 -> 1e-4
#    - epochs: 10 -> 50
#    - Add ReduceLROnPlateau scheduler
#    - Keep BCEWithLogitsLoss with pos_weight
# ============================================================
TRAIN_CESP_NEW = [
    "# ============================================================\n",
    "# CELL 34: CESP Training - Paper 1 Parameters (Rasheed 2021)\n",
    "# Adam lr=1e-4, 50 epochs, BCE loss, ReduceLROnPlateau\n",
    "# ============================================================\n",
    "from sklearn.model_selection import StratifiedKFold\n",
    "import torch.nn.functional as F\n",
    "\n",
    "cesp_device = device  # Uses CUDA GPU if available\n",
    "\n",
    "def train_cesp(train_dataset, val_dataset, epochs=50, lr=1e-4, batch_size=64, threshold=0.5):\n",
    "    cesp_model = CESP().to(cesp_device)\n",
    "    optimizer = optim.Adam(cesp_model.parameters(), lr=lr, weight_decay=1e-4)\n",
    "    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=False)\n",
    "\n",
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
    "        avg_loss = running_loss / max(len(train_loader), 1)\n",
    "        scheduler.step(avg_loss)\n",
    "        if (epoch + 1) % 10 == 0:\n",
    "            print(f'    Epoch {epoch+1:2d}/{epochs} | Loss: {avg_loss:.4f}')\n",
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
    "print('train_cesp() defined -- Paper 1 params: Adam lr=1e-4, 50 epochs, BCE+pos_weight, ReduceLROnPlateau')\n",
    "print('Device:', cesp_device)\n",
]

for i, cell in enumerate(nb["cells"]):
    if cell["cell_type"] != "code":
        continue
    src = "".join(cell["source"])
    if "def train_cesp" in src and "CELL 34" in src:
        cell["source"] = TRAIN_CESP_NEW
        changes.append(f"Cell {i}: train_cesp() -> lr=1e-4, epochs=50, +scheduler")
        break

# ============================================================
# 2. FIX Cell 46: TRTR experiment
#    - epochs=50, lr=1e-4, remove custom threshold
# ============================================================
TRTR_NEW = [
    "# ============================================================\n",
    "# CELL 35: Run CESP -- Experiment 1: TRTR (Train Real, Test Real)\n",
    "# Paper 1 params: 50 epochs, lr=1e-4, 10-fold Stratified KFold\n",
    "# ============================================================\n",
    "print('='*60)\n",
    "print('EXPERIMENT 1: TRTR -- Train on Real, Test on Real')\n",
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
    "    _, metrics = train_cesp(train_ds, val_ds, epochs=50, lr=1e-4, batch_size=64)\n",
    "    trtr_results.append(metrics)\n",
    "    print(f'  Fold {fold+1:2d} | Sen: {metrics[\"sensitivity\"]:.4f} | Spec: {metrics[\"specificity\"]:.4f} | Acc: {metrics[\"accuracy\"]:.4f}')\n",
    "\n",
    "trtr_sen  = np.mean([r['sensitivity'] for r in trtr_results])\n",
    "trtr_spec = np.mean([r['specificity'] for r in trtr_results])\n",
    "trtr_acc  = np.mean([r['accuracy']    for r in trtr_results])\n",
    "print(f'\\nTRTR Average | Sen: {trtr_sen:.4f} | Spec: {trtr_spec:.4f} | Acc: {trtr_acc:.4f}')\n",
    "print(f'Paper target | Sen: ~0.89 (89.02%)')\n",
]

for i, cell in enumerate(nb["cells"]):
    if cell["cell_type"] != "code":
        continue
    src = "".join(cell["source"])
    if "TRTR" in src and "EXPERIMENT 1" in src and "train_cesp" in src:
        cell["source"] = TRTR_NEW
        changes.append(f"Cell {i}: TRTR -> epochs=50, lr=1e-4, threshold=0.5")
        break

# ============================================================
# 3. FIX Cell 47: TSTR experiment (keep anchor strategy)
#    - lr: 2e-4 -> 1e-4
# ============================================================
TSTR_NEW = [
    "# ============================================================\n",
    "# CELL 36: CESP - Experiment 2: TSTR (Train Synthetic, Test Real)\n",
    "# Paper 1 params: 50 epochs, lr=1e-4, 10-fold CV\n",
    "# Uses ANCHOR STRATEGY: real preictal + synthetic preictal + real interictal\n",
    "# Paper target: Sen~0.8821, FPR/h~0.14\n",
    "# ============================================================\n",
    "print('='*60)\n",
    "print('EXPERIMENT 2: TSTR -- Train on Synthetic, Test on Real')\n",
    "print('='*60)\n",
    "\n",
    "import os\n",
    "from sklearn.model_selection import StratifiedKFold\n",
    "\n",
    "TSTR_MODEL_PATH = os.path.join(SAVE_MODEL, 'cesp_tstr.pth')\n",
    "\n",
    "# -- Data preparation --\n",
    "real_pre_arr   = np.array(real_pre_paths)\n",
    "real_inter_arr = np.array(real_inter_paths)\n",
    "syn_arr        = np.array(syn_paths_accepted)\n",
    "\n",
    "print(f'Real preictal   : {len(real_pre_arr)}')\n",
    "print(f'Real interictal : {len(real_inter_arr)}')\n",
    "print(f'Synthetic       : {len(syn_arr)}')\n",
    "\n",
    "# -- TSTR training function (10-fold, anchor strategy) --\n",
    "def train_cesp_tstr(real_pre_paths, real_inter_paths, syn_paths,\n",
    "                    epochs=50, lr=1e-4, batch_size=64, n_folds=10):\n",
    '    """\n',
    "    TSTR setup per fold (ANCHOR STRATEGY):\n",
    "      Train  -> real preictal anchor (label=1) + synthetic preictal (label=1)\n",
    "                + real interictal train split (label=0)\n",
    "      Test   -> real preictal test split (label=1)\n",
    "                + real interictal test split (label=0)\n",
    '    """\n',
    "    all_real   = np.concatenate([real_pre_paths, real_inter_paths])\n",
    "    all_labels = np.array([1]*len(real_pre_paths) + [0]*len(real_inter_paths))\n",
    "\n",
    "    skf     = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)\n",
    "    results = []\n",
    "\n",
    "    for fold, (train_idx, val_idx) in enumerate(skf.split(all_real, all_labels)):\n",
    "        val_paths  = all_real[val_idx]\n",
    "        val_labels = all_labels[val_idx]\n",
    "\n",
    "        val_pre   = val_paths[val_labels == 1]\n",
    "        val_inter = val_paths[val_labels == 0]\n",
    "        # Take matching real data from train split\n",
    "        tr_inter  = all_real[train_idx][all_labels[train_idx] == 0]\n",
    "        tr_pre    = all_real[train_idx][all_labels[train_idx] == 1]  # Anchor strategy\n",
    "\n",
    "        # Balance: use as many synthetic as real interictal in train\n",
    "        n_inter = len(tr_inter)\n",
    "        syn_subset = syn_paths[:n_inter] if len(syn_paths) >= n_inter else syn_paths\n",
    "\n",
    "        train_ds = CESPDataset(tr_pre.tolist(), tr_inter.tolist(), syn_subset.tolist())\n",
    "        val_ds   = CESPDataset(val_pre.tolist(), val_inter.tolist())\n",
    "\n",
    "        _, metrics = train_cesp(train_ds, val_ds, epochs=epochs, lr=lr, batch_size=batch_size)\n",
    "        results.append(metrics)\n",
    "        print(f'  Fold {fold+1:2d} | Sen: {metrics[\"sensitivity\"]:.4f} | '\n",
    "              f'Spec: {metrics[\"specificity\"]:.4f} | Acc: {metrics[\"accuracy\"]:.4f}')\n",
    "\n",
    "    return results\n",
    "\n",
    "# -- Run 10-fold TSTR --\n",
    "print('[INFO] Running 10-fold TSTR training (anchor strategy)...')\n",
    "\n",
    "tstr_results = train_cesp_tstr(\n",
    "    real_pre_arr, real_inter_arr, syn_arr,\n",
    "    epochs=50, lr=1e-4, batch_size=64, n_folds=10\n",
    ")\n",
    "\n",
    "tstr_sen  = np.mean([r['sensitivity'] for r in tstr_results])\n",
    "tstr_spec = np.mean([r['specificity'] for r in tstr_results])\n",
    "tstr_acc  = np.mean([r['accuracy']    for r in tstr_results])\n",
    "\n",
    "print(f'\\nTSTR Average | Sen: {tstr_sen:.4f} | Spec: {tstr_spec:.4f} | Acc: {tstr_acc:.4f}')\n",
    "print(f'Paper target | Sen: ~0.8821 (88.21%), FPR/h ~0.14')\n",
    "\n",
    "# -- Save best TSTR model (retrain once on all data) --\n",
    "print('\\n[INFO] Training final TSTR model on all data for saving...')\n",
    "full_train_ds = CESPDataset([], real_inter_arr.tolist(), syn_arr.tolist())\n",
    "full_test_ds  = CESPDataset(real_pre_arr.tolist(), real_inter_arr.tolist())\n",
    "tstr_model, _ = train_cesp(full_train_ds, full_test_ds, epochs=50, lr=1e-4, batch_size=64)\n",
    "torch.save(tstr_model.state_dict(), TSTR_MODEL_PATH)\n",
    "print(f'CESP (TSTR) saved to {TSTR_MODEL_PATH}')\n",
]

for i, cell in enumerate(nb["cells"]):
    if cell["cell_type"] != "code":
        continue
    src = "".join(cell["source"])
    if "TSTR" in src and "train_cesp_tstr" in src and "EXPERIMENT 2" in src:
        cell["source"] = TSTR_NEW
        changes.append(f"Cell {i}: TSTR -> lr=1e-4, batch=64, anchor strategy kept")
        break

# ============================================================
# 4. FIX Cell 48: TRTS & TSTS experiments
#    - Add 10-fold CV (like Exp 1 & 2)
#    - lr=1e-4, epochs=50
# ============================================================
TRTS_TSTS_NEW = [
    "# ============================================================\n",
    "# CELL 37: CESP -- Experiment 3 & 4: TRTS, TSTS\n",
    "# Paper 1 params: 50 epochs, lr=1e-4, 10-fold Stratified KFold\n",
    "# ============================================================\n",
    "import json as json_lib\n",
    "from pathlib import Path\n",
    "from sklearn.model_selection import StratifiedKFold\n",
    "\n",
    "# ---- Re-load data paths from disk ----\n",
    "real_pre_arr   = np.array(list(Path(PROJECT_ROOT, '02_preprocessed', 'preictal').rglob('*.png')))\n",
    "real_inter_arr = np.array(list(Path(PROJECT_ROOT, '02_preprocessed', 'interictal').rglob('*.png')))\n",
    "\n",
    "with open(os.path.join(PROJECT_ROOT, '05_results/tables/accepted_synthetic.json')) as _f:\n",
    "    syn_arr = np.array([Path(p) for p in json_lib.load(_f)])\n",
    "\n",
    "print(f'Loaded: real_pre={len(real_pre_arr)}, real_inter={len(real_inter_arr)}, syn={len(syn_arr)}')\n",
    "\n",
    "# ============================================================\n",
    "# EXPERIMENT 3: TRTS -- Train on Real, Test on Synthetic\n",
    "# 10-fold CV: train on real preictal+interictal, test on synthetic preictal+real interictal\n",
    "# ============================================================\n",
    "print('='*60)\n",
    "print('EXPERIMENT 3: TRTS -- Train on Real, Test on Synthetic')\n",
    "print('='*60)\n",
    "\n",
    "all_real   = np.concatenate([real_pre_arr, real_inter_arr])\n",
    "all_labels = np.array([1]*len(real_pre_arr) + [0]*len(real_inter_arr))\n",
    "\n",
    "skf_trts = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)\n",
    "trts_results = []\n",
    "\n",
    "for fold, (train_idx, val_idx) in enumerate(skf_trts.split(all_real, all_labels)):\n",
    "    # Train: real preictal + real interictal (from train split)\n",
    "    tr_pre   = all_real[train_idx][all_labels[train_idx] == 1]\n",
    "    tr_inter = all_real[train_idx][all_labels[train_idx] == 0]\n",
    "    # Test: synthetic preictal + real interictal (from val split)\n",
    "    val_inter = all_real[val_idx][all_labels[val_idx] == 0]\n",
    "    n_val_pre = (all_labels[val_idx] == 1).sum()\n",
    "    syn_subset = syn_arr[:n_val_pre] if len(syn_arr) >= n_val_pre else syn_arr\n",
    "\n",
    "    train_ds = CESPDataset(tr_pre.tolist(), tr_inter.tolist())\n",
    "    test_ds  = CESPDataset(syn_subset.tolist(), val_inter.tolist())\n",
    "\n",
    "    _, metrics = train_cesp(train_ds, test_ds, epochs=50, lr=1e-4, batch_size=64)\n",
    "    trts_results.append(metrics)\n",
    "    print(f'  Fold {fold+1:2d} | Sen: {metrics[\"sensitivity\"]:.4f} | Spec: {metrics[\"specificity\"]:.4f} | Acc: {metrics[\"accuracy\"]:.4f}')\n",
    "\n",
    "trts_sen  = np.mean([r['sensitivity'] for r in trts_results])\n",
    "trts_spec = np.mean([r['specificity'] for r in trts_results])\n",
    "trts_acc  = np.mean([r['accuracy']    for r in trts_results])\n",
    "print(f'\\nTRTS Average | Sen: {trts_sen:.4f} | Spec: {trts_spec:.4f} | Acc: {trts_acc:.4f}')\n",
    "\n",
    "# ============================================================\n",
    "# EXPERIMENT 4: TSTS -- Train on Synthetic, Test on Synthetic\n",
    "# 10-fold CV: train on syn preictal+real interictal, test on syn preictal+real interictal\n",
    "# ============================================================\n",
    "print('\\n' + '='*60)\n",
    "print('EXPERIMENT 4: TSTS -- Train Synthetic, Test Synthetic')\n",
    "print('='*60)\n",
    "\n",
    "# Split synthetic into train/test per fold using interictal folds\n",
    "skf_tsts = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)\n",
    "tsts_results = []\n",
    "\n",
    "# Use all_real splits just to partition interictal consistently\n",
    "for fold, (train_idx, val_idx) in enumerate(skf_tsts.split(all_real, all_labels)):\n",
    "    tr_inter = all_real[train_idx][all_labels[train_idx] == 0]\n",
    "    val_inter = all_real[val_idx][all_labels[val_idx] == 0]\n",
    "\n",
    "    # Split synthetic 50/50 for train/test\n",
    "    syn_half = len(syn_arr) // 2\n",
    "    syn_train = syn_arr[:syn_half]\n",
    "    syn_test  = syn_arr[syn_half:]\n",
    "\n",
    "    train_ds = CESPDataset(syn_train.tolist(), tr_inter.tolist())\n",
    "    test_ds  = CESPDataset(syn_test.tolist(),  val_inter.tolist())\n",
    "\n",
    "    _, metrics = train_cesp(train_ds, test_ds, epochs=50, lr=1e-4, batch_size=64)\n",
    "    tsts_results.append(metrics)\n",
    "    print(f'  Fold {fold+1:2d} | Sen: {metrics[\"sensitivity\"]:.4f} | Spec: {metrics[\"specificity\"]:.4f} | Acc: {metrics[\"accuracy\"]:.4f}')\n",
    "\n",
    "tsts_sen  = np.mean([r['sensitivity'] for r in tsts_results])\n",
    "tsts_spec = np.mean([r['specificity'] for r in tsts_results])\n",
    "tsts_acc  = np.mean([r['accuracy']    for r in tsts_results])\n",
    "print(f'\\nTSTS Average | Sen: {tsts_sen:.4f} | Spec: {tsts_spec:.4f} | Acc: {tsts_acc:.4f}')\n",
    "\n",
    "# -- Save metrics to JSON --\n",
    "import json as _json\n",
    "_save_base = os.path.join(PROJECT_ROOT, '05_results/tables')\n",
    "with open(os.path.join(_save_base, 'trts_metrics.json'), 'w') as _f:\n",
    "    _json.dump({k: float(v) for k, v in {**{'sensitivity': trts_sen, 'specificity': trts_spec, 'accuracy': trts_acc}}.items()}, _f, indent=2)\n",
    "with open(os.path.join(_save_base, 'tsts_metrics.json'), 'w') as _f:\n",
    "    _json.dump({k: float(v) for k, v in {**{'sensitivity': tsts_sen, 'specificity': tsts_spec, 'accuracy': tsts_acc}}.items()}, _f, indent=2)\n",
    "print('\\nTRTS & TSTS metrics saved to JSON.')\n",
]

for i, cell in enumerate(nb["cells"]):
    if cell["cell_type"] != "code":
        continue
    src = "".join(cell["source"])
    if "TRTS" in src and "TSTS" in src and "Experiment 3" in src:
        cell["source"] = TRTS_TSTS_NEW
        changes.append(f"Cell {i}: TRTS+TSTS -> 10-fold CV, lr=1e-4, epochs=50")
        break

# ============================================================
# 5. Also fix CESP Linear layer (in case it got reverted again)
# ============================================================
for i, cell in enumerate(nb["cells"]):
    if cell["cell_type"] != "code":
        continue
    src = "".join(cell["source"])
    if "class CESP(nn.Module)" in src and "64 * 32 * 32" in src:
        new_source = []
        for line in cell["source"]:
            line = line.replace("64 * 32 * 32 = 65,536", "64 * 16 * 16 = 16,384  (128x128 input, 3x MaxPool -> 16x16)")
            line = line.replace("64 * 32 * 32", "64 * 16 * 16")
            new_source.append(line)
        cell["source"] = new_source
        changes.append(f"Cell {i}: CESP Linear 64*32*32 -> 64*16*16")

# Fix test tensor too
for i, cell in enumerate(nb["cells"]):
    if cell["cell_type"] != "code":
        continue
    src = "".join(cell["source"])
    if "cesp_test = CESP()" in src and "256, 256" in src:
        new_source = []
        for line in cell["source"]:
            line = line.replace("3, 256, 256", "3, 128, 128")
            new_source.append(line)
        cell["source"] = new_source
        changes.append(f"Cell {i}: test tensor 256x256 -> 128x128")

# ============================================================
# SAVE
# ============================================================
if changes:
    with open(NB_PATH, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)
    print("PATCHED SUCCESSFULLY!")
    for c in changes:
        print(f"  {c}")
    print(f"\nTotal changes: {len(changes)}")
    print("\nIMPORTANT: Close and reopen notebook in Jupyter, then restart kernel and run all cells.")
else:
    print("WARNING: No changes made - cells not found!")
