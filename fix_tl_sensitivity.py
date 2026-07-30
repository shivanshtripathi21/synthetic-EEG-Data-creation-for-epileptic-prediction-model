"""
fix_tl_sensitivity.py
======================
Fixes all 4 Transfer Learning cells in the notebook:
  1. build_tl_model()     → Freeze backbone, only train head
  2. train_tl_model()     → BCEWithLogitsLoss + pos_weight + 2-phase training
  3. evaluate_tl_model()  → threshold=0.40 for better sensitivity
  4. Delete old .pth files → Force fresh retrain with new settings
"""

import json, os, glob

NB_PATH = r'c:\Users\SHIVA\Downloads\Paper1_Reproduction-20260629T171224Z-3-001\Paper1_Full_Pipeline_FINAL.ipynb'
SAVE_MODEL_DIR = r'c:\Users\SHIVA\Downloads\Paper1_Reproduction-20260629T171224Z-3-001\Paper1_Reproduction\06_models'

# ── Step 1: Delete old saved TL .pth files so fresh retrain happens ──────────
deleted = []
for model_name in ['vgg16', 'vgg19', 'resnet50', 'inceptionv3']:
    pth_path = os.path.join(SAVE_MODEL_DIR, f'tl_{model_name}.pth')
    if os.path.exists(pth_path):
        os.remove(pth_path)
        deleted.append(pth_path)
        print(f"[DELETE] {pth_path}")

if not deleted:
    print("[INFO] No old TL .pth files found (will train fresh anyway).")

# ── Step 2: Load notebook ─────────────────────────────────────────────────────
with open(NB_PATH, 'r', encoding='utf-8') as f:
    nb = json.load(f)

cells = nb['cells']

# ── Step 3: Fix Cell 50 — build_tl_model() with frozen backbone ─────────────
cell_50_source = [
    "# ============================================================\n",
    "# CELL 38: Transfer Learning Setup\n",
    "# Paper: Pre-train on augmented data -> fine-tune patient-specific\n",
    "# FIX: Backbone frozen → only head trained (prevents overfit)\n",
    "# FIX: Final layer outputs raw logits (no Sigmoid) for BCEWithLogitsLoss\n",
    "# ============================================================\n",
    "def build_tl_model(model_name, num_classes=1, freeze_backbone=True):\n",
    "    \"\"\"Build TL model. Backbone frozen by default for stable head training.\"\"\"\n",
    "    if model_name == 'vgg16':\n",
    "        model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)\n",
    "        if freeze_backbone:\n",
    "            for param in model.features.parameters():\n",
    "                param.requires_grad = False\n",
    "        model.classifier[-1] = nn.Linear(4096, num_classes)  # raw logits\n",
    "\n",
    "    elif model_name == 'vgg19':\n",
    "        model = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)\n",
    "        if freeze_backbone:\n",
    "            for param in model.features.parameters():\n",
    "                param.requires_grad = False\n",
    "        model.classifier[-1] = nn.Linear(4096, num_classes)\n",
    "\n",
    "    elif model_name == 'resnet50':\n",
    "        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)\n",
    "        if freeze_backbone:\n",
    "            for name, param in model.named_parameters():\n",
    "                if 'fc' not in name:\n",
    "                    param.requires_grad = False\n",
    "        model.fc = nn.Linear(2048, num_classes)\n",
    "\n",
    "    elif model_name == 'inceptionv3':\n",
    "        model = models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1)\n",
    "        if freeze_backbone:\n",
    "            for name, param in model.named_parameters():\n",
    "                if 'fc' not in name and 'AuxLogits.fc' not in name:\n",
    "                    param.requires_grad = False\n",
    "        model.fc = nn.Linear(2048, num_classes)\n",
    "        model.AuxLogits.fc = nn.Linear(768, num_classes)\n",
    "\n",
    "    return model.to(device)\n",
    "\n",
    "print('Transfer learning build_tl_model() ready (backbone frozen by default).')\n",
    "print('Models: VGG16, VGG19, ResNet50, InceptionV3')\n",
]

cells[50]['source'] = cell_50_source
cells[50]['outputs'] = []
cells[50]['execution_count'] = None
print("[FIXED] Cell 50 — build_tl_model() with frozen backbone + raw logits")

# ── Step 4: Fix Cell 52 — train_tl_model() + evaluate_tl_model() ─────────────
cell_52_source = [
    "# ============================================================\n",
    "# CELL 40: Train & Evaluate All 4 TL Models\n",
    "# FIX 1: threshold=0.40 for higher sensitivity\n",
    "# FIX 2: BCEWithLogitsLoss + pos_weight for class imbalance\n",
    "# FIX 3: 2-phase training (frozen head first, then full fine-tune)\n",
    "# FIX 4: Force retrain (old .pth deleted by fix script)\n",
    "# ============================================================\n",
    "import json as json_lib\n",
    "from pathlib import Path\n",
    "from sklearn.model_selection import train_test_split\n",
    "\n",
    "# ---- Re-load data paths from disk (self-contained) ----\n",
    "if 'pre_train' not in dir() or pre_train is None:\n",
    "    _real_pre_arr   = np.array(list(Path(PROJECT_ROOT, '02_preprocessed', 'preictal').rglob('*.png')))\n",
    "    _real_inter_arr = np.array(list(Path(PROJECT_ROOT, '02_preprocessed', 'interictal').rglob('*.png')))\n",
    "    with open(os.path.join(PROJECT_ROOT, '05_results/tables/accepted_synthetic.json')) as _f:\n",
    "        _syn_arr = np.array([Path(p) for p in json_lib.load(_f)])\n",
    "    pre_train,   pre_test   = train_test_split(_real_pre_arr,   test_size=0.2, random_state=42)\n",
    "    inter_train, inter_test = train_test_split(_real_inter_arr, test_size=0.2, random_state=42)\n",
    "    syn_train,   syn_test   = train_test_split(_syn_arr,        test_size=0.2, random_state=42)\n",
    "    print(f'[TL] Loaded from disk: pre_train={len(pre_train)}, inter_train={len(inter_train)}, syn_train={len(syn_train)}')\n",
    "else:\n",
    "    if 'syn_train' not in dir() or syn_train is None:\n",
    "        syn_train, syn_test = train_test_split(syn_arr, test_size=0.2, random_state=42)\n",
    "    print(f'[TL] Using existing splits: pre_train={len(pre_train)}, syn_train={len(syn_train)}')\n",
    "\n",
    "# ── evaluate_tl_model: threshold=0.40 for better sensitivity ────────────────\n",
    "def evaluate_tl_model(model, test_loader, model_name, threshold=0.40):\n",
    "    \"\"\"Evaluate model. threshold=0.40 boosts sensitivity vs default 0.5.\"\"\"\n",
    "    model.eval()\n",
    "    all_probs, all_labels_list = [], []\n",
    "    with torch.no_grad():\n",
    "        for imgs, labels in test_loader:\n",
    "            out = model(imgs.to(device))\n",
    "            if hasattr(out, 'logits'):\n",
    "                out = out.logits\n",
    "            probs = torch.sigmoid(out.squeeze().cpu())\n",
    "            all_probs.extend(probs.numpy())\n",
    "            all_labels_list.extend(labels.numpy())\n",
    "\n",
    "    all_probs     = np.array(all_probs)\n",
    "    all_labels_np = np.array(all_labels_list)\n",
    "    all_preds     = (all_probs >= threshold).astype(float)\n",
    "\n",
    "    tp = ((all_preds==1)&(all_labels_np==1)).sum()\n",
    "    fn = ((all_preds==0)&(all_labels_np==1)).sum()\n",
    "    fp = ((all_preds==1)&(all_labels_np==0)).sum()\n",
    "    tn = ((all_preds==0)&(all_labels_np==0)).sum()\n",
    "    return {\n",
    "        'model':       model_name,\n",
    "        'sensitivity': float(tp/(tp+fn+1e-8)),\n",
    "        'specificity': float(tn/(tn+fp+1e-8)),\n",
    "        'accuracy':    float((tp+tn)/len(all_labels_np)),\n",
    "        'fpr':         float(fp/(fp+tn+1e-8)),\n",
    "    }\n",
    "\n",
    "# ── train_tl_model: 2-phase training + BCEWithLogitsLoss + pos_weight ────────\n",
    "def train_tl_model(model_name, train_ds, test_ds,\n",
    "                   phase1_epochs=10, phase2_epochs=20, lr_head=1e-3, lr_finetune=1e-4):\n",
    "    \"\"\"\n",
    "    2-Phase Training:\n",
    "      Phase 1: Backbone frozen  → train only head (lr=1e-3, 10 epochs)\n",
    "      Phase 2: All layers open  → fine-tune whole network (lr=1e-4, 20 epochs)\n",
    "    Loss: BCEWithLogitsLoss + pos_weight to handle class imbalance.\n",
    "    \"\"\"\n",
    "    input_size = 299 if model_name == 'inceptionv3' else 224\n",
    "    _transform = transforms.Compose([\n",
    "        transforms.ToPILImage(),\n",
    "        transforms.Resize((input_size, input_size)),\n",
    "        transforms.RandomHorizontalFlip(),\n",
    "        transforms.RandomRotation(10),\n",
    "        transforms.ColorJitter(brightness=0.2, contrast=0.2),\n",
    "        transforms.ToTensor(),\n",
    "        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])\n",
    "    ])\n",
    "    _test_transform = transforms.Compose([\n",
    "        transforms.ToPILImage(),\n",
    "        transforms.Resize((input_size, input_size)),\n",
    "        transforms.ToTensor(),\n",
    "        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])\n",
    "    ])\n",
    "    train_ds.transform = _transform\n",
    "    test_ds.transform  = _test_transform\n",
    "    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True,  num_workers=0)\n",
    "    test_loader  = DataLoader(test_ds,  batch_size=16, shuffle=False, num_workers=0)\n",
    "\n",
    "    model_path = os.path.join(SAVE_MODEL, f'tl_{model_name}.pth')\n",
    "\n",
    "    # ── Load if already saved (skip training) ────────────────────────────────\n",
    "    if os.path.exists(model_path):\n",
    "        print(f'  [SKIP] {model_name.upper()} already trained — loading {model_path}')\n",
    "        model = build_tl_model(model_name, freeze_backbone=False)\n",
    "        model.load_state_dict(torch.load(model_path, map_location=device))\n",
    "        return evaluate_tl_model(model, test_loader, model_name)\n",
    "\n",
    "    # ── Compute pos_weight for class imbalance ────────────────────────────────\n",
    "    train_labels = np.array(train_ds.labels)\n",
    "    n_pos = (train_labels == 1.0).sum()\n",
    "    n_neg = (train_labels == 0.0).sum()\n",
    "    pos_weight = torch.tensor([n_neg / max(1, n_pos)]).to(device)\n",
    "    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)\n",
    "    print(f'  pos_weight={pos_weight.item():.2f} (neg={n_neg}, pos={n_pos})')\n",
    "\n",
    "    # ── Phase 1: Frozen backbone — train only head ────────────────────────────\n",
    "    print(f'  [Phase 1] {model_name.upper()} — training HEAD only ({phase1_epochs} epochs, lr={lr_head})...')\n",
    "    model = build_tl_model(model_name, freeze_backbone=True)\n",
    "    optimizer = optim.Adam(\n",
    "        filter(lambda p: p.requires_grad, model.parameters()), lr=lr_head\n",
    "    )\n",
    "    for epoch in range(phase1_epochs):\n",
    "        model.train()\n",
    "        epoch_loss = 0.0\n",
    "        for imgs, labels in train_loader:\n",
    "            imgs, labels = imgs.to(device), labels.to(device)\n",
    "            optimizer.zero_grad()\n",
    "            output = model(imgs)\n",
    "            if model_name == 'inceptionv3' and hasattr(output, 'logits'):\n",
    "                loss = criterion(output.logits.squeeze(), labels) \\\n",
    "                     + 0.4 * criterion(output.aux_logits.squeeze(), labels)\n",
    "            else:\n",
    "                loss = criterion(output.squeeze(), labels)\n",
    "            loss.backward()\n",
    "            optimizer.step()\n",
    "            epoch_loss += loss.item()\n",
    "        if (epoch+1) % 5 == 0:\n",
    "            print(f'    Phase1 Epoch {epoch+1}/{phase1_epochs} | Loss: {epoch_loss/len(train_loader):.4f}')\n",
    "\n",
    "    # ── Phase 2: Unfreeze all — fine-tune whole network ───────────────────────\n",
    "    print(f'  [Phase 2] {model_name.upper()} — fine-tuning ALL layers ({phase2_epochs} epochs, lr={lr_finetune})...')\n",
    "    for param in model.parameters():\n",
    "        param.requires_grad = True\n",
    "    optimizer = optim.Adam(model.parameters(), lr=lr_finetune, weight_decay=1e-4)\n",
    "    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)\n",
    "\n",
    "    for epoch in range(phase2_epochs):\n",
    "        model.train()\n",
    "        epoch_loss = 0.0\n",
    "        for imgs, labels in train_loader:\n",
    "            imgs, labels = imgs.to(device), labels.to(device)\n",
    "            optimizer.zero_grad()\n",
    "            output = model(imgs)\n",
    "            if model_name == 'inceptionv3' and hasattr(output, 'logits'):\n",
    "                loss = criterion(output.logits.squeeze(), labels) \\\n",
    "                     + 0.4 * criterion(output.aux_logits.squeeze(), labels)\n",
    "            else:\n",
    "                loss = criterion(output.squeeze(), labels)\n",
    "            loss.backward()\n",
    "            optimizer.step()\n",
    "            epoch_loss += loss.item()\n",
    "        scheduler.step()\n",
    "        if (epoch+1) % 5 == 0:\n",
    "            print(f'    Phase2 Epoch {epoch+1}/{phase2_epochs} | Loss: {epoch_loss/len(train_loader):.4f}')\n",
    "\n",
    "    torch.save(model.state_dict(), model_path)\n",
    "    print(f'  Saved to {model_path}')\n",
    "    return evaluate_tl_model(model, test_loader, model_name)\n",
    "\n",
    "# ── Build datasets and run all 4 TL models ───────────────────────────────────\n",
    "tl_train = TLDataset(pre_train.tolist(), inter_train.tolist(), syn_train.tolist())\n",
    "tl_test  = TLDataset(pre_test.tolist(),  inter_test.tolist())\n",
    "\n",
    "tl_results = []\n",
    "for model_name in ['vgg16', 'vgg19', 'resnet50', 'inceptionv3']:\n",
    "    print(f\"\\n{'='*60}\")\n",
    "    print(f'Transfer Learning: {model_name.upper()}')\n",
    "    print('='*60)\n",
    "    result = train_tl_model(model_name, tl_train, tl_test)\n",
    "    tl_results.append(result)\n",
    "    print(f\"  Sen: {result['sensitivity']:.4f} | Spec: {result['specificity']:.4f} | \"\n",
    "          f\"Acc: {result['accuracy']:.4f} | FPR: {result['fpr']:.4f}\")\n",
    "\n",
    "print('\\n[DONE] All 4 TL models trained and evaluated.')\n",
]

cells[52]['source'] = cell_52_source
cells[52]['outputs'] = []
cells[52]['execution_count'] = None
print("[FIXED] Cell 52 — train_tl_model() + evaluate_tl_model() with all 4 fixes")

# ── Step 5: Fix Cell 53 (duplicate build_tl_model — update to match) ─────────
cell_53_source = [
    "# build_tl_model defined in Cell 50 above\n",
    "# This cell is kept for reference only\n",
    "print('TL model builder already defined in Cell 50.')\n",
]
cells[53]['source'] = cell_53_source
cells[53]['outputs'] = []
cells[53]['execution_count'] = None
print("[FIXED] Cell 53 — duplicate build_tl_model replaced with reference note")

# ── Step 6: Save notebook ─────────────────────────────────────────────────────
with open(NB_PATH, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("\n" + "="*60)
print("SUCCESS: All TL cells fixed in notebook!")
print("="*60)
print("Fixes applied:")
print("  [1] Cell 50: build_tl_model() — backbone frozen, raw logit output")
print("  [2] Cell 52: evaluate_tl_model() — threshold=0.40 (was 0.50)")
print("  [3] Cell 52: train_tl_model() — BCEWithLogitsLoss + pos_weight")
print("  [4] Cell 52: 2-Phase Training (10ep frozen + 20ep finetune)")
print("  [5] Cell 52: Data augmentation (flip, rotate, colorjitter)")
print("  [6] Cell 53: Duplicate build_tl_model removed")
print(f"  [7] {len(deleted)} old .pth file(s) deleted for fresh retrain")
