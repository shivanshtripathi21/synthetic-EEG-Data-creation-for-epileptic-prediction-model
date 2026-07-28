"""
fix_cesp_notebook.py
--------------------
Ye script Paper1_Full_Pipeline_FINAL.ipynb mein CESP training cells ko fix karta hai.
Problem: RuntimeError: GET was unable to find an engine to execute this computation
Fix:
  1. CESP ke liye force CPU device (cesp_device = torch.device('cpu'))
  2. imgs.float() add karna (dtype mismatch fix)
  3. batch_size 32 → 16 (memory ke liye)
  4. pin_memory=False (Windows fix)
  5. TRTR cell ka stale error output clear karna
"""

import json, copy, re

NOTEBOOK_PATH = r"C:\Users\SHIVA\Downloads\Paper1_Reproduction-20260629T171224Z-3-001\Paper1_Full_Pipeline_FINAL.ipynb"

# ── New fixed train_cesp source ──────────────────────────────────────────────
FIXED_TRAIN_CESP = [
    "# ============================================================\n",
    "# CELL 34: CESP Training — 10-Fold Cross Validation\n",
    "# Paper: k=10 fold, 90% train / 10% val, LR=1e-4, Adam, BCELoss\n",
    "# 4 Experiments: TRTR, TSTR, TRTS, TSTS\n",
    "# FIX: Force CPU to avoid 'GET was unable to find an engine' error\n",
    "# ============================================================\n",
    "from sklearn.model_selection import KFold\n",
    "import torch.nn.functional as F\n",
    "\n",
    "# Force CPU — avoids CUDA engine error on Windows\n",
    "cesp_device = torch.device('cpu')\n",
    "print('CESP will use device:', cesp_device)\n",
    "\n",
    "def train_cesp(train_dataset, val_dataset, epochs=30, lr=1e-4):\n",
    "    cesp_model = CESP().to(cesp_device)\n",
    "    optimizer = optim.Adam(cesp_model.parameters(), lr=lr)\n",
    "    criterion_cesp = nn.BCELoss()\n",
    "\n",
    "    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True,\n",
    "                              num_workers=0, pin_memory=False)\n",
    "    val_loader   = DataLoader(val_dataset,   batch_size=16, shuffle=False,\n",
    "                              num_workers=0, pin_memory=False)\n",
    "\n",
    "    for epoch in range(epochs):\n",
    "        cesp_model.train()\n",
    "        running_loss = 0.0\n",
    "        for imgs, labels in train_loader:\n",
    "            imgs   = imgs.float().to(cesp_device)    # .float() fixes dtype error\n",
    "            labels = labels.float().to(cesp_device)\n",
    "            optimizer.zero_grad()\n",
    "            out  = cesp_model(imgs)\n",
    "            loss = criterion_cesp(out, labels)\n",
    "            loss.backward()\n",
    "            optimizer.step()\n",
    "            running_loss += loss.item()\n",
    "        if (epoch + 1) % 10 == 0:\n",
    "            avg = running_loss / max(len(train_loader), 1)\n",
    "            print(f'    Epoch {epoch+1}/{epochs} | Loss: {avg:.4f}')\n",
    "\n",
    "    # Evaluate\n",
    "    cesp_model.eval()\n",
    "    all_preds, all_lbl = [], []\n",
    "    with torch.no_grad():\n",
    "        for imgs, labels in val_loader:\n",
    "            imgs  = imgs.float().to(cesp_device)\n",
    "            out   = cesp_model(imgs).cpu()\n",
    "            preds = (out >= 0.5).float()\n",
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
    "print('train_cesp() function defined. Device:', cesp_device)\n",
]

# ── Load notebook ────────────────────────────────────────────────────────────
print("Loading notebook...")
with open(NOTEBOOK_PATH, "r", encoding="utf-8") as f:
    nb = json.load(f)

cells = nb["cells"]
fixed_count = 0

for i, cell in enumerate(cells):
    if cell.get("cell_type") != "code":
        continue

    src = "".join(cell.get("source", []))

    # ── Fix 1: train_cesp definition cell ───────────────────────────────────
    if "def train_cesp(" in src and "criterion_cesp = nn.BCELoss()" in src:
        print(f"  [Cell {i}] Fixing train_cesp() definition...")
        cell["source"] = FIXED_TRAIN_CESP
        cell["outputs"] = []          # clear stale outputs
        cell["execution_count"] = None
        fixed_count += 1

    # ── Fix 2: TRTR cell — clear stale RuntimeError output ──────────────────
    elif "EXPERIMENT 1: TRTR" in src and "trtr_results" in src:
        print(f"  [Cell {i}] Clearing stale RuntimeError from TRTR cell...")
        # Remove error outputs, keep stdout outputs
        cell["outputs"] = [o for o in cell.get("outputs", [])
                           if o.get("output_type") != "error"]
        cell["execution_count"] = None
        fixed_count += 1

    # ── Fix 3: TSTR cell — clear stale outputs ──────────────────────────────
    elif "EXPERIMENT 2: TSTR" in src and "tstr_model" in src:
        print(f"  [Cell {i}] Clearing stale outputs from TSTR cell...")
        cell["outputs"] = []
        cell["execution_count"] = None
        fixed_count += 1

    # ── Fix 4: Any other CESP experiment cell ───────────────────────────────
    elif ("EXPERIMENT" in src and "train_cesp(" in src and
          "TRTR" not in src and "TSTR" not in src):
        print(f"  [Cell {i}] Clearing stale outputs from other CESP cell...")
        cell["outputs"] = []
        cell["execution_count"] = None
        fixed_count += 1

    # ── Fix 5: Debug/sanity cell that references train_loader before it's made ──
    elif ("sample_imgs, sample_labels = next(iter(train_loader))" in src and
          "CUDA available" in src):
        print(f"  [Cell {i}] Removing broken debug cell source...")
        cell["source"] = [
            "# Debug cell — skipping (train_loader not defined at this point)\n",
            "print('Skipping debug cell — train_loader created inside train_cesp()')\n"
        ]
        cell["outputs"] = []
        cell["execution_count"] = None
        fixed_count += 1

# ── Save fixed notebook ──────────────────────────────────────────────────────
print(f"\nFixed {fixed_count} cell(s). Saving notebook...")
with open(NOTEBOOK_PATH, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print("✅ Done! Notebook saved successfully.")
print("\nAb Jupyter mein karo:")
print("  1. Kernel → Restart Kernel")
print("  2. Cell 31 (CESP Architecture) se Run All karein")
