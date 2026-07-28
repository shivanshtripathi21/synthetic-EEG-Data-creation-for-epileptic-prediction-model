import json

with open('Paper1_Full_Pipeline_FINAL.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

cells = nb['cells']
c = cells[44]
src = ''.join(c.get('source', []))

if 'cesp_device' not in src and 'def train_cesp' in src:
    print('OLD code found - applying fix...')
    new_source = [
        "# ============================================================\n",
        "# CELL 34: CESP Training - 10-Fold Cross Validation\n",
        "# FIX: Force CPU device + .float() to avoid RuntimeError\n",
        "# ============================================================\n",
        "from sklearn.model_selection import KFold\n",
        "import torch.nn.functional as F\n",
        "\n",
        "# Force CPU - avoids CUDA engine error on Windows\n",
        "cesp_device = torch.device('cpu')\n",
        "\n",
        "def train_cesp(train_dataset, val_dataset, epochs=30, lr=1e-4):\n",
        "    cesp_model = CESP().to(cesp_device)\n",
        "    optimizer = optim.Adam(cesp_model.parameters(), lr=lr)\n",
        "    criterion_cesp = nn.BCELoss()\n",
        "\n",
        "    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True,  num_workers=0, pin_memory=False)\n",
        "    val_loader   = DataLoader(val_dataset,   batch_size=16, shuffle=False, num_workers=0, pin_memory=False)\n",
        "\n",
        "    for epoch in range(epochs):\n",
        "        cesp_model.train()\n",
        "        running_loss = 0.0\n",
        "        for imgs, labels in train_loader:\n",
        "            imgs   = imgs.float().to(cesp_device)\n",
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
        "    return cesp_model, {'sensitivity': sensitivity, 'specificity': specificity, 'accuracy': accuracy}\n",
        "\n",
        "print('train_cesp() defined. Device:', cesp_device)\n",
    ]

    cells[44]['source'] = new_source
    cells[44]['outputs'] = []
    cells[44]['execution_count'] = None

    with open('Paper1_Full_Pipeline_FINAL.ipynb', 'w', encoding='utf-8') as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)

    print('SUCCESS! Fix applied to cell 44.')
    print('Verify: cesp_device in new source =', 'cesp_device' in ''.join(new_source))
else:
    print('cesp_device already present =', 'cesp_device' in src)
    print('Already fixed!')
