import json
import re

nb_path = 'Paper1_Full_Pipeline_FINAL.ipynb'
with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] != 'code': continue
    src = ''.join(cell['source'])
    
    if 'def train_cesp(' in src:
        new_src = """# ============================================================
# CELL 34: CESP Training - Stratified K-Fold Cross Validation
# Optimized: Fixed squeeze bug, bounded pos_weight, correctly computes metrics
# ============================================================
from sklearn.model_selection import StratifiedKFold
import torch.nn.functional as F

cesp_device = device  # Uses CUDA GPU if available

def train_cesp(train_dataset, val_dataset, epochs=25, lr=1e-4, batch_size=64, threshold=0.5):
    cesp_model = CESP().to(cesp_device)
    optimizer = optim.Adam(cesp_model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    train_labels = np.array(train_dataset.labels)
    n_pos = (train_labels == 1.0).sum()
    n_neg = (train_labels == 0.0).sum()
    
    base_weight = n_neg / max(1, n_pos)
    clamped_weight = min(10.0, base_weight) 
    pos_weight = torch.tensor([clamped_weight]).to(cesp_device)
    criterion_cesp = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=0, pin_memory=True if torch.cuda.is_available() else False, drop_last=False)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False,
                              num_workers=0, pin_memory=True if torch.cuda.is_available() else False)

    for epoch in range(epochs):
        cesp_model.train()
        running_loss = 0.0
        n_batches = 0
        for imgs, labels in train_loader:
            imgs   = imgs.float().to(cesp_device)
            labels = labels.float().to(cesp_device)
            optimizer.zero_grad()
            logits = cesp_model(imgs)
            logits = logits.squeeze(-1) if logits.dim() > 1 else logits.view(-1)
            loss   = criterion_cesp(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(cesp_model.parameters(), 1.0)
            optimizer.step()
            running_loss += loss.item()
            n_batches += 1
            
        avg_loss = running_loss / max(n_batches, 1)
        scheduler.step(avg_loss)

    cesp_model.eval()
    all_preds, all_lbl = [], []
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs   = imgs.float().to(cesp_device)
            logits = cesp_model(imgs)
            logits = logits.squeeze(-1) if logits.dim() > 1 else logits.view(-1)
            probs  = torch.sigmoid(logits).cpu()
            preds  = (probs >= threshold).float()
            all_preds.extend(preds.numpy())
            all_lbl.extend(labels.numpy())

    all_preds = np.array(all_preds)
    all_lbl   = np.array(all_lbl)

    tp = ((all_preds == 1) & (all_lbl == 1)).sum()
    fn = ((all_preds == 0) & (all_lbl == 1)).sum()
    fp = ((all_preds == 1) & (all_lbl == 0)).sum()
    tn = ((all_preds == 0) & (all_lbl == 0)).sum()

    sensitivity = tp / (tp + fn + 1e-8)
    specificity = tn / (tn + fp + 1e-8)
    accuracy    = (tp + tn) / len(all_lbl)
    fpr         = fp / (tn + fp + 1e-8)

    return cesp_model, {'sensitivity': float(sensitivity),
                        'specificity': float(specificity),
                        'accuracy': float(accuracy),
                        'fpr': float(fpr)}
"""
        cell['source'] = [line + '\n' for line in new_src.split('\n')]
        
    elif 'EXPERIMENT 1: TRTR' in src:
        src = re.sub(r'epochs=\d+, lr=[^,]+, batch_size=64, threshold=[^)]+', 'epochs=25, lr=1e-4, batch_size=64, threshold=0.5', src)
        if 'trtr_fpr' not in src:
            src = src.replace('trtr_acc  = np.mean([r["accuracy"]    for r in trtr_results])', 
                              'trtr_acc  = np.mean([r["accuracy"]    for r in trtr_results])\n    trtr_fpr  = np.mean([r.get("fpr", 0.0) for r in trtr_results])')
            src = src.replace('Spec: {trtr_spec:.4f} | Acc: {trtr_acc:.4f}\')', 
                              'Spec: {trtr_spec:.4f} | Acc: {trtr_acc:.4f} | FPR: {trtr_fpr:.4f}\')')
        cell['source'] = src.splitlines(True)
        
    elif 'EXPERIMENT 2: TSTR' in src:
        src = re.sub(r'epochs=\d+, lr=[^,]+, batch_size=64, threshold=[^)]+', 'epochs=25, lr=1e-4, batch_size=64, threshold=0.5', src)
        if 'tstr_fpr' not in src:
            src = src.replace('tstr_acc  = np.mean([r["accuracy"]    for r in tstr_results])', 
                              'tstr_acc  = np.mean([r["accuracy"]    for r in tstr_results])\n    tstr_fpr  = np.mean([r.get("fpr", 0.0) for r in tstr_results])')
            src = src.replace('Spec: {tstr_spec:.4f} | Acc: {tstr_acc:.4f}\')', 
                              'Spec: {tstr_spec:.4f} | Acc: {tstr_acc:.4f} | FPR: {tstr_fpr:.4f}\')')
        cell['source'] = src.splitlines(True)
        
    elif 'EXPERIMENT 3 & 4:' in src or 'TRTS' in src:
        src = re.sub(r'epochs=\d+, lr=[^,]+, batch_size=64, threshold=[^)]+', 'epochs=25, lr=1e-4, batch_size=64, threshold=0.5', src)
        cell['source'] = src.splitlines(True)

with open(nb_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)
print("Notebook metrics patched successfully!")
