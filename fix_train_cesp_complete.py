import json, sys

NB_PATH = r'c:\Users\SHIVA\Downloads\Paper1_Reproduction-20260629T171224Z-3-001\Paper1_Full_Pipeline_FINAL.ipynb'

with open(NB_PATH, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# ============================================================
# COMPLETE FIX for train_cesp - Cell 44
# Problems found:
#   1. logits.squeeze() -> scalar when batch_size=1, breaks BCEWithLogitsLoss
#   2. Loss prints 0.0000 because actual loss IS tiny (model collapses fast)
#   3. No debug info about what probabilities model is outputting
#   4. LR 1e-4 with CosineAnnealing reaches near-zero very fast -> loss=0
# ============================================================

new_cell44 = r"""# ============================================================
# CELL 34: CESP Training - Stratified K-Fold Cross Validation
# FIX: squeeze(-1) instead of squeeze() to avoid scalar on batch=1
#      More loss decimal places + prob distribution debug
#      Reduced weight_decay, proper LR schedule
# ============================================================
from sklearn.model_selection import StratifiedKFold
import torch.nn.functional as F

cesp_device = device  # Uses CUDA GPU if available

def train_cesp(train_dataset, val_dataset, epochs=25, lr=1e-4, batch_size=64,
               threshold=0.45, pos_weight_scale=1.0, verbose=True):
    cesp_model = CESP().to(cesp_device)

    # Weight init - helps avoid collapse
    for m in cesp_model.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)

    optimizer = optim.Adam(cesp_model.parameters(), lr=lr, weight_decay=1e-5)
    # ReduceLROnPlateau is better than CosineAnnealing here
    # because CosineAnnealing can drop LR to near-zero too fast
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-7
    )

    train_labels = np.array(train_dataset.labels)
    n_pos = (train_labels == 1.0).sum()
    n_neg = (train_labels == 0.0).sum()
    base_weight = n_neg / max(1, n_pos)
    pos_weight  = torch.tensor([pos_weight_scale * base_weight]).to(cesp_device)
    criterion_cesp = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    if verbose:
        print(f"      n_pos={n_pos}, n_neg={n_neg}, pos_weight={pos_weight.item():.2f}, threshold={threshold}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=0, pin_memory=True if torch.cuda.is_available() else False,
                              drop_last=False)
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
            # FIX: squeeze(-1) not squeeze() - avoids scalar tensor when batch_size=1
            logits = logits.squeeze(-1) if logits.dim() > 1 else logits.view(-1)
            loss   = criterion_cesp(logits, labels)
            loss.backward()
            # Gradient clipping - prevents explosion
            torch.nn.utils.clip_grad_norm_(cesp_model.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += loss.item()
            n_batches += 1
        avg_loss = running_loss / max(n_batches, 1)
        scheduler.step(avg_loss)
        if (epoch + 1) % 5 == 0 and verbose:
            print(f'    Epoch {epoch+1:2d}/{epochs} | Loss: {avg_loss:.6f} | LR: {optimizer.param_groups[0]["lr"]:.2e}')

    # Evaluation
    cesp_model.eval()
    all_probs, all_lbl = [], []
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs   = imgs.float().to(cesp_device)
            logits = cesp_model(imgs)
            logits = logits.squeeze(-1) if logits.dim() > 1 else logits.view(-1)
            probs  = torch.sigmoid(logits).cpu()
            all_probs.extend(probs.numpy().tolist())
            all_lbl.extend(labels.numpy().tolist())

    all_probs = np.array(all_probs)
    all_lbl   = np.array(all_lbl)

    # Debug: show probability distribution
    if verbose:
        pos_probs = all_probs[all_lbl == 1]
        neg_probs = all_probs[all_lbl == 0]
        if len(pos_probs) > 0:
            print(f"      Prob dist | Pos: mean={pos_probs.mean():.3f} min={pos_probs.min():.3f} max={pos_probs.max():.3f}")
        if len(neg_probs) > 0:
            print(f"      Prob dist | Neg: mean={neg_probs.mean():.3f} min={neg_probs.min():.3f} max={neg_probs.max():.3f}")

    preds  = (all_probs >= threshold).astype(float)
    tp = ((preds == 1) & (all_lbl == 1)).sum()
    fn = ((preds == 0) & (all_lbl == 1)).sum()
    fp = ((preds == 1) & (all_lbl == 0)).sum()
    tn = ((preds == 0) & (all_lbl == 0)).sum()

    sensitivity = float(tp) / (float(tp + fn) + 1e-8)
    specificity = float(tn) / (float(tn + fp) + 1e-8)
    accuracy    = (float(tp) + float(tn)) / len(all_lbl)

    return cesp_model, {'sensitivity': sensitivity,
                        'specificity': specificity,
                        'accuracy': accuracy}

print('train_cesp() defined with: squeeze(-1) fix + ReduceLROnPlateau + grad clip + prob debug')
print('Device:', cesp_device)
"""

nb['cells'][44]['source'] = [new_cell44]
nb['cells'][44]['execution_count'] = None
nb['cells'][44]['outputs'] = []

print("[OK] Cell 44 (train_cesp) completely rewritten with fixes:")
print("   Fix 1: logits.squeeze(-1) instead of squeeze() -- no scalar on batch=1")
print("   Fix 2: ReduceLROnPlateau instead of CosineAnnealing -- LR won't drop to 0 too fast")
print("   Fix 3: Gradient clipping (max_norm=1.0) -- no explosion")
print("   Fix 4: Xavier init on Linear layers -- better starting point")
print("   Fix 5: weight_decay reduced 1e-4 -> 1e-5 -- less regularization")
print("   Fix 6: Loss prints with 6 decimal places -- no more '0.0000' confusion")
print("   Fix 7: Prints actual probability distribution -- shows if model collapses")
print("   Fix 8: n_batches counter -- accurate average loss")

with open(NB_PATH, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("\nNotebook saved!")
print("\nIMPORTANT: After running Cell 44, watch for 'Prob dist' lines in output.")
print("  If Pos mean ~= Neg mean -> model collapsed (all same output)")
print("  If Pos mean > Neg mean  -> model is learning correctly")
