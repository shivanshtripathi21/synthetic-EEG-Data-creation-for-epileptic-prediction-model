import json

NB_PATH = r'c:\Users\SHIVA\Downloads\Paper1_Reproduction-20260629T171224Z-3-001\Paper1_Full_Pipeline_FINAL.ipynb'

with open(NB_PATH, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# New fixed CELL 36 - Experiment 2 TSTR
new_cell_36_source = '''# ============================================================
# CELL 36: CESP — Experiment 2: TSTR (Train Synthetic+Anchor, Test Real)
# Anchor Strategy: real preictal from train fold added to synthetic training
# Paper target: Sen ~88.21%, FPR/h ~0.14
# ============================================================
print("="*60)
print("EXPERIMENT 2: TSTR — Train on Synthetic+Anchor, Test on Real")
print("="*60)

import os
from sklearn.model_selection import StratifiedKFold

TSTR_MODEL_PATH = os.path.join(SAVE_MODEL, "cesp_tstr.pth")

# ── Data preparation ─────────────────────────────────────────
real_pre_arr   = np.array(real_pre_paths)
real_inter_arr = np.array(real_inter_paths)
syn_arr        = np.array(syn_paths_accepted)

print(f"Real preictal   : {len(real_pre_arr)}")
print(f"Real interictal : {len(real_inter_arr)}")
print(f"Synthetic       : {len(syn_arr)}")

# ── 10-Fold TSTR with Anchor Strategy ────────────────────────
all_real   = np.concatenate([real_pre_arr, real_inter_arr])
all_labels = np.array([1]*len(real_pre_arr) + [0]*len(real_inter_arr))

skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
tstr_results = []

for fold, (train_idx, val_idx) in enumerate(skf.split(all_real, all_labels)):
    val_paths  = all_real[val_idx]
    val_labels = all_labels[val_idx]
    val_pre   = val_paths[val_labels == 1]
    val_inter = val_paths[val_labels == 0]

    # Training data from train split
    tr_inter = all_real[train_idx][all_labels[train_idx] == 0]
    tr_pre   = all_real[train_idx][all_labels[train_idx] == 1]  # Anchor

    # Balance synthetic count to interictal count
    n_inter    = len(tr_inter)
    syn_subset = syn_arr[:n_inter] if len(syn_arr) >= n_inter else syn_arr

    # Anchor strategy: real preictal + real interictal + synthetic preictal
    train_ds = CESPDataset(tr_pre.tolist(), tr_inter.tolist(), syn_subset.tolist())
    val_ds   = CESPDataset(val_pre.tolist(), val_inter.tolist())

    _, metrics = train_cesp(train_ds, val_ds, epochs=30, lr=5e-4, batch_size=64, threshold=0.35)
    tstr_results.append(metrics)
    print(f"  Fold {fold+1:2d} | Sen: {metrics[\'sensitivity\']:.4f} | "
          f"Spec: {metrics[\'specificity\']:.4f} | Acc: {metrics[\'accuracy\']:.4f}")

tstr_sen  = np.mean([r["sensitivity"] for r in tstr_results])
tstr_spec = np.mean([r["specificity"] for r in tstr_results])
tstr_acc  = np.mean([r["accuracy"]    for r in tstr_results])
# FPR = 1 - Specificity  (FP / (FP + TN) = 1 - TN/(FP+TN))
tstr_fpr  = 1.0 - tstr_spec

print(f"\\nTSTR Average | Sen: {tstr_sen:.4f} | Spec: {tstr_spec:.4f} | "
      f"Acc: {tstr_acc:.4f} | FPR: {tstr_fpr:.4f}")
print(f"Paper target | Sen: ~0.8821 (88.21%), FPR/h ~0.14")

# ── Save final TSTR model (train on all data) ─────────────────
print("\\n[INFO] Training final TSTR model on all data for saving...")
full_train_ds = CESPDataset(real_pre_arr.tolist(), real_inter_arr.tolist(), syn_arr.tolist())
full_test_ds  = CESPDataset(real_pre_arr.tolist(), real_inter_arr.tolist())
tstr_model, _ = train_cesp(full_train_ds, full_test_ds, epochs=30, lr=5e-4, batch_size=64, threshold=0.35)
torch.save(tstr_model.state_dict(), TSTR_MODEL_PATH)
print(f"CESP (TSTR) saved to {TSTR_MODEL_PATH}")
'''

# Replace cell 46
nb['cells'][46]['source'] = [new_cell_36_source]
nb['cells'][46]['execution_count'] = None
nb['cells'][46]['outputs'] = []

with open(NB_PATH, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("SUCCESS: Cell 46 (CELL 36 - Experiment 2 TSTR) updated!")
print("Fixes applied:")
print("  1. Removed unused 'train_test_split' import")
print("  2. FPR correctly computed as (1 - Specificity)")
print("  3. threshold=0.35 added to all train_cesp() calls")
print("  4. Final model trains with anchor (real_pre + real_inter + syn)")
print("  5. epochs=30, lr=5e-4, batch_size=64 applied consistently")
