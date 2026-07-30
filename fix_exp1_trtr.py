import json

NB_PATH = r'c:\Users\SHIVA\Downloads\Paper1_Reproduction-20260629T171224Z-3-001\Paper1_Full_Pipeline_FINAL.ipynb'

with open(NB_PATH, 'r', encoding='utf-8') as f:
    nb = json.load(f)

new_trtr_source = '''# ============================================================
# CELL 35: Run CESP — Experiment 1: TRTR (Train Real, Test Real)
# FIX: epochs 10->30, lr 1e-3->5e-4, threshold 0.4->0.45
# Paper target: Sen ~89.02%
# ============================================================
print('='*60)
print('EXPERIMENT 1: TRTR — Train on Real, Test on Real')
print('='*60)

all_paths      = real_pre_paths + real_inter_paths
all_labels     = [1]*len(real_pre_paths) + [0]*len(real_inter_paths)
all_paths      = np.array(all_paths)
all_labels_arr = np.array(all_labels)

print(f"Real preictal   : {len(real_pre_paths)}")
print(f"Real interictal : {len(real_inter_paths)}")

skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
trtr_results = []

for fold, (train_idx, val_idx) in enumerate(skf.split(all_paths, all_labels_arr)):
    train_pre   = all_paths[train_idx][all_labels_arr[train_idx] == 1]
    train_inter = all_paths[train_idx][all_labels_arr[train_idx] == 0]
    val_pre     = all_paths[val_idx][all_labels_arr[val_idx] == 1]
    val_inter   = all_paths[val_idx][all_labels_arr[val_idx] == 0]

    train_ds = CESPDataset(train_pre, train_inter)
    val_ds   = CESPDataset(val_pre, val_inter)

    # FIX: epochs=30 (was 10), lr=5e-4 (was 1e-3), threshold=0.45
    _, metrics = train_cesp(train_ds, val_ds, epochs=30, lr=5e-4, batch_size=64, threshold=0.45)
    trtr_results.append(metrics)
    print(f\'  Fold {fold+1:2d} | Sen: {metrics["sensitivity"]:.4f} | \'
          f\'Spec: {metrics["specificity"]:.4f} | Acc: {metrics["accuracy"]:.4f}\')

trtr_sen  = np.mean([r["sensitivity"] for r in trtr_results])
trtr_spec = np.mean([r["specificity"] for r in trtr_results])
trtr_acc  = np.mean([r["accuracy"]    for r in trtr_results])
trtr_fpr  = 1.0 - trtr_spec   # FPR = 1 - Specificity

print(f\'\\nTRTR Average | Sen: {trtr_sen:.4f} | Spec: {trtr_spec:.4f} | Acc: {trtr_acc:.4f} | FPR: {trtr_fpr:.4f}\')
print(f\'Paper target | Sen: ~0.8902 (89.02%)\')
'''

# Apply fix to Cell 46 (CELL 35 — TRTR)
nb['cells'][46]['source'] = [new_trtr_source]
nb['cells'][46]['execution_count'] = None
nb['cells'][46]['outputs'] = []

with open(NB_PATH, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("SUCCESS: Cell 46 (CELL 35 - TRTR Experiment 1) fixed!")
print("Fixes applied:")
print("  1. epochs: 10 -> 30  (model now trains enough to differentiate classes)")
print("  2. lr: 1e-3 -> 5e-4  (stable learning, no divergence)")
print("  3. threshold: 0.4 -> 0.45  (matches train_cesp default, avoids all-positive bias)")
print("  4. Added FPR = 1 - Specificity output")
print("  5. Added data count print for verification")
