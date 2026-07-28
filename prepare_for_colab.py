"""
Colab Upload Preparation Script
================================
Yeh script Google Drive mein upload ke liye
data ZIP files banata hai.

Run: python prepare_for_colab.py
"""

import zipfile
import os
import shutil
from pathlib import Path

BASE = Path(r'c:\Users\SHIVA\Downloads\Paper1_Reproduction-20260629T171224Z-3-001\Paper1_Reproduction')
OUT_DIR = Path(r'c:\Users\SHIVA\Downloads\Paper1_Reproduction-20260629T171224Z-3-001\colab_upload')
OUT_DIR.mkdir(exist_ok=True)

print("="*60)
print("Colab Upload Preparation")
print("="*60)

# ── 1. ZIP preictal images ──────────────────────────────────
print("\n[1/4] Zipping preictal images...")
preictal_dir = BASE / '02_preprocessed' / 'preictal'
zip1_path = OUT_DIR / 'preictal.zip'
files = list(preictal_dir.rglob('*.png'))
with zipfile.ZipFile(zip1_path, 'w', zipfile.ZIP_DEFLATED) as zf:
    for f in files:
        zf.write(f, f.relative_to(preictal_dir))
print(f"   Done! {len(files)} files -> preictal.zip ({zip1_path.stat().st_size/1e6:.1f} MB)")

# ── 2. ZIP interictal images ────────────────────────────────
print("\n[2/4] Zipping interictal images...")
inter_dir = BASE / '02_preprocessed' / 'interictal'
zip2_path = OUT_DIR / 'interictal.zip'
files2 = list(inter_dir.rglob('*.png'))
with zipfile.ZipFile(zip2_path, 'w', zipfile.ZIP_DEFLATED) as zf:
    for f in files2:
        zf.write(f, f.relative_to(inter_dir))
print(f"   Done! {len(files2)} files -> interictal.zip ({zip2_path.stat().st_size/1e6:.1f} MB)")

# ── 3. ZIP synthetic/generated images (first 2660 accepted) ─
print("\n[3/4] Zipping synthetic images (first 2660)...")
gen_dir = BASE / '04_generated'
zip3_path = OUT_DIR / 'synthetic.zip'
all_gen = sorted(gen_dir.glob('*.png'))[:2660]
with zipfile.ZipFile(zip3_path, 'w', zipfile.ZIP_DEFLATED) as zf:
    for f in all_gen:
        zf.write(f, f.name)
print(f"   Done! {len(all_gen)} files -> synthetic.zip ({zip3_path.stat().st_size/1e6:.1f} MB)")

# ── 4. Copy notebook ────────────────────────────────────────
print("\n[4/4] Copying notebook...")
nb_src = Path(r'c:\Users\SHIVA\Downloads\Paper1_Reproduction-20260629T171224Z-3-001\Paper1_Full_Pipeline_FINAL.ipynb')
nb_dst = OUT_DIR / 'Paper1_Full_Pipeline_FINAL.ipynb'
shutil.copy2(nb_src, nb_dst)
print(f"   Done! -> {nb_dst.name}")

print("\n" + "="*60)
print("UPLOAD FOLDER:", OUT_DIR)
print("="*60)
print("\nFiles to upload to Google Drive:")
for f in sorted(OUT_DIR.iterdir()):
    size_mb = f.stat().st_size / 1e6
    print(f"  {f.name:40s} {size_mb:.1f} MB")

total = sum(f.stat().st_size for f in OUT_DIR.iterdir()) / 1e6
print(f"\n  Total size: {total:.1f} MB")
print("\n" + "="*60)
print("NEXT STEPS:")
print("  1. Open Google Drive: drive.google.com")
print("  2. Create folder: 'Paper1_Reproduction'")
print("  3. Upload all files from:", OUT_DIR)
print("  4. Open Colab notebook from Drive")
print("  5. Set Runtime -> T4 GPU")
print("="*60)
