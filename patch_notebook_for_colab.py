"""
Colab Notebook Path Patcher
============================
Notebook mein Cell 1 (paths) ko Colab ke liye update karta hai.
Baaki sab cells same rehte hain.
"""

import json
from pathlib import Path

NB_PATH = r'c:\Users\SHIVA\Downloads\Paper1_Reproduction-20260629T171224Z-3-001\Paper1_Full_Pipeline_FINAL.ipynb'
OUT_PATH = r'c:\Users\SHIVA\Downloads\Paper1_Reproduction-20260629T171224Z-3-001\colab_upload\Paper1_Full_Pipeline_FINAL.ipynb'

with open(NB_PATH, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# ── NEW CELL 0: Colab setup (Drive mount + unzip) ───────────
colab_setup_cell = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# ============================================================\n",
        "# COLAB SETUP CELL - Run this FIRST before anything else!\n",
        "# ============================================================\n",
        "\n",
        "# Step 1: Mount Google Drive\n",
        "from google.colab import drive\n",
        "drive.mount('/content/drive')\n",
        "\n",
        "import os, zipfile\n",
        "from pathlib import Path\n",
        "\n",
        "# Step 2: Set your Drive folder path\n",
        "DRIVE_FOLDER = '/content/drive/MyDrive/Paper1_Reproduction'\n",
        "\n",
        "# Step 3: Create working directories in Colab\n",
        "os.makedirs('/content/data/preictal',   exist_ok=True)\n",
        "os.makedirs('/content/data/interictal', exist_ok=True)\n",
        "os.makedirs('/content/data/synthetic',  exist_ok=True)\n",
        "os.makedirs('/content/models',          exist_ok=True)\n",
        "\n",
        "# Step 4: Unzip data from Drive to Colab local storage (fast!)\n",
        "def unzip_if_needed(zip_path, out_dir):\n",
        "    if not os.path.exists(zip_path):\n",
        "        print(f'  [SKIP] {zip_path} not found in Drive')\n",
        "        return\n",
        "    files_exist = len(list(Path(out_dir).glob('**/*.png')))\n",
        "    if files_exist > 0:\n",
        "        print(f'  [SKIP] {out_dir} already has {files_exist} files')\n",
        "        return\n",
        "    print(f'  Unzipping {os.path.basename(zip_path)}...', end=' ', flush=True)\n",
        "    with zipfile.ZipFile(zip_path, 'r') as zf:\n",
        "        zf.extractall(out_dir)\n",
        "    count = len(list(Path(out_dir).glob('**/*.png')))\n",
        "    print(f'Done! ({count} files)')\n",
        "\n",
        "print('Extracting data files...')\n",
        "unzip_if_needed(f'{DRIVE_FOLDER}/preictal.zip',   '/content/data/preictal')\n",
        "unzip_if_needed(f'{DRIVE_FOLDER}/interictal.zip', '/content/data/interictal')\n",
        "unzip_if_needed(f'{DRIVE_FOLDER}/synthetic.zip',  '/content/data/synthetic')\n",
        "\n",
        "print('Setup complete!')\n",
        "print(f'  Preictal:   {len(list(Path(\"/content/data/preictal\").rglob(\"*.png\")))} images')\n",
        "print(f'  Interictal: {len(list(Path(\"/content/data/interictal\").rglob(\"*.png\")))} images')\n",
        "print(f'  Synthetic:  {len(list(Path(\"/content/data/synthetic\").rglob(\"*.png\")))} images')\n"
    ]
}

# ── REPLACE CELL 1: Paths update for Colab ──────────────────
new_cell1_source = [
    "# ============================================================\n",
    "# CELL 1: Paths & Project Setup - COLAB VERSION\n",
    "# ============================================================\n",
    "from pathlib import Path\n",
    "import os\n",
    "\n",
    "# Colab paths (data already unzipped in setup cell above)\n",
    "PROJECT_ROOT  = '/content/Paper1_Reproduction'\n",
    "DATASET_ROOT  = Path('/content/data')   # not used for CESP, but kept for compatibility\n",
    "SAVE_MODEL    = '/content/models'\n",
    "\n",
    "# Data paths (already extracted)\n",
    "PREICTAL_DIR   = '/content/data/preictal'\n",
    "INTERICTAL_DIR = '/content/data/interictal'\n",
    "SYNTHETIC_DIR  = '/content/data/synthetic'\n",
    "\n",
    "# Create project folders\n",
    "folders = [\n",
    "    '01_dataset/metadata',\n",
    "    '02_preprocessed/preictal',\n",
    "    '02_preprocessed/interictal',\n",
    "    '03_models',\n",
    "    '04_generated',\n",
    "    '05_results/figures',\n",
    "    '05_results/tables',\n",
    "]\n",
    "for f in folders:\n",
    "    os.makedirs(os.path.join(PROJECT_ROOT, f), exist_ok=True)\n",
    "\n",
    "print('PROJECT_ROOT :', PROJECT_ROOT)\n",
    "print('SAVE_MODEL   :', SAVE_MODEL)\n",
    "print('Setup complete!')\n"
]

# Insert Colab setup as first cell
nb['cells'].insert(0, colab_setup_cell)

# Update cell 1 (now cell 2 after insert) = original cell 1
nb['cells'][1]['source'] = new_cell1_source
nb['cells'][1]['execution_count'] = None
nb['cells'][1]['outputs'] = []

# Save Colab version
with open(OUT_PATH, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("[OK] Colab notebook created!")
print(f"     Saved to: {OUT_PATH}")
print()
print("Changes made:")
print("  + Added COLAB SETUP CELL at top (Drive mount + unzip)")
print("  + Updated Cell 1 paths to /content/... (Colab paths)")
print("  + All other cells unchanged")
