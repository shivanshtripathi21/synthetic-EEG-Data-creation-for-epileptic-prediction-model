"""
Fix CESP model Linear layer size mismatch.

Problem: CESPDataset resizes images to 128x128, but the CESP model's Linear
layer assumes 256x256 input (64*32*32 = 65536). After 3 MaxPool2d(2,2) on
128x128 input, spatial dims become 16x16, so correct size is 64*16*16 = 16384.

This script patches the notebook cell that defines CESP to use the correct size.
"""

import json
from pathlib import Path

nb_path = Path(r"Paper1_Full_Pipeline_FINAL.ipynb")

with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

fixed = 0
for i, cell in enumerate(nb["cells"]):
    if cell["cell_type"] != "code":
        continue
    src = "".join(cell["source"])
    if "class CESP(nn.Module)" in src and "64 * 32 * 32" in src:
        new_source = []
        for line in cell["source"]:
            line = line.replace("64 * 32 * 32 = 65,536", "64 * 16 * 16 = 16,384  (input is 128x128, 3x MaxPool -> 16x16)")
            line = line.replace("64 * 32 * 32", "64 * 16 * 16")
            new_source.append(line)
        cell["source"] = new_source
        fixed += 1
        print(f"  Fixed cell {i}: CESP model Linear layer 64*32*32 -> 64*16*16")

# Also fix the test tensor shape if it uses 256x256
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
        fixed += 1
        print(f"  Fixed cell {i}: test tensor 256x256 -> 128x128")

if fixed > 0:
    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)
    print(f"\n✅ Done! Fixed {fixed} cell(s). Restart kernel and re-run from the CESP cell.")
else:
    print("⚠️ Nothing to fix — already patched or cell not found.")
