import json, sys

NB_PATH = r'c:\Users\SHIVA\Downloads\Paper1_Reproduction-20260629T171224Z-3-001\Paper1_Full_Pipeline_FINAL.ipynb'

with open(NB_PATH, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# ============================================================
# FIX: Cell 42 - CESPDataset with RAM caching
# Problem: Every __getitem__ call does cv2.imread from disk
#   = 3.4 MILLION disk reads for 10-fold x 50 epochs x 215 batches
# Solution: Load ALL images into RAM once in __init__
#   then __getitem__ just reads from memory (instant)
# ============================================================

new_cell42 = (
'# ============================================================\n'
'# CELL 32: CESP Dataset with RAM Caching (SPEED FIX)\n'
'# Without cache: 3.4M disk reads -> takes HOURS\n'
'# With cache: images loaded ONCE into RAM -> fast training\n'
'# ============================================================\n'
'class CESPDataset(Dataset):\n'
'    def __init__(self, preictal_paths, interictal_paths, synthetic_paths=None, cache=True):\n'
'        self.files = []\n'
'        self.labels = []\n'
'\n'
'        # Real preictal = 1\n'
'        for p in preictal_paths:\n'
'            self.files.append(p)\n'
'            self.labels.append(1.0)\n'
'\n'
'        # Real interictal = 0\n'
'        for p in interictal_paths:\n'
'            self.files.append(p)\n'
'            self.labels.append(0.0)\n'
'\n'
'        # Synthetic preictal = 1 (augmentation)\n'
'        if synthetic_paths:\n'
'            for p in synthetic_paths:\n'
'                self.files.append(p)\n'
'                self.labels.append(1.0)\n'
'\n'
'        self.transform = transforms.Compose([\n'
'            transforms.ToPILImage(),\n'
'            transforms.Resize((128, 128)),\n'
'            transforms.ToTensor(),\n'
'            transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))\n'
'        ])\n'
'\n'
'        # --- RAM CACHE: load all images ONCE into memory ---\n'
'        self.cache = cache\n'
'        self.cached_imgs = None\n'
'        if cache and len(self.files) > 0:\n'
'            print(f"      Caching {len(self.files)} images into RAM...", end=" ", flush=True)\n'
'            self.cached_imgs = []\n'
'            for f in self.files:\n'
'                img = cv2.imread(str(f))\n'
'                if img is None:\n'
'                    img = np.zeros((128, 128, 3), dtype=np.uint8)\n'
'                else:\n'
'                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)\n'
'                    img = cv2.resize(img, (128, 128))\n'
'                self.cached_imgs.append(img)\n'
'            print("Done!")\n'
'\n'
'    def __len__(self):\n'
'        return len(self.files)\n'
'\n'
'    def __getitem__(self, idx):\n'
'        if self.cached_imgs is not None:\n'
'            img = self.cached_imgs[idx]   # RAM access - instant!\n'
'        else:\n'
'            img = cv2.imread(str(self.files[idx]))\n'
'            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)\n'
'        return self.transform(img), torch.tensor(self.labels[idx], dtype=torch.float32)\n'
'\n'
'print("CESPDataset defined with RAM caching + 128x128 resize.")\n'
)

# ============================================================
# FIX: Cell 41 - CESP model update for 128x128 input
# 128 -> 64 -> 32 -> 16 after 3x MaxPool2d(2,2)
# So FC: 64 * 16 * 16 = 16384 -> 32 -> 1
# ============================================================
new_cell41 = (
'# ============================================================\n'
'# CELL 31: CESP Architecture - updated for 128x128 input\n'
'# 3 Conv blocks (126, 64, 64 filters)\n'
'# 128->64->32->16 after MaxPool, so FC: 64*16*16=16384->32->1\n'
'# ============================================================\n'
'class CESPBlock(nn.Module):\n'
'    def __init__(self, in_ch, out_ch):\n'
'        super().__init__()\n'
'        self.block = nn.Sequential(\n'
'            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),\n'
'            nn.BatchNorm2d(out_ch),\n'
'            nn.ReLU(inplace=True),\n'
'            nn.MaxPool2d(2, 2)\n'
'        )\n'
'    def forward(self, x):\n'
'        return self.block(x)\n'
'\n'
'class CESP(nn.Module):\n'
'    def __init__(self):\n'
'        super().__init__()\n'
'        self.features = nn.Sequential(\n'
'            CESPBlock(3, 126),   # 128->64\n'
'            CESPBlock(126, 64),  # 64->32\n'
'            CESPBlock(64, 64),   # 32->16\n'
'        )\n'
'        # 64 * 16 * 16 = 16,384 (for 128x128 input)\n'
'        self.classifier = nn.Sequential(\n'
'            nn.Flatten(),\n'
'            nn.Linear(64 * 16 * 16, 32),\n'
'            nn.ReLU(inplace=True),\n'
'            nn.Dropout(0.3),\n'
'            nn.Linear(32, 1)\n'
'        )\n'
'\n'
'    def forward(self, x):\n'
'        x = self.features(x)\n'
'        return self.classifier(x).squeeze(-1)\n'
'\n'
'cesp_test = CESP().to(device)\n'
'with torch.no_grad():\n'
'    x_test = torch.randn(2, 3, 128, 128, device=device)\n'
'    out = cesp_test(x_test)\n'
'print("CESP output shape:", out.shape)\n'
'print("CESP total params:", f\'{sum(p.numel() for p in cesp_test.parameters()):,}\')\n'
'del cesp_test\n'
'print("CESP model defined for 128x128 input.")\n'
)

nb['cells'][41]['source'] = [new_cell41]
nb['cells'][41]['execution_count'] = None
nb['cells'][41]['outputs'] = []

nb['cells'][42]['source'] = [new_cell42]
nb['cells'][42]['execution_count'] = None
nb['cells'][42]['outputs'] = []

with open(NB_PATH, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("[OK] Cell 41 (CESP model) updated for 128x128 input")
print("[OK] Cell 42 (CESPDataset) updated with RAM caching + 128x128")
print()
print("SPEED COMPARISON:")
print("  Before: ~3.4 million disk reads -> HOURS")
print("  After:  1 disk read per image at start, then RAM -> MINUTES")
print()
print("NOW IN JUPYTER - run these cells in order:")
print("  1. Stop running cell (square stop button)")
print("  2. Run Cell 41 (CESP model - 128x128 version)")
print("  3. Run Cell 42 (CESPDataset - with RAM cache)")
print("  4. Run Cell 44 (train_cesp - already fixed)")
print("  5. Run Cell 46 (Experiment 2 TSTR)")
print()
print("First fold will say 'Caching 6845 images...' - wait 2-3 mins.")
print("After that all 10 folds run fast!")
