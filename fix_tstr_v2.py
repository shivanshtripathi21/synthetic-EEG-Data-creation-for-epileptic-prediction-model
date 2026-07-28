import json, re

NB_PATH = r'c:\Users\SHIVA\Downloads\Paper1_Reproduction-20260629T171224Z-3-001\Paper1_Full_Pipeline_FINAL.ipynb'

with open(NB_PATH, 'r', encoding='utf-8') as f:
    nb = json.load(f)

src = ''.join(nb['cells'][46]['source'])

# Change epochs=25 to epochs=50 everywhere in cell 46
src = src.replace('epochs=25', 'epochs=50')
# Update the print line too
src = src.replace('epochs=25', 'epochs=50')
src = src.replace("anchor_ratio=0.3, threshold=0.35, epochs=25", 
                  "anchor_ratio=0.3, threshold=0.35, epochs=50")

nb['cells'][46]['source'] = [src]
nb['cells'][46]['execution_count'] = None
nb['cells'][46]['outputs'] = []

with open(NB_PATH, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

# Verify
src2 = ''.join(nb['cells'][46]['source'])
matches = re.findall(r'epochs=\d+', src2)
print('Epochs in cell 46:', matches)
print('[OK] epochs=50 set!')
