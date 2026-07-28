import json

NB_PATH = r'c:\Users\SHIVA\Downloads\Paper1_Reproduction-20260629T171224Z-3-001\Paper1_Full_Pipeline_FINAL.ipynb'
with open(NB_PATH, 'r', encoding='utf-8') as f:
    nb = json.load(f)

print("=== UPDATED CELL 46 (Experiment 2) ===")
print(''.join(nb['cells'][46]['source']))
