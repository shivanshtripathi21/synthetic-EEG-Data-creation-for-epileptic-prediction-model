import json

nb_path = 'Paper1_Full_Pipeline_FINAL.ipynb'
with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for c in nb['cells']:
    if c['cell_type'] == 'code':
        new_source = []
        for line in c['source']:
            # Fix the unexpected indent
            if line.startswith('    trtr_fpr  ='):
                line = line.replace('    trtr_fpr  =', 'trtr_fpr  =')
            elif line.startswith('    tstr_fpr  ='):
                line = line.replace('    tstr_fpr  =', 'tstr_fpr  =')
            elif line.startswith('    trts_fpr  ='):
                line = line.replace('    trts_fpr  =', 'trts_fpr  =')
            elif line.startswith('    tsts_fpr  ='):
                line = line.replace('    tsts_fpr  =', 'tsts_fpr  =')
            new_source.append(line)
        c['source'] = new_source

with open(nb_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)
print("Indentation fixed successfully!")
