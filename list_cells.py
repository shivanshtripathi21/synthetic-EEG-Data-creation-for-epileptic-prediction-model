import json

with open('Paper1_Full_Pipeline_FINAL.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

cells = nb['cells']
for i, c in enumerate(cells):
    if c.get('cell_type') == 'code':
        src = ''.join(c.get('source', []))
        # Remove non-ascii for safe printing
        preview = src.strip()[:90].encode('ascii','replace').decode()
        ec = c.get('execution_count','?')
        print(f'Cell {i:2d} [exec={str(ec):>4}]: {preview}')
