import json, sys
sys.stdout.reconfigure(encoding='utf-8')

with open('google_colab.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

cells = nb.get('cells', [])
print(f'Total cells: {len(cells)}')
for i, cell in enumerate(cells):
    outputs = cell.get('outputs', [])
    if not outputs:
        continue
    print(f'\n=== Cell {i} outputs ===')
    for o in outputs:
        otype = o.get('output_type', '')
        text = o.get('text', [])
        if isinstance(text, list): text = ''.join(text)
        if not text:
            text = ''.join(o.get('data', {}).get('text/plain', []))
        print(str(text)[:1000])
