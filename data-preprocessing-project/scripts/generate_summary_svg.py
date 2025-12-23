import os
import pandas as pd

base = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
cleaned = os.path.join(base, 'data', 'cleaned_data.csv')
outdir = os.path.join(base, 'docs', 'figures')
os.makedirs(outdir, exist_ok=True)

df = pd.read_csv(cleaned)

# compute survival rate by Pclass and Sex
agg = df.groupby(['Pclass', 'Sex'])['Survived'].mean().unstack()
agg = agg.sort_index()

# simple SVG bar chart
width = 600
height = 200
margin = 40
bar_h = 20
gap = 10

max_val = agg.values.max()
scale = (width - 2*margin) / 1.0

rows = []
for i, pclass in enumerate(agg.index):
    male = agg.loc[pclass, 'male'] if 'male' in agg.columns else 0
    female = agg.loc[pclass, 'female'] if 'female' in agg.columns else 0
    rows.append((pclass, 'male', male))
    rows.append((pclass, 'female', female))

svg_lines = [f'<?xml version="1.0" encoding="UTF-8"?>', f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">', '<style>.label{font:12px sans-serif}</style>']

y = margin
for idx, (pclass, sex, val) in enumerate(rows):
    bar_w = val * scale
    color = '#2b7cff' if sex == 'male' else '#ff6b6b'
    svg_lines.append(f'<rect x="{margin}" y="{y}" width="{bar_w:.1f}" height="{bar_h}" fill="{color}" />')
    svg_lines.append(f'<text x="{margin+bar_w+5}" y="{y+bar_h-6}" class="label">P{pclass} {sex} ({val:.2f})</text>')
    y += bar_h + gap

svg_lines.append('</svg>')

out = os.path.join(outdir, 'survival_by_pclass_sex.svg')
with open(out, 'w', encoding='utf-8') as f:
    f.write('\n'.join(svg_lines))

print('Wrote', out)