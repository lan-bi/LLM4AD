#!/usr/bin/env python3
"""Debug: test template parsing for epoch 240 function."""
import sys,os,re,json,textwrap,ast
sys.path.insert(0,os.path.abspath(os.path.join(os.path.dirname(__file__),'../../../../')))
from llm4ad.base import TextFunctionProgramConverter

def _build_updated_template(fn_source, design_insights=''):
    text = fn_source.strip()
    def_match = re.search(r'^def \w+\s*\([^)]*\)\s*:\s*\n', text)
    body = text[def_match.end():]
    body = re.sub(r'^\s*"""[\s\S]*?"""\s*\n?', '', body)
    body = re.sub(r"^\s*'''[\s\S]*?'''\s*\n?", '', body)
    lines = body.strip('\n').splitlines()
    indented = []
    for l in lines:
        s = l.strip()
        indented.append('    ' + s if s else '')
    body = '\n'.join(indented)
    header = (
        'import torch\n\n'
        'def compute_advantage(reward,load,at_the_depot,finished,loss_ema,reward_ema,epoch):\n'
        '    """\n    Compute advantage.\n    """\n'
    )
    parts = [header]
    if design_insights:
        for line in design_insights.strip().splitlines():
            parts.append(f'    # {line.strip()}')
        parts.append('')
    parts.append(body)
    return '\n'.join(parts)

STATE_DIR = '/public/home/qinjz/lanl7_files/logs/online_eoh'
state = json.load(open(f'{STATE_DIR}/full_state.json'))
fn_source = state.get('advantage_fn_source','')
tmpl = _build_updated_template(fn_source)

# Save template
with open(f'{STATE_DIR}/debug_template.py','w') as f:
    f.write(tmpl)
print('Template saved')

# Try ast
try:
    ast.parse(tmpl)
    print('ast.parse OK')
except SyntaxError as e:
    print(f'ast.parse FAIL: {e}')
    # Show lines around error
    lines = tmpl.splitlines()
    lineno = e.lineno - 1
    lo = max(0, lineno-2)
    hi = min(len(lines), lineno+3)
    for i in range(lo, hi):
        mark = '>>>' if i == lineno else '   '
        print(f'{mark} {i+1}: {lines[i]}')
