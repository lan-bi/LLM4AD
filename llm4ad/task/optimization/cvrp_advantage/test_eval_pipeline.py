#!/usr/bin/env python3
"""End-to-end test for the CVRP advantage function evaluation pipeline."""
import sys, os, re, json, textwrap, traceback

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_POMO_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, '../../../../../POMO/NEW_py_ver'))
sys.path.insert(0, _POMO_ROOT)
sys.path.insert(0, os.path.join(_POMO_ROOT, 'CVRP'))
sys.path.insert(0, os.path.join(_POMO_ROOT, 'CVRP', 'POMO'))
sys.path.insert(0, os.path.abspath(os.path.join(_SCRIPT_DIR, '../../../../')))

import torch
from llm4ad.base import TextFunctionProgramConverter

STATE_DIR = '/public/home/qinjz/lanl7_files/logs/online_eoh'

def _build_updated_template(fn_source, design_insights):
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
        "import torch\n\n"
        "def compute_advantage(reward, load, at_the_depot, finished, "
        "loss_ema, reward_ema, epoch):\n"
        '    """\n'
        "    Compute advantage for REINFORCE training of CVRP.\n\n"
        "    Args:\n"
        "        reward:       (batch, pomo) float — negative route distance\n"
        "        load:         (batch, pomo) float — remaining capacity [0,1]\n"
        "        at_the_depot: (batch, pomo) bool  — last step at depot\n"
        "        finished:     (batch, pomo) bool  — all customers visited\n"
        "        loss_ema:     float — EMA of recent loss\n"
        "        reward_ema:   float — EMA of recent reward\n"
        "        epoch:        int   — current training epoch\n\n"
        "    Returns:\n"
        "        advantage:    (batch, pomo) float tensor\n"
        '    """\n'
    )
    parts = [header]
    if design_insights:
        parts.append("    # -- design insights --")
        for line in design_insights.strip().splitlines():
            parts.append(f"    # {line.strip()}")
        parts.append("")
    parts.append(body)
    return '\n'.join(parts)

def _compile_function(fn_source, template):
    try:
        program = TextFunctionProgramConverter.function_to_program(fn_source, template)
        if program is None:
            return None, "function_to_program returned None"
        callables = program.exec()
        if not callables:
            return None, "program.exec() returned empty list"
        return callables[0], None
    except Exception as e:
        return None, str(e)

# --- Load state ---
state = json.load(open(f'{STATE_DIR}/full_state.json'))
fn_source = state.get('advantage_fn_source')
if not fn_source:
    fn_source = "def compute_advantage(reward, load, at_the_depot, finished, loss_ema, reward_ema, epoch):\n    return reward - reward.float().mean(dim=1, keepdims=True)"
ref = json.load(open(f'{STATE_DIR}/reflections.json')) if os.path.exists(f'{STATE_DIR}/reflections.json') else {}
design = ref.get('design_lessons', '')

tmpl = _build_updated_template(fn_source, design)

# Test 1: parse template
print("Test 1: parse template...")
fn_parsed = TextFunctionProgramConverter.text_to_function(tmpl)
print(f"  {'OK' if fn_parsed else 'FAIL'}: {fn_parsed.name if fn_parsed else 'None'}")

# Test 2: compile simple candidate
print("\nTest 2: compile simple candidate...")
simple = "def compute_advantage(reward, load, at_the_depot, finished, loss_ema, reward_ema, epoch):\n    return reward - reward.float().mean(dim=1, keepdims=True)"
fn, err = _compile_function(simple, tmpl)
print(f"  {'OK' if fn else 'FAIL'}: {err or 'compiled'}")

# Test 3: compile default
print("\nTest 3: compile default advantage...")
default = "def compute_advantage(reward, load, at_the_depot, finished, loss_ema, reward_ema, epoch):\n    return reward - reward.float().mean(dim=1, keepdims=True)"
fn, err = _compile_function(default, tmpl)
print(f"  {'OK' if fn else 'FAIL'}: {err or 'compiled'}")

# Test 4: compile real candidates from latest EoH
print("\nTest 4: compile real EoH candidates...")
eoh_dirs = sorted([d for d in os.listdir(STATE_DIR) if d.startswith('eoh_epoch') and os.path.isdir(f'{STATE_DIR}/{d}')])
ok, fail = 0, 0
for edir in eoh_dirs[-3:]:  # last 3 epochs
    eoh_path = f'{STATE_DIR}/{edir}'
    for root, dirs, files in os.walk(eoh_path):
        for fname in files:
            if fname.endswith('.json') and 'samples' in root:
                data = json.load(open(os.path.join(root, fname)))
                for s in data[:5]:
                    fn_str = s.get('function', '')
                    if not fn_str: continue
                    fn2, err2 = _compile_function(fn_str, tmpl)
                    if fn2: ok += 1
                    else: fail += 1
                break
        break
print(f"  OK={ok} FAIL={fail}")

# Test 5: Full evaluation (compile + train_n_batches)
print("\nTest 5: Full evaluation pipeline...")
from CVRPTrainer import CVRPTrainer as Trainer
from utils.utils import set_result_folder
from llm4ad.task.optimization.cvrp_advantage import CVRPAdvantageEvaluation

ckpt_path = f'{STATE_DIR}/resume_checkpoint.pt'
if not os.path.exists(ckpt_path):
    print(f"  SKIP: checkpoint not found")
else:
    eval_obj = CVRPAdvantageEvaluation(timeout_seconds=120)
    eval_obj.template_program = tmpl
    eval_obj.set_context(ckpt_path, 100, 100, '/tmp/eval_err.log', 300)

    # Compile default function through evaluation
    default_fn = _compile_function(
        "def compute_advantage(reward, load, at_the_depot, finished, loss_ema, reward_ema, epoch):\n    return reward - reward.float().mean(dim=1, keepdims=True)",
        tmpl)
    if default_fn[0] is None:
        print(f"  FAIL: can't compile default: {default_fn[1]}")
    else:
        score = eval_obj.evaluate_program("test", default_fn[0])
        print(f"  Score: {score}")
        if score is None:
            print("  FAIL: evaluation returned None")
            if os.path.exists('/tmp/eval_err.log'):
                with open('/tmp/eval_err.log') as f:
                    err = f.read()
                print(f"  Error log ({len(err)} chars):")
                print(err[-1000:])

# Test 6: GPU
print(f"\nTest 6: GPU available={torch.cuda.is_available()}, devices={torch.cuda.device_count()}")

print("\n=== SUMMARY ===")
all_ok = fn_parsed is not None and fn is not None
print(f"{'ALL PASSED' if all_ok else 'SOME FAILED'}")
