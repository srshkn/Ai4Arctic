"""Smoke-test: проверяет, что проект корректно собран."""
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch


def test(name, fn):
    try:
        r = fn()
        msg = f" — {r}" if isinstance(r, str) else ""
        print(f"  PASS: {name}{msg}")
        return True
    except Exception as e:
        print(f"  FAIL: {name}")
        print(f"    {type(e).__name__}: {e}")
        return False


results = []

print("\n[1/4] ИМПОРТЫ src/")
def imp(modname, items):
    def _f():
        mod = __import__(f'src.{modname}', fromlist=items)
        for it in items:
            getattr(mod, it)
        return f"{len(items)} объектов"
    return _f

results.append(test("src.model", imp('model', ['ConvLSTMCell', 'ConvLSTMNet'])))
results.append(test("src.data", imp('data', ['make_pairs', 'TileDataset'])))
results.append(test("src.inference", imp('inference', ['predict_full_map'])))
results.append(test("src.metrics", imp('metrics', ['all_metrics'])))
results.append(test("src.landcover", imp('landcover', ['classify_landcover'])))
results.append(test("src.bias_correction", imp('bias_correction', ['compare_strategies'])))
results.append(test("src.ablation", imp('ablation', ['FEATURE_NAMES'])))
results.append(test("src.reproducibility", imp('reproducibility', ['set_global_seed', 'DEFAULT_SEED'])))
results.append(test("src.data_extended", imp('data_extended', ['make_sliding_windows'])))
results.append(test("src.esa_cci", imp('esa_cci', ['build_esa_target_tensor'])))


print("\n[2/4] ENV-DETECTION")
def env_local():
    in_colab = 'COLAB_RELEASE_TAG' in os.environ or 'COLAB_GPU' in os.environ
    return "Colab VM" if in_colab else "Local runtime"
results.append(test("env detection", env_local))


print("\n[3/4] MODEL FORWARD")
def smoke_model():
    from src.model import ConvLSTMNet
    try:
        model = ConvLSTMNet(in_channels=20, hidden_ch=8, dropout=0.0)
    except TypeError:
        try:
            model = ConvLSTMNet(in_ch=20, hidden_ch=8, dropout=0.0)
        except TypeError:
            model = ConvLSTMNet()
    x = torch.randn(2, 4, 20, 16, 16)
    y = model(x)
    return f"output shape {tuple(y.shape)}"
results.append(test("model forward", smoke_model))


print("\n[4/4] NOTEBOOKS")
import ast, json, re

def check_nb(path):
    with open(path) as f:
        nb = json.load(f)
    errors = []

    def transform(ln):
        s = ln.lstrip()
        if s.startswith('!') or s.startswith('%'):
            indent = ln[:len(ln) - len(s)]
            return f"{indent}pass  # magic: {s}"
        return ln

    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] != 'code':
            continue
        src = cell['source']
        if isinstance(src, list):
            src = ''.join(src)
        cleaned = '\n'.join(transform(ln) for ln in src.split('\n'))
        try:
            ast.parse(cleaned)
        except SyntaxError as e:
            errors.append(f"cell {i}: {e}")
        if re.search(r'\bIN_COLAB\b(?!_)', src):
            errors.append(f"cell {i}: устаревшая IN_COLAB")
    if errors:
        raise SyntaxError(errors[0])
    n_code = sum(1 for c in nb['cells'] if c['cell_type'] == 'code')
    return f"{n_code} code cells OK"


nb_dir = PROJECT_ROOT / 'notebooks'
for nb_file in sorted(nb_dir.glob('*.ipynb')):
    results.append(test(nb_file.name, lambda p=nb_file: check_nb(p)))


print(f"\nИТОГ: {sum(results)}/{len(results)} тестов прошло")
if sum(results) == len(results):
    print("Всё хорошо.")
else:
    print(f"FAIL: {len(results) - sum(results)} тестов упало.")
