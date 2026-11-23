"""
evaluation/pseudo_norm_paths_test.py

Regression test: every evaluation path must run the same forward pass the mapper was
trained with, including the fixed-norm projection of unit-norm runs,

    z <- rho * z / ||z||        (cfg.pseudo_norm = rho; 0.0 = off)

which model.py applies in both forward() (training) and encode_query() (inference).
The model reads rho once, from its config. So an evaluation silently runs a *different*
function (the raw, never-trained token norm reaches the text encoder) if either
  (1) a loader rebuilds the config from a checkpoint without forwarding `pseudo_norm`, or
  (2) a script builds the pseudo-token by hand (mapper -> merger -> text encoder) and
      skips the projection.
Both happened before this test existed: the paper's unit-norm row was evaluated
without the projection.

Static (AST) checks only, so it needs no torch, GPU, data or checkpoints.

Run (from codes/):
    python evaluation/pseudo_norm_paths_test.py
    python evaluation/pseudo_norm_paths_test.py <codes_root>    # scan another tree
"""

import ast
import sys
from pathlib import Path

ROOT = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent

SKIP_DIRS  = {"analysis_notebooks", "__pycache__", ".ipynb_checkpoints"}
SKIP_FILES = {"model_test.py", "train_test.py",   # unit tests build configs by hand
              "model_old.py"}                     # superseded model file, imported nowhere
CONFIG_BUILDERS = {"load_from_yaml", "get_config", "Config"}

# Paths the scan must find, so a rename can never turn this test into a vacuous pass.
REQUIRED_LOADER_FILES = {
    "evaluation/eval_circo.py", "evaluation/eval_circo_v2.py", "evaluation/eval_circo_test.py",
    "evaluation/eval_cirr.py", "evaluation/eval_fashion.py", "evaluation/eval_fashion_v2.py",
}
REQUIRED_TOKEN_PATHS = {("model/model.py", "forward"), ("model/model.py", "encode_query")}

# ---------------------------------------------------------------------------
# Test runner
# ---------------------------------------------------------------------------

PASS  = 0
FAIL  = 0

def run(name, fn):
    global PASS, FAIL
    try:
        fn()
        print(f"  PASS  {name}")
        PASS += 1
    except Exception as e:
        print(f"  FAIL  {name}")
        for line in str(e).splitlines():
            print(f"        {line}")
        FAIL += 1

# ---------------------------------------------------------------------------
# AST scan (built once)
# ---------------------------------------------------------------------------

def _call_name(call):
    f = call.func
    return f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", None)


def _is_method_call(node, name):
    return (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            and node.func.attr == name)


def _reads_ckpt_cfg(call):
    """True if a keyword value reads a checkpoint config, i.e. contains `<name>.get(...)`."""
    for kw in call.keywords:
        for sub in ast.walk(kw.value):
            if _is_method_call(sub, "get") and isinstance(sub.func.value, ast.Name):
                return True
    return False


def _mentions(node, attr):
    return any(isinstance(n, ast.Attribute) and n.attr == attr for n in ast.walk(node))


def _scan(root):
    loaders, token_paths = [], []
    for path in sorted(root.rglob("*.py")):
        if SKIP_DIRS & set(path.parts) or path.name in SKIP_FILES:
            continue
        rel  = path.relative_to(root).as_posix()
        tree = ast.parse(path.read_text(), filename=str(path))

        # Config builders inside a function that torch.load()s a checkpoint are loaders,
        # even when the checkpoint fields were first copied into local variables.
        in_ckpt_fn = set()
        for fn in ast.walk(tree):
            if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                calls = [n for n in ast.walk(fn) if isinstance(n, ast.Call)]
                if any(_is_method_call(c, "load") and getattr(c.func.value, "id", None) == "torch"
                       for c in calls):
                    in_ckpt_fn.update(id(c) for c in calls)

        for node in ast.walk(tree):
            if (isinstance(node, ast.Call) and _call_name(node) in CONFIG_BUILDERS
                    and (_reads_ckpt_cfg(node) or id(node) in in_ckpt_fn)):
                kw = next((k for k in node.keywords if k.arg == "pseudo_norm"), None)
                loaders.append((rel, node.lineno, kw))

            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                mapper = [n.lineno for n in ast.walk(node) if _is_method_call(n, "mapper")]
                merger = [n.lineno for n in ast.walk(node) if _is_method_call(n, "merger")]
                build  = [n.lineno for n in ast.walk(node) if _is_method_call(n, "_build_token_sequence")]
                if mapper and build:
                    gates = [n for n in ast.walk(node)
                             if isinstance(n, ast.If) and _mentions(n.test, "pseudo_norm")]
                    token_paths.append((rel, node.name, node.lineno,
                                        max(mapper + merger), min(build), gates))
    return loaders, token_paths


LOADERS, TOKEN_PATHS = _scan(ROOT)

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_scan_finds_known_paths():
    found_loaders = {rel for rel, _, _ in LOADERS}
    found_paths   = {(rel, name) for rel, name, *_ in TOKEN_PATHS}
    missing = sorted(REQUIRED_LOADER_FILES - found_loaders) + \
              [f"{r}:{n}()" for r, n in sorted(REQUIRED_TOKEN_PATHS - found_paths)]
    assert not missing, "scan did not find (renamed? test would pass vacuously):\n" + "\n".join(missing)


def test_loaders_forward_pseudo_norm():
    """Each checkpoint loader passes pseudo_norm=<cfg>.get('pseudo_norm', 0.0).
    The literal 0.0 fallback is the only correct one: the key is absent exactly in
    checkpoints trained before the feature existed, i.e. without the projection."""
    bad = []
    for rel, line, kw in LOADERS:
        if kw is None:
            bad.append(f"{rel}:{line}: config built from a checkpoint without pseudo_norm")
            continue
        v = kw.value
        ok = (_is_method_call(v, "get") and len(v.args) == 2
              and isinstance(v.args[0], ast.Constant) and v.args[0].value == "pseudo_norm"
              and isinstance(v.args[1], ast.Constant) and v.args[1].value == 0.0)
        if not ok:
            bad.append(f"{rel}:{line}: pseudo_norm must be <cfg>.get('pseudo_norm', 0.0), "
                       f"got {ast.unparse(v)}")
    assert not bad, "\n".join(bad)


def test_hand_built_tokens_apply_projection():
    """Token built by hand: the projection sits after mapper/merger, before the sequence
    is built: the same order as model.py forward() and encode_query()."""
    bad = [f"{rel}:{line}: {name}() feeds the mapper output to _build_token_sequence "
           f"without the pseudo_norm projection in between"
           for rel, name, line, last_map, first_build, gates in TOKEN_PATHS
           if not any(last_map < g.lineno < first_build for g in gates)]
    assert not bad, "\n".join(bad)


def test_projection_not_training_only():
    """The projection is a forward-pass edit, applied at test time too (unlike additive
    noise, a training-only regularizer). Its gate must therefore not depend on .training,
    and it must actually normalize the token."""
    bad = []
    for rel, name, line, last_map, first_build, gates in TOKEN_PATHS:
        for g in gates:
            if not (last_map < g.lineno < first_build):
                continue
            if _mentions(g.test, "training"):
                bad.append(f"{rel}:{g.lineno}: {name}() gates the projection on .training")
            if not any(_is_method_call(n, "normalize") for b in g.body for n in ast.walk(b)):
                bad.append(f"{rel}:{g.lineno}: {name}() pseudo_norm branch does not normalize")
    assert not bad, "\n".join(bad)


TESTS = [
    ("scan   | known loaders and token paths found", test_scan_finds_known_paths),
    ("loader | checkpoint pseudo_norm forwarded",    test_loaders_forward_pseudo_norm),
    ("tokens | hand-built paths apply projection",   test_hand_built_tokens_apply_projection),
    ("tokens | projection not training-only",        test_projection_not_training_only),
]


if __name__ == "__main__":
    print(f"{'='*65}")
    print(f"  pseudo_norm_paths_test.py  —  {len(TESTS)} tests")
    print(f"  scanned {ROOT}")
    print(f"  {len(LOADERS)} checkpoint loaders, {len(TOKEN_PATHS)} token-building functions")
    print(f"{'='*65}")

    for name, fn in TESTS:
        run(name, fn)

    print(f"\n{'='*65}")
    print(f"  {PASS} passed   {FAIL} failed   {PASS + FAIL} total")
    print(f"{'='*65}\n")

    sys.exit(0 if FAIL == 0 else 1)
