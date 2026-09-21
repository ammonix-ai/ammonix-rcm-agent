"""Checker for the v4-fixes milestone (post-v0.3 platform hardening).

Verifies the three fixes without touching any shipped artefact:
1. uncertainty: fold-ensemble spread exists end to end (UniverseRecord.
   scores_std, SwarmResult.fold_models, score_live fold_ensemble,
   uncertainty-aware ambiguity in retrieve, cardessa score_spread);
2. hygiene: strict payload validator (unknown keywords raise), single
   balanced-weights implementation, config-sourced thresholds, loud
   errors instead of asserts;
3. genericity: no domain tokens in the generic runtime, ClassifierSpec
   model choice implemented, NaN/bool feature handling.

Plus the standing gates: pytest, ruff and the determinism check must all
be green. Prints a one-line JSON verdict as its final act.
"""

import inspect
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "ammonix_core"))


def run(args: list[str]) -> int:
    return subprocess.run(args, cwd=ROOT).returncode


def main() -> int:
    checks: dict[str, bool] = {}

    pytest_exit = run([sys.executable, "-m", "pytest", "-q"])
    ruff_exit = run([sys.executable, "-m", "ruff", "check", "."])
    determinism_exit = run([sys.executable, "scripts/determinism_check.py"])
    checks["pytest"] = pytest_exit == 0
    checks["ruff"] = ruff_exit == 0
    checks["determinism"] = determinism_exit == 0

    # -- fix 1: uncertainty machinery -----------------------------------
    from ammonix_core.schema import UniverseRecord

    from ammonix_core import pipeline, runtime, universe

    checks["universe_scores_std_field"] = "scores_std" in UniverseRecord.model_fields
    checks["swarm_keeps_fold_models"] = "fold_models" in {
        f.name for f in __import__("dataclasses").fields(pipeline.SwarmResult)
    }
    checks["fold_ensemble_inference"] = callable(getattr(pipeline, "score_live", None))
    checks["retrieve_uses_spread"] = "scores_std" in inspect.getsource(runtime.retrieve)
    from cardessa import harness

    checks["cardessa_score_spread"] = callable(getattr(harness, "score_spread", None))
    checks["basis_persists_folds"] = "fold_" in (
        (ROOT / "scripts" / "build_basis.py").read_text(encoding="utf-8")
    )

    # -- fix 2: hygiene ---------------------------------------------------
    try:
        runtime.validate_payload({"type": "object", "oneOf": []}, {})
        checks["validator_rejects_unknown"] = False
    except NotImplementedError:
        checks["validator_rejects_unknown"] = True
    checks["one_balanced_weights"] = universe.balanced_weights is pipeline.balanced_weights
    checks["thresholds_from_config"] = (
        "AmmonixConfig()" in inspect.getsource(runtime)
        and "AmmonixConfig()" in inspect.getsource(universe)
    )
    import re

    core_src = "".join(
        (ROOT / "ammonix_core" / "ammonix_core" / f"{m}.py").read_text(encoding="utf-8")
        for m in ("pipeline", "universe")
    )
    # verifier finding (v4-fixes): match ANY assert statement, not one literal
    checks["no_bare_asserts_on_scores"] = not re.search(
        r"^\s*assert\b", core_src, flags=re.MULTILINE
    )

    # -- fix 3: genericity ------------------------------------------------
    runtime_src = inspect.getsource(runtime)
    checks["runtime_domain_free"] = not any(
        token in runtime_src for token in ("cpt", "carc", "payer_correspondence")
    )
    from ammonix_core.schema import ClassifierSpec

    try:
        pipeline.build_classifier(0, ClassifierSpec(model="logreg"))
        pipeline.build_classifier(0, ClassifierSpec(model="gbdt"))
        checks["classifier_spec_implemented"] = True
    except Exception:
        checks["classifier_spec_implemented"] = False
    import numpy as np

    x = pipeline.matrix_from_rows([{"a": None, "b": True}], ["a", "b"])
    checks["missing_and_bool_features"] = bool(np.isnan(x[0, 0]) and x[0, 1] == 1.0)
    checks["cardessa_owns_prompt_fields"] = callable(
        getattr(harness, "prompt_fields", None)
    )

    verdict = all(checks.values())
    print(json.dumps({"loop": "V4F", "verdict": verdict, **checks}))
    return 0 if verdict else 1


if __name__ == "__main__":
    raise SystemExit(main())
