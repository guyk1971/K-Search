"""k_search.eval — evaluation utilities for local and remote kernel evaluation."""

from k_search.eval.remote_eval_package import RemoteEvalPackage, parse_driver_output
from k_search.eval.remote_evaluator import RemoteEvaluator

__all__ = [
    "RemoteEvalPackage",
    "RemoteEvaluator",
    "parse_driver_output",
]
