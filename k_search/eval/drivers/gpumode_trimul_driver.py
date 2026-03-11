"""Self-contained GPUMode TriMul driver template for remote evaluation.

Generates a complete Python script that can run on a remote GPU server
with only PyTorch as a dependency.  The script imports the submission's
``custom_kernel`` from a co-located ``submission.py``, runs correctness
tests and benchmarks, then prints a JSON result between markers.
"""

from __future__ import annotations

import json
import textwrap
from typing import Any


# Marker strings used to delimit JSON output in driver stdout.
RESULT_JSON_START = "===KSEARCH_RESULT_JSON_START==="
RESULT_JSON_END = "===KSEARCH_RESULT_JSON_END==="


def build_gpumode_driver_source(
    *,
    test_specs: list[dict[str, Any]],
    benchmark_specs: list[dict[str, Any]],
) -> str:
    """Return a self-contained Python driver script as a string.

    Parameters
    ----------
    test_specs:
        List of dicts, each containing keyword arguments for
        ``generate_input`` (e.g. seqlen, bs, dim, hiddendim, seed,
        nomask, distribution).  Used for correctness checking.
    benchmark_specs:
        Same format as *test_specs* but used for timing.

    Returns
    -------
    str
        Complete Python source that, when executed with a co-located
        ``submission.py`` defining ``custom_kernel``, will print a
        JSON result between ``===KSEARCH_RESULT_JSON_START===`` and
        ``===KSEARCH_RESULT_JSON_END===`` markers.
    """
    test_specs_json = json.dumps(test_specs)
    benchmark_specs_json = json.dumps(benchmark_specs)

    source = textwrap.dedent(f"""\
        #!/usr/bin/env python3
        \"\"\"Auto-generated GPUMode TriMul evaluation driver.

        This script is self-contained: it embeds the reference implementation,
        correctness checker, and benchmark harness so it can run on a remote
        GPU server with only PyTorch installed.
        \"\"\"
        import json
        import math
        import sys
        import time
        import traceback
        from typing import Tuple

        import torch
        from torch import nn, einsum

        # ---------------------------------------------------------------------------
        # Embedded test / benchmark specs
        # ---------------------------------------------------------------------------
        TEST_SPECS = {test_specs_json}
        BENCHMARK_SPECS = {benchmark_specs_json}

        RESULT_JSON_START = "{RESULT_JSON_START}"
        RESULT_JSON_END = "{RESULT_JSON_END}"

        # ---------------------------------------------------------------------------
        # DisableCuDNNTF32 context manager  (from utils.py)
        # ---------------------------------------------------------------------------
        class DisableCuDNNTF32:
            def __init__(self):
                self.allow_tf32 = torch.backends.cudnn.allow_tf32
                self.deterministic = torch.backends.cudnn.deterministic

            def __enter__(self):
                torch.backends.cudnn.allow_tf32 = False
                torch.backends.cudnn.deterministic = True
                return self

            def __exit__(self, exc_type, exc_value, traceback):
                torch.backends.cudnn.allow_tf32 = self.allow_tf32
                torch.backends.cudnn.deterministic = self.deterministic

        # ---------------------------------------------------------------------------
        # verbose_allclose  (from utils.py)
        # ---------------------------------------------------------------------------
        @torch.no_grad()
        def verbose_allclose(
            received: torch.Tensor,
            expected: torch.Tensor,
            rtol=2e-2,
            atol=2e-2,
            max_print=5,
        ) -> Tuple[bool, list]:
            if received.shape != expected.shape:
                return False, ["SIZE MISMATCH"]

            diff = torch.abs(received.to(torch.float32) - expected.to(torch.float32))
            tolerance = atol + rtol * torch.abs(expected)
            tol_mismatched = diff > tolerance

            nan_mismatched = torch.logical_xor(torch.isnan(received), torch.isnan(expected))
            posinf_mismatched = torch.logical_xor(torch.isposinf(received), torch.isposinf(expected))
            neginf_mismatched = torch.logical_xor(torch.isneginf(received), torch.isneginf(expected))

            mismatched = torch.logical_or(
                torch.logical_or(tol_mismatched, nan_mismatched),
                torch.logical_or(posinf_mismatched, neginf_mismatched),
            )

            mismatched_indices = torch.nonzero(mismatched)
            num_mismatched = mismatched.count_nonzero().item()
            if num_mismatched >= 1:
                mismatch_details = [f"Number of mismatched elements: {{num_mismatched}}"]
                for index in mismatched_indices[:max_print]:
                    i = tuple(index.tolist())
                    mismatch_details.append(f"ERROR at {{i}}: {{received[i]}} {{expected[i]}}")
                if num_mismatched > max_print:
                    mismatch_details.append(
                        f"... and {{num_mismatched - max_print}} more mismatched elements."
                    )
                return False, mismatch_details

            return True, [f"Maximum error: {{torch.max(diff)}}"]

        # ---------------------------------------------------------------------------
        # TriMul reference implementation  (from reference.py)
        # ---------------------------------------------------------------------------
        class TriMul(nn.Module):
            # Based on https://github.com/lucidrains/triangle-multiplicative-module
            def __init__(self, dim: int, hidden_dim: int, device="cuda", dtype=""):
                super().__init__()
                self.norm = nn.LayerNorm(dim)
                self.left_proj = nn.Linear(dim, hidden_dim, bias=False, device=device)
                self.right_proj = nn.Linear(dim, hidden_dim, bias=False, device=device)
                self.left_gate = nn.Linear(dim, hidden_dim, bias=False, device=device)
                self.right_gate = nn.Linear(dim, hidden_dim, bias=False, device=device)
                self.out_gate = nn.Linear(dim, hidden_dim, bias=False, device=device)
                self.to_out_norm = nn.LayerNorm(hidden_dim, device=device)
                self.to_out = nn.Linear(hidden_dim, dim, bias=False, device=device)

            def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
                batch_size, seq_len, _, dim = x.shape
                x = self.norm(x)
                left = self.left_proj(x)
                right = self.right_proj(x)
                mask = mask.unsqueeze(-1)
                left = left * mask
                right = right * mask
                left_gate = self.left_gate(x).sigmoid()
                right_gate = self.right_gate(x).sigmoid()
                out_gate = self.out_gate(x).sigmoid()
                left = left * left_gate
                right = right * right_gate
                out = einsum("... i k d, ... j k d -> ... i j d", left, right)
                out = self.to_out_norm(out)
                out = out * out_gate
                return self.to_out(out)

        def ref_kernel(data):
            with DisableCuDNNTF32():
                input_tensor, mask, weights, config = data
                trimul = TriMul(
                    dim=config["dim"],
                    hidden_dim=config["hidden_dim"],
                    device=input_tensor.device,
                )
                trimul.norm.weight = nn.Parameter(weights["norm.weight"])
                trimul.norm.bias = nn.Parameter(weights["norm.bias"])
                trimul.left_proj.weight = nn.Parameter(weights["left_proj.weight"])
                trimul.right_proj.weight = nn.Parameter(weights["right_proj.weight"])
                trimul.left_gate.weight = nn.Parameter(weights["left_gate.weight"])
                trimul.right_gate.weight = nn.Parameter(weights["right_gate.weight"])
                trimul.out_gate.weight = nn.Parameter(weights["out_gate.weight"])
                trimul.to_out_norm.weight = nn.Parameter(weights["to_out_norm.weight"])
                trimul.to_out_norm.bias = nn.Parameter(weights["to_out_norm.bias"])
                trimul.to_out.weight = nn.Parameter(weights["to_out.weight"])
                output = trimul(input_tensor, mask)
                return output

        def generate_input(
            seqlen: int,
            bs: int,
            dim: int,
            hiddendim: int,
            seed: int,
            nomask: bool,
            distribution: str,
        ):
            batch_size = bs
            seq_len = seqlen
            hidden_dim = hiddendim
            no_mask = nomask

            config = {{"hidden_dim": hidden_dim, "dim": dim}}

            gen = torch.Generator(device="cuda")
            gen.manual_seed(seed)

            if distribution == "cauchy":
                u = torch.empty(
                    (batch_size, seq_len, seq_len, dim),
                    device="cuda", dtype=torch.float32,
                )
                u.uniform_(0.0, 1.0, generator=gen)
                input_tensor = 2.0 * torch.tan(math.pi * (u - 0.5))
            else:
                input_tensor = torch.randn(
                    (batch_size, seq_len, seq_len, dim),
                    device="cuda",
                    dtype=torch.float32,
                    generator=gen,
                ).contiguous()

            if no_mask:
                mask = torch.ones(batch_size, seq_len, seq_len, device=input_tensor.device)
            else:
                mask = torch.randint(
                    0, 2, (batch_size, seq_len, seq_len),
                    device=input_tensor.device, generator=gen,
                )

            weights = {{}}
            weights["norm.weight"] = torch.randn(dim, device="cuda", dtype=torch.float32)
            weights["norm.bias"] = torch.randn(dim, device="cuda", dtype=torch.float32)
            weights["left_proj.weight"] = torch.randn(
                hidden_dim, dim, device="cuda", dtype=torch.float32
            ) / math.sqrt(hidden_dim)
            weights["right_proj.weight"] = torch.randn(
                hidden_dim, dim, device="cuda", dtype=torch.float32
            ) / math.sqrt(hidden_dim)
            weights["left_gate.weight"] = torch.randn(
                hidden_dim, dim, device="cuda", dtype=torch.float32
            ) / math.sqrt(hidden_dim)
            weights["right_gate.weight"] = torch.randn(
                hidden_dim, dim, device="cuda", dtype=torch.float32
            ) / math.sqrt(hidden_dim)
            weights["out_gate.weight"] = torch.randn(
                hidden_dim, dim, device="cuda", dtype=torch.float32
            ) / math.sqrt(hidden_dim)
            weights["to_out_norm.weight"] = torch.randn(
                hidden_dim, device="cuda", dtype=torch.float32
            )
            weights["to_out.weight"] = torch.randn(
                dim, hidden_dim, device="cuda", dtype=torch.float32
            ) / math.sqrt(dim)
            weights["to_out_norm.bias"] = torch.randn(
                hidden_dim, device="cuda", dtype=torch.float32
            )

            return (input_tensor, mask, weights, config)

        # ---------------------------------------------------------------------------
        # Helpers
        # ---------------------------------------------------------------------------
        def _clone_data(data):
            if isinstance(data, tuple):
                return tuple(_clone_data(x) for x in data)
            elif isinstance(data, list):
                return [_clone_data(x) for x in data]
            elif isinstance(data, dict):
                return {{k: _clone_data(v) for k, v in data.items()}}
            elif isinstance(data, torch.Tensor):
                return data.clone()
            else:
                return data

        def check_implementation(data, output):
            expected = ref_kernel(data)
            good, reasons = verbose_allclose(output, expected, rtol=2e-2, atol=2e-2)
            if len(reasons) > 0:
                return good, "\\n".join(reasons)
            return good, ""

        def calculate_stats(durations):
            runs = len(durations)
            total = sum(durations)
            best = min(durations)
            worst = max(durations)
            avg = total / runs
            variance = sum((x - avg) ** 2 for x in durations)
            std = math.sqrt(variance / (runs - 1)) if runs > 1 else 0.0
            err = std / math.sqrt(runs) if runs > 0 else 0.0
            return {{
                "runs": runs,
                "mean": avg,
                "std": std,
                "err": err,
                "best": float(best),
                "worst": float(worst),
            }}

        # ---------------------------------------------------------------------------
        # Correctness testing
        # ---------------------------------------------------------------------------
        def run_correctness_tests(specs):
            from submission import custom_kernel

            errors = []
            for idx, spec in enumerate(specs):
                try:
                    data = generate_input(**spec)
                    torch.cuda.synchronize()
                    submission_output = custom_kernel(_clone_data(data))
                    torch.cuda.synchronize()
                    good, message = check_implementation(data, submission_output)
                    if not good:
                        errors.append(f"Test {{idx}} ({{spec}}): {{message}}")
                except Exception as exc:
                    errors.append(f"Test {{idx}} ({{spec}}): EXCEPTION: {{exc}}")
            return errors

        # ---------------------------------------------------------------------------
        # Benchmark harness
        # ---------------------------------------------------------------------------
        def run_benchmarks(specs):
            from submission import custom_kernel

            all_stats = []
            for spec in specs:
                data = generate_input(**spec)
                check_copy = _clone_data(data)
                output = custom_kernel(data)
                good, message = check_implementation(check_copy, output)
                if not good:
                    return None, f"Benchmark correctness check failed: {{message}}"

                # Warmup
                for _ in range(3):
                    _ = custom_kernel(_clone_data(data))
                    torch.cuda.synchronize()

                durations = []
                max_repeats = 100
                max_time_ns = 10e9
                bm_start_time = time.perf_counter_ns()
                for i in range(max_repeats):
                    torch.cuda.synchronize()
                    start_event = torch.cuda.Event(enable_timing=True)
                    end_event = torch.cuda.Event(enable_timing=True)
                    start_event.record()
                    _ = custom_kernel(_clone_data(data))
                    end_event.record()
                    torch.cuda.synchronize()
                    duration_ns = start_event.elapsed_time(end_event) * 1e6  # ms -> ns
                    durations.append(duration_ns)

                    if i > 1:
                        total_bm_duration = time.perf_counter_ns() - bm_start_time
                        stats = calculate_stats(durations)
                        if (
                            stats["mean"] > 0
                            and stats["err"] / stats["mean"] < 0.001
                        ) or (
                            stats["mean"] * stats["runs"] > max_time_ns
                        ) or (
                            total_bm_duration > 120e9
                        ):
                            break

                all_stats.append(calculate_stats(durations))

            return all_stats, None

        # ---------------------------------------------------------------------------
        # Main entry point
        # ---------------------------------------------------------------------------
        def main():
            result = {{}}
            try:
                # Phase 1: correctness
                errors = run_correctness_tests(TEST_SPECS)
                if errors:
                    result["status"] = "failed"
                    result["errors"] = errors
                    result["latency_ms"] = None
                    result["metrics"] = {{}}
                else:
                    # Phase 2: benchmarks
                    stats_list, bench_err = run_benchmarks(BENCHMARK_SPECS)
                    if bench_err is not None:
                        result["status"] = "failed"
                        result["errors"] = [bench_err]
                        result["latency_ms"] = None
                        result["metrics"] = {{}}
                    else:
                        # Use the mean of the first benchmark spec as the
                        # primary latency (in milliseconds).
                        mean_ns = stats_list[0]["mean"]
                        latency_ms = mean_ns / 1e6
                        inv_latency = 1.0 / latency_ms if latency_ms > 0 else 0.0
                        result["status"] = "passed"
                        result["latency_ms"] = round(latency_ms, 6)
                        result["metrics"] = {{
                            "score_name": "inv_latency_ms",
                            "score": round(inv_latency, 6),
                        }}
                        result["benchmark_details"] = stats_list
            except Exception:
                result["status"] = "error"
                result["errors"] = [traceback.format_exc()]
                result["latency_ms"] = None
                result["metrics"] = {{}}

            print(RESULT_JSON_START)
            print(json.dumps(result))
            print(RESULT_JSON_END)

        if __name__ == "__main__":
            main()
    """)

    return source
