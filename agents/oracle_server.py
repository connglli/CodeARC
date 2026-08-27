#!/usr/bin/env python3
"""
Host-side Central Oracle and Differential Testing Daemon for CodeARC Benchmark.

Serves test-time oracle queries and candidate code differential checks over local HTTP RPC.
Zero reference ground-truth code is ever leaked to the client agent workspaces.
Temporary files for differential testing are kept strictly in outdir/tmp/.
All code executions are loaded from templates and isolated in subprocesses with strict timeouts.
Provides verify_solution as the single source of truth for online checks and offline evaluation.
"""

from __future__ import annotations

import http.server
import json
import os
import re
import shutil
import socketserver
import subprocess
import sys
import threading
import time
import warnings
from pathlib import Path
from typing import Any

warnings.filterwarnings("ignore", category=SyntaxWarning)

TEMPLATES_DIR = Path(__file__).parent / "templates"


def make_diff_test_script(sample: dict[str, Any], sol_code: str) -> str:
  """Generates diff_test_solution script from template for live differential execution."""
  fn_name = sample["function_name"]
  gt_code = sample["gt_code"]
  invocations = sample["invocations"]
  template = (TEMPLATES_DIR / "diff_test_solution.py.template").read_text(
    encoding="utf-8"
  )
  return (
    template.replace("__TASK_ID__", str(sample["id"]))
    .replace("__TARGET_FN_NAME__", repr(fn_name))
    .replace("__GT_CODE__", repr(gt_code))
    .replace("__SOL_CODE__", repr(sol_code))
    .replace("__INVOCATIONS__", repr(invocations))
  )


def make_eval_oracle_script(gt_code: str, fn_name: str, raw_input: str) -> str:
  """Generates oracle evaluation script from template."""
  template = (TEMPLATES_DIR / "eval_oracle.py.template").read_text(encoding="utf-8")
  return (
    template.replace("__TARGET_FN_NAME__", repr(fn_name))
    .replace("__RAW_INPUT__", repr(raw_input))
    .replace("__GT_CODE__", repr(gt_code))
  )


def safe_eval_oracle(
  gt_code: str, fn_name: str, raw_input: str, timeout: float = 4.0
) -> str:
  """
  Safely executes the ground-truth function on raw_input in an isolated subprocess.
  Includes a smart fallback for unquoted string literals and enforces a strict timeout.
  """
  script = make_eval_oracle_script(gt_code, fn_name, raw_input)
  try:
    proc = subprocess.run(
      [sys.executable, "-c", script],
      capture_output=True,
      text=True,
      timeout=timeout,
      check=False,
    )
    out = proc.stdout.strip()
    if not out and proc.stderr:
      out = f"<Exception: {proc.stderr.strip()}>"
    return out or "None"
  except subprocess.TimeoutExpired:
    return f"<Exception: TimeoutExpired: Oracle evaluation exceeded {timeout}s>"
  except (OSError, subprocess.SubprocessError, ValueError) as exc:
    return f"<Exception: {type(exc).__name__}: {exc}>"


def run_pynguin_verification(
  gt_code: str,
  sol_code: str,
  timeout: float = 10.0,
  tmp_dir: Path | None = None,
) -> dict[str, Any]:
  """
  Runs Pynguin automated differential test generation and execution.
  All temporary artifacts are created in tmp_dir (defaults to outdir/tmp).
  Returns structured dictionary with test counts, result status, and failure details.
  """
  venv_bin = Path(sys.executable).parent
  pynguin_bin = venv_bin / "pynguin"
  pytest_bin = venv_bin / "pytest"
  pynguin_cmd = (
    str(pynguin_bin) if pynguin_bin.exists() else (shutil.which("pynguin") or "")
  )
  pytest_cmd = (
    str(pytest_bin) if pytest_bin.exists() else (shutil.which("pytest") or "")
  )

  if not pynguin_cmd or not pytest_cmd:
    return {
      "passed": True,
      "tests_generated": 0,
      "result": "UNAVAILABLE",
      "details": "pynguin or pytest not found in PATH or virtualenv",
    }

  base_tmp = (tmp_dir if tmp_dir is not None else Path("agents/output/tmp")).resolve()
  base_tmp.mkdir(parents=True, exist_ok=True)
  temp_dir = base_tmp / f"pynguin_{os.getpid()}_{int(time.time() * 1000) % 1000000}"
  temp_dir.mkdir(parents=True, exist_ok=True)

  try:
    (temp_dir / "solution.py").write_text(gt_code, encoding="utf-8")
    env = os.environ.copy()
    env["PYNGUIN_DANGER_AWARE"] = "TRUE"
    env["PYTHONWARNINGS"] = "ignore::SyntaxWarning"

    cmd = [
      pynguin_cmd,
      "--maximum-search-time",
      "5",
      "--project-path",
      str(temp_dir.resolve()),
      "--output-path",
      str(temp_dir.resolve()),
      "--module-name",
      "solution",
      "--assertion-generation",
      "SIMPLE",
    ]
    subprocess.run(
      cmd,
      stdout=subprocess.DEVNULL,
      stderr=subprocess.DEVNULL,
      timeout=timeout + 4,
      env=env,
      check=False,
    )

    test_files = list(temp_dir.glob("test_*.py"))
    if not test_files:
      return {
        "passed": True,
        "tests_generated": 0,
        "result": "NO_TESTS_GENERATED",
        "details": "Pynguin completed without generating test cases",
      }

    test_content = test_files[0].read_text(encoding="utf-8")
    test_count = len(re.findall(r"^def test_", test_content, re.MULTILINE))

    (temp_dir / "solution.py").write_text(sol_code, encoding="utf-8")
    proc = subprocess.run(
      [pytest_cmd, str(test_files[0]), "-v"],
      cwd=str(temp_dir),
      capture_output=True,
      text=True,
      timeout=timeout,
      env=env,
      check=False,
    )
    if proc.returncode != 0:
      failure_line = "Pynguin differential test suite failed"
      for line in proc.stdout.splitlines():
        if "FAILED" in line or "AssertionError" in line:
          failure_line = line.strip()
          break
      return {
        "passed": False,
        "tests_generated": test_count,
        "result": "FAIL",
        "details": failure_line,
      }
    return {
      "passed": True,
      "tests_generated": test_count,
      "result": "PASS",
      "details": None,
    }
  except subprocess.TimeoutExpired:
    return {
      "passed": False,
      "tests_generated": 0,
      "result": "TIMEOUT",
      "details": f"Pynguin verification timed out ({timeout}s)",
    }
  except Exception as exc:  # noqa: BLE001
    return {
      "passed": False,
      "tests_generated": 0,
      "result": "ERROR",
      "details": str(exc),
    }
  finally:
    if temp_dir.exists():
      shutil.rmtree(temp_dir, ignore_errors=True)


def verify_solution(
  sample: dict[str, Any],
  sol_code: str,
  tmp_dir: Path | None = None,
  timeout: float = 6.0,
) -> dict[str, Any]:
  """
  Single Source of Truth for candidate code verification.

  Executes:
  1. Live side-by-side differential test on problem examples via diff_test_solution.py.template.
  2. Pynguin differential test generation and execution in tmp_dir.

  Returns a standardized result dict with comprehensive diff_test and pynguin metrics.
  """
  fn_name = sample.get("function_name", "solution")
  gt_code = sample.get("gt_code", "")

  # Stage 1: Differential testing on problem examples
  pbe_passed = False
  pbe_score = "0/0"
  passed_count = 0
  total_count = 0
  failed_input = None
  expected_output = None
  actual_output = None
  msg = "Passed"

  script = make_diff_test_script(sample, sol_code)
  try:
    proc = subprocess.run(
      [sys.executable, "-c", script],
      capture_output=True,
      text=True,
      timeout=timeout,
      check=False,
    )
    out = proc.stdout.strip()
    m = re.search(r"Summary:\s*(\d+)/(\d+)\s*examples passed", out)
    if m:
      passed_count = int(m.group(1))
      total_count = int(m.group(2))
      pbe_score = f"{passed_count}/{total_count}"
      if passed_count == total_count and total_count > 0:
        pbe_passed = True
    elif "__PBE_PASSED__=True" in out and proc.returncode == 0:
      pbe_passed = True
      pbe_score = "All"

    if "DIFF_ALL_PASSED" not in out or proc.returncode != 0:
      pbe_passed = False
      for line in out.splitlines():
        if line.startswith("DIFF_MISMATCH\t"):
          parts = line.split("\t", 3)
          failed_input = parts[1]
          expected_output = parts[2]
          actual_output = parts[3]
          msg = f"Differential mismatch on '{failed_input}'"
          break
        if line.startswith("SOL_LOAD_ERROR\t"):
          failed_input = "Candidate Code"
          expected_output = "Valid Python Code"
          actual_output = line.split("\t", 1)[1]
          msg = f"Candidate syntax error: {actual_output}"
          break
        if line.startswith("SOL_NO_CALLABLE\t"):
          failed_input = "Candidate Function"
          expected_output = f"Callable function '{fn_name}'"
          actual_output = "No callable function found in solution.py"
          msg = actual_output
          break
        if line.startswith("GT_LOAD_ERROR\t"):
          failed_input = "Reference Code"
          expected_output = "Valid Execution"
          actual_output = line.split("\t", 1)[1]
          msg = f"Reference error: {actual_output}"
          break
      if not failed_input:
        failed_input = "Execution"
        expected_output = "Valid execution"
        actual_output = proc.stderr.strip() or out
        msg = actual_output or "Differential execution failed"

  except subprocess.TimeoutExpired:
    pbe_passed = False
    failed_input = "Execution"
    expected_output = "Execution within timeout"
    actual_output = f"TimeoutExpired: Execution exceeded {timeout}s"
    msg = actual_output
  except (OSError, subprocess.SubprocessError, ValueError) as exc:
    pbe_passed = False
    failed_input = "Execution"
    expected_output = "Valid execution"
    actual_output = str(exc)
    msg = actual_output

  pbe_diff_result = {
    "passed": pbe_passed,
    "score": pbe_score,
    "passed_count": passed_count,
    "total_count": total_count,
    "failed_input": failed_input,
    "expected_output": expected_output,
    "actual_output": actual_output,
  }

  if not pbe_passed:
    pynguin_result = {
      "passed": False,
      "tests_generated": 0,
      "result": "SKIPPED_DUE_TO_PBE_FAIL",
      "details": None,
    }
    return {
      "is_correct": False,
      "pbe_passed": False,
      "pbe_score": pbe_score,
      "diff_test": pbe_diff_result,
      "pynguin": pynguin_result,
      "message": msg,
    }

  # Stage 2: Pynguin differential test generation
  pynguin_result = run_pynguin_verification(
    gt_code, sol_code, timeout=timeout, tmp_dir=tmp_dir
  )

  is_correct = pbe_passed and pynguin_result["passed"]
  final_msg = (
    "Passed"
    if is_correct
    else (pynguin_result["details"] or "Pynguin differential test failure")
  )

  return {
    "is_correct": is_correct,
    "pbe_passed": pbe_passed,
    "pbe_score": pbe_score,
    "diff_test": pbe_diff_result,
    "pynguin": pynguin_result,
    "message": final_msg,
  }


class CentralOracleServer(socketserver.ThreadingTCPServer):
  """Single host-side central daemon serving oracle queries and differential checks for all tasks."""

  allow_reuse_address = True

  def __init__(
    self,
    server_address: tuple[str, int],
    dataset: list[dict[str, Any]],
    max_queries: int = 20,
    max_checks: int = 2,
    tmp_dir: Path | None = None,
  ):
    self.dataset_map = {s["id"]: s for s in dataset}
    self.max_queries = max_queries
    self.max_checks = max_checks
    self.tmp_dir = tmp_dir
    self.query_counts: dict[str, int] = {}
    self.check_counts: dict[str, int] = {}
    self.lock = threading.Lock()
    super().__init__(server_address, CentralOracleHandler)


class CentralOracleHandler(http.server.BaseHTTPRequestHandler):
  """HTTP request handler for the central oracle service."""

  def do_POST(self):  # noqa: N802
    try:
      length = int(self.headers.get("Content-Length", 0))
      raw_body = self.rfile.read(length).decode("utf-8")
      body = json.loads(raw_body)
    except Exception as exc:  # noqa: BLE001
      self.send_json({"error": f"Invalid JSON request: {exc}"})
      return

    server: CentralOracleServer = self.server  # type: ignore[assignment]
    prob_id = str(body.get("problem_id", "")).strip()
    sample = server.dataset_map.get(prob_id)
    if not sample:
      self.send_json({"error": f"Problem ID '{prob_id}' not found in dataset"})
      return

    gt_code = sample["gt_code"]
    fn_name = sample["function_name"]

    if self.path == "/oracle":
      with server.lock:
        used = server.query_counts.get(prob_id, 0)
        if used >= server.max_queries:
          self.send_json(
            {
              "error": (
                f"Oracle retrieval limit reached: {used}/{server.max_queries} queries used."
              )
            }
          )
          return
        server.query_counts[prob_id] = used + 1
        current_used = server.query_counts[prob_id]

      raw_input = body.get("input", "").strip()
      result_val = safe_eval_oracle(gt_code, fn_name, raw_input, timeout=4.0)

      resp = {
        "result": result_val,
        "queries_remaining": server.max_queries - current_used,
        "max_queries": server.max_queries,
      }
      self.send_json(resp)

    elif self.path == "/check":
      with server.lock:
        used = server.check_counts.get(prob_id, 0)
        if used >= server.max_checks:
          self.send_json(
            {
              "error": (
                f"Differential testing check budget exhausted: {used}/{server.max_checks} checks used."
              )
            }
          )
          return
        server.check_counts[prob_id] = used + 1
        current_used = server.check_counts[prob_id]

      sol_code = body.get("code", "")
      verif = verify_solution(
        sample=sample,
        sol_code=sol_code,
        tmp_dir=server.tmp_dir,
        timeout=6.0,
      )

      if verif["is_correct"]:
        resp = {
          "passed": True,
          "checks_remaining": server.max_checks - current_used,
          "max_checks": server.max_checks,
        }
      else:
        resp = {
          "passed": False,
          "checks_remaining": server.max_checks - current_used,
          "max_checks": server.max_checks,
        }
      self.send_json(resp)

    else:
      self.send_json({"error": f"Endpoint '{self.path}' not found"})

  def send_json(self, data: dict[str, Any]) -> None:
    try:
      payload = json.dumps(data).encode("utf-8")
      self.send_response(200)
      self.send_header("Content-Type", "application/json")
      self.send_header("Content-Length", str(len(payload)))
      self.end_headers()
      self.wfile.write(payload)
      self.wfile.flush()
    except (OSError, BrokenPipeError, ConnectionResetError):
      pass

  def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
    pass
