#!/usr/bin/env python3
"""
Host-side Central Oracle and Differential Testing Daemon for CodeARC Benchmark.

Serves test-time oracle queries and candidate code differential checks over local HTTP RPC.
Zero reference ground-truth code is ever leaked to the client agent workspaces.
Temporary files for differential testing are kept strictly in outdir/tmp/.
"""

from __future__ import annotations

import http.server
import io
import json
import os
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


def run_pynguin_verification(
  gt_code: str,
  sol_code: str,
  timeout: float = 10.0,
  tmp_dir: Path | None = None,
) -> str | None:
  """
  Runs Pynguin automated differential test generation and execution.
  All temporary artifacts are created in tmp_dir (defaults to outdir/tmp).
  """
  pynguin_cmd = shutil.which("pynguin")
  pytest_cmd = shutil.which("pytest")
  if not pynguin_cmd or not pytest_cmd:
    return None

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
      "pynguin",
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
      return None

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
      for line in proc.stdout.splitlines():
        if "FAILED" in line or "AssertionError" in line:
          return f"Pynguin test failure: {line.strip()}"
      return "Pynguin differential test suite failed"
    return "PASS"
  except Exception:  # noqa: BLE001
    return None
  finally:
    if temp_dir.exists():
      shutil.rmtree(temp_dir, ignore_errors=True)


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
    length = int(self.headers.get("Content-Length", 0))
    body = json.loads(self.rfile.read(length).decode("utf-8"))
    server: CentralOracleServer = self.server  # type: ignore[assignment]

    prob_id = str(body.get("problem_id", ""))
    sample = server.dataset_map.get(prob_id)
    if not sample:
      self.send_json({"error": f"Problem ID {prob_id} not found in dataset"})
      return

    gt_code = sample["gt_code"]
    fn_name = sample["function_name"]
    invocations = sample["invocations"]

    with server.lock:
      if self.path == "/oracle":
        used = server.query_counts.get(prob_id, 0)
        if used >= server.max_queries:
          resp = {
            "error": (
              f"Oracle retrieval limit reached: {used}/{server.max_queries} "
              "queries used (30/30 total examples)."
            )
          }
        else:
          server.query_counts[prob_id] = used + 1
          raw_input = body.get("input", "").strip()
          gt_globals: dict[str, Any] = {}
          try:
            exec(gt_code, gt_globals)
            res = repr(eval(f"{fn_name}({raw_input})", gt_globals))
          except Exception as exc:  # noqa: BLE001
            res = f"<Exception: {type(exc).__name__}: {exc}>"

          resp = {
            "result": res,
            "queries_remaining": server.max_queries - server.query_counts[prob_id],
            "max_queries": server.max_queries,
            "total_remaining": server.max_queries + 10 - server.query_counts[prob_id],
          }
        self.send_json(resp)

      elif self.path == "/check":
        used = server.check_counts.get(prob_id, 0)
        if used >= server.max_checks:
          resp = {
            "error": (
              f"Differential testing check budget exhausted: {used}/{server.max_checks} checks used."
            )
          }
        else:
          server.check_counts[prob_id] = used + 1
          sol_code = body.get("code", "")
          gt_globals = {}
          sol_globals: dict[str, Any] = {}
          mismatch = None

          try:
            exec(gt_code, gt_globals)
            f_gt = gt_globals.get(fn_name) or gt_globals.get("solution")
          except Exception as exc:  # noqa: BLE001
            f_gt = None
            mismatch = ("Reference load", "Valid function", str(exc))

          try:
            exec(sol_code, sol_globals)
            f_sol = sol_globals.get(fn_name) or sol_globals.get("solution")
            if not callable(f_sol):
              for k, v in sol_globals.items():
                if callable(v) and not k.startswith("__"):
                  f_sol = v
                  break
            if not callable(f_sol):
              mismatch = (
                "Candidate load",
                f"Callable function '{fn_name}'",
                "No callable function found",
              )
          except Exception as exc:  # noqa: BLE001
            mismatch = ("Candidate execution", "Normal execution", str(exc))

          if not mismatch and f_gt and f_sol:
            for ex in invocations:
              inp = ex.get("input", "")
              if not inp:
                continue
              try:
                buf_gt = io.StringIO()
                old_stdout = sys.stdout
                try:
                  sys.stdout = buf_gt
                  exec(inp, gt_globals)
                finally:
                  sys.stdout = old_stdout

                buf_sol = io.StringIO()
                old_stdout = sys.stdout
                try:
                  sys.stdout = buf_sol
                  exec(inp, sol_globals)
                finally:
                  sys.stdout = old_stdout

                out_gt = buf_gt.getvalue().strip()
                out_sol = buf_sol.getvalue().strip()
                if out_gt != out_sol:
                  mismatch = (inp, out_gt, out_sol)
                  break
              except Exception as exc:  # noqa: BLE001
                if not ex.get("errored", False):
                  mismatch = (
                    inp,
                    "Normal execution",
                    f"Exception: {type(exc).__name__}: {exc}",
                  )
                  break

          if not mismatch:
            pynguin_res = run_pynguin_verification(
              gt_code, sol_code, timeout=6.0, tmp_dir=server.tmp_dir
            )
            if pynguin_res and pynguin_res != "PASS":
              mismatch = (
                "Automated Pynguin test",
                "Expected ground truth behavior",
                pynguin_res,
              )

          if mismatch:
            resp = {
              "passed": False,
              "failed_input": mismatch[0],
              "expected_output": mismatch[1],
              "actual_output": mismatch[2],
              "checks_remaining": server.max_checks - server.check_counts[prob_id],
              "max_checks": server.max_checks,
            }
          else:
            resp = {
              "passed": True,
              "checks_remaining": server.max_checks - server.check_counts[prob_id],
              "max_checks": server.max_checks,
            }
        self.send_json(resp)

      else:
        self.send_json({"error": "Endpoint not found"})

  def send_json(self, data: dict[str, Any]) -> None:
    self.send_response(200)
    self.send_header("Content-Type", "application/json")
    self.end_headers()
    self.wfile.write(json.dumps(data).encode("utf-8"))

  def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
    pass
