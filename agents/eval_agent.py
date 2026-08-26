#!/usr/bin/env python3
"""
OpenCode and Claude Code Agent Evaluation Framework for CodeARC Benchmark (Anonymous Dataset).

Completely self-contained evaluation runner for CodeARC Programming-by-Example (PBE)
program synthesis using OpenCode and Claude Code as autonomous coding agents.

Security & Architecture:
- Central Oracle Server: Evaluator runs `oracle_server.CentralOracleServer` in background,
  holding ground truths in memory and serving test-time oracle queries and differential checks.
- File Templates: Cleanly loaded from `agents/templates/`.
- Zero reference code is written to the agent workspaces.
- `ask_oracle.py` in each workspace is a lightweight HTTP RPC client connecting to the central server.
- Test-time Oracle retrieval (`ask_oracle.py query <input>`): up to 20 additional queries (30 total example budget).
- Test-time Differential testing (`ask_oracle.py check`): up to 2 candidate code checks with Pynguin.
- Verification-time: Full differential testing with Pynguin across all dataset tests.
"""

from __future__ import annotations

import argparse
import ast
import concurrent.futures
import io
import json
import os
import re
import subprocess
import sys
import threading
import time
import warnings
from pathlib import Path
from typing import Any

warnings.filterwarnings("ignore", category=SyntaxWarning)

# Ensure warnings are suppressed in child processes as well
os.environ["PYTHONWARNINGS"] = "ignore::SyntaxWarning"

try:
  from oracle_server import CentralOracleServer, run_pynguin_verification
except ImportError:
  from agents.oracle_server import CentralOracleServer, run_pynguin_verification

# Paths resolved relative to this script
REPO_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = REPO_ROOT / "dataset" / "anonymous"
TEMPLATES_DIR = REPO_ROOT / "agents" / "templates"
INVOCATIONS_FILE = (
  REPO_ROOT
  / "prompt_invocations"
  / "anonymous"
  / "num_invocations=10"
  / "llm"
  / "invocations.json"
)


def extract_function_name(
  code_str: str, raw_examples: dict[str, Any] | None = None
) -> str:
  """Extracts target function name from Python source code or invocation examples."""
  if raw_examples:
    for ex in raw_examples.values():
      inp = ex.get("input", "") if isinstance(ex, dict) else ""
      if inp:
        try:
          tree = ast.parse(inp)
          for node in ast.walk(tree):
            if (
              isinstance(node, ast.Call)
              and isinstance(node.func, ast.Name)
              and node.func.id not in ("print", "str")
            ):
              return node.func.id
        except (SyntaxError, ValueError, TypeError, AttributeError):
          pass
        m = re.search(r"str\s*\(\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\(", inp)
        if m:
          return m.group(1)

  try:
    tree = ast.parse(code_str)
    top_level_fns = [
      node.name for node in tree.body if isinstance(node, ast.FunctionDef)
    ]
    if top_level_fns:
      if "main" in top_level_fns and len(top_level_fns) > 1:
        non_main = [name for name in top_level_fns if name != "main"]
        return non_main[-1]
      return top_level_fns[-1]
  except (SyntaxError, ValueError, TypeError, AttributeError):
    pass

  match = re.findall(r"^def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(", code_str, re.MULTILINE)
  if match:
    if "main" in match and len(match) > 1:
      return [m for m in match if m != "main"][-1]
    return match[-1]

  return "solution"


def load_dataset() -> list[dict[str, Any]]:
  """Loads CodeARC anonymous dataset files and merges with prompt invocations."""
  if not DATASET_DIR.exists():
    raise FileNotFoundError(f"Dataset directory not found: {DATASET_DIR}")
  if not INVOCATIONS_FILE.exists():
    raise FileNotFoundError(f"Invocations file not found: {INVOCATIONS_FILE}")

  with open(INVOCATIONS_FILE, "r", encoding="utf-8") as f:
    invocations_raw = json.load(f)

  py_files = [f for f in DATASET_DIR.iterdir() if f.name.endswith(".py")]

  def get_file_num(path: Path) -> int:
    m = re.match(r"^(\d+)\.py$", path.name)
    return int(m.group(1)) if m else 999999

  py_files.sort(key=get_file_num)

  records = []
  for file_path in py_files:
    prob_id = str(get_file_num(file_path))
    if prob_id == "999999":
      continue
    gt_code = file_path.read_text(encoding="utf-8")
    raw_examples = invocations_raw.get(prob_id, {})
    fn_name = extract_function_name(gt_code, raw_examples=raw_examples)

    sorted_examples = sorted(
      raw_examples.items(),
      key=lambda item: int(item[0]) if item[0].isdigit() else 999,
    )

    examples_list = []
    for idx_str, ex_dict in sorted_examples:
      examples_list.append(
        {
          "index": int(idx_str) if idx_str.isdigit() else 0,
          "input": ex_dict.get("input", ""),
          "output": ex_dict.get("output", ""),
          "errored": ex_dict.get("errored", False),
        }
      )

    records.append(
      {
        "id": prob_id,
        "problem_id": int(prob_id),
        "dataset": "anonymous",
        "function_name": fn_name,
        "gt_code": gt_code,
        "invocations": examples_list,
      }
    )

  return records


def make_test_file(sample: dict[str, Any]) -> str:
  """Generates test_solution.py from template."""
  fn_name = sample["function_name"]
  invocations = sample["invocations"]

  valid_tests = [
    (ex["input"], ex["output"])
    for ex in invocations
    if not ex.get("errored", False) and ex.get("input")
  ]

  template = (TEMPLATES_DIR / "test_solution.py.template").read_text(encoding="utf-8")
  return (
    template.replace("__TASK_ID__", str(sample["id"]))
    .replace("__TARGET_FN_NAME__", repr(fn_name))
    .replace("__TEST_CASES__", repr(valid_tests))
  )


def make_ask_oracle_client_file(sample: dict[str, Any], server_port: int) -> str:
  """Generates ask_oracle.py client from template."""
  template = (TEMPLATES_DIR / "ask_oracle.py.template").read_text(encoding="utf-8")
  return template.replace("__SERVER_URL__", f"http://127.0.0.1:{server_port}").replace(
    "__PROBLEM_ID__", str(sample["id"])
  )


def make_agent_prompt(sample: dict[str, Any]) -> str:
  """Generates agent prompt from template."""
  fn_name = sample["function_name"]
  invocations = sample["invocations"]

  examples_lines = []
  for i, ex in enumerate(invocations, 1):
    inp = ex.get("input", "")
    out = ex.get("output", "")
    if inp:
      examples_lines.append(f"Example {i}:")
      examples_lines.append(f"  Invocation: {inp}")
      examples_lines.append(f"  Output    : {out}")

  examples_text = "\n".join(examples_lines)

  template = (TEMPLATES_DIR / "prompt.txt.template").read_text(encoding="utf-8")
  return template.replace("__FUNCTION_NAME__", fn_name).replace(
    "__EXAMPLES_TEXT__", examples_text
  )


def make_diff_test_file(sample: dict[str, Any], sol_code: str) -> str:
  """Generates diff_test_solution.py from template for live differential execution."""
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


def is_allowed_env_var(key: str) -> bool:
  """Selectively determines if an environment variable should be passed into the container."""
  key_upper = key.upper()
  exact_keys = {
    "OPENROUTER_API_KEY",
    "OPENROUTER_BASE_URL",
    "OPENAI_API_KEY",
    "OPENAI_BASE_URL",
    "OPENAI_ORG_ID",
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_BASE_URL",
    "ANTHROPIC_DEFAULT_OPUS_MODEL",
    "ANTHROPIC_DEFAULT_SONNET_MODEL",
    "ANTHROPIC_DEFAULT_HAIKU_MODEL",
    "DEEPSEEK_API_KEY",
    "DEEPSEEK_BASE_URL",
    "GEMINI_API_KEY",
    "GOOGLE_API_KEY",
    "TOGETHER_API_KEY",
    "TOGETHER_BASE_URL",
    "OPENCODE_API_KEY",
    "OPENCODE_MODEL",
    "OPENCODE_BASE_URL",
    "CLAUDE_CODE_SUBAGENT_MODEL",
    "TAVILY_API_KEY",
  }
  if key_upper in exact_keys:
    return True

  allowed_prefixes = (
    "OPENCODE_",
    "ANTHROPIC_",
    "OPENAI_",
    "DEEPSEEK_",
    "OPENROUTER_",
    "GEMINI_",
    "CLAUDE_",
    "TOGETHER_",
  )
  if key_upper.startswith(allowed_prefixes):
    return True

  allowed_suffixes = ("_API_KEY", "_BASE_URL")
  return bool(key_upper.endswith(allowed_suffixes))


def run_agent(
  prompt: str,
  workspace: Path,
  agent: str,
  model: str,
  timeout: int = 300,
  docker_image: str = "codearc-agent:latest",
  opencode_config: str | None = None,
  no_docker: bool = False,
) -> int:
  """
  Executes an AI coding agent (OpenCode or Claude Code) in an isolated Docker container
  (or directly on host if no_docker is True) mounting ONLY the workspace directory.
  Pipes stdout directly to traj.jsonl and stderr to error.txt.
  Returns exit_code.
  """
  env = os.environ.copy()
  env["PYTHONWARNINGS"] = "ignore::SyntaxWarning"

  if agent == "opencode":
    agent_cmd = [
      "opencode",
      "run",
      prompt,
      "--model",
      model,
      "--auto",
      "--format",
      "json",
    ]
  elif agent == "claude":
    agent_cmd = [
      "claude",
      "--print",
      "--verbose",
      "--model",
      model,
      "--output-format",
      "stream-json",
      "--dangerously-skip-permissions",
      prompt,
    ]
    env["ANTHROPIC_DEFAULT_OPUS_MODEL"] = model
    env["ANTHROPIC_DEFAULT_SONNET_MODEL"] = model
    env["ANTHROPIC_DEFAULT_HAIKU_MODEL"] = model
    env["CLAUDE_CODE_SUBAGENT_MODEL"] = model
  else:
    raise ValueError(f"Unsupported agent '{agent}'. Choose 'opencode' or 'claude'.")

  if no_docker:
    cmd = agent_cmd
    container_name = None
  else:
    container_name = (
      f"codearc-{workspace.name}-{os.getpid()}-{int(time.time() * 1000) % 100000}"
    )
    cmd = [
      "docker",
      "run",
      "--rm",
      "--net=host",
      "--name",
      container_name,
      "-e",
      "PYTHONUNBUFFERED=1",
      "-e",
      "PYTHONWARNINGS=ignore::SyntaxWarning",
      "-e",
      "PYNGUIN_DANGER_AWARE=TRUE",
      "-v",
      f"{workspace.resolve()}:/workspace",
      "-w",
      "/workspace",
    ]

    if opencode_config:
      config_src = Path(opencode_config).expanduser().resolve()
      if not config_src.is_file():
        raise FileNotFoundError(f"OpenCode config file not found: {config_src}")
      cmd.extend(
        ["-v", f"{config_src}:/home/runner/.config/opencode/opencode.jsonc:ro"]
      )

    for key, val in env.items():
      if is_allowed_env_var(key):
        cmd.extend(["-e", f"{key}={val}"])

    cmd.append(docker_image)
    cmd.extend(agent_cmd)

  traj_path = workspace / "traj.jsonl"
  error_path = workspace / "error.txt"

  with (
    open(traj_path, "w", encoding="utf-8") as fout,
    open(error_path, "w", encoding="utf-8") as ferr,
  ):
    proc = None
    try:
      proc = subprocess.Popen(
        cmd,
        stdout=fout,
        stderr=ferr,
        cwd=str(workspace),
        env=env,
      )
      return proc.wait(timeout=timeout + 15)
    except subprocess.TimeoutExpired:
      if proc:
        proc.kill()
        proc.wait()
      with open(error_path, "a", encoding="utf-8") as append_err:
        append_err.write(f"\n{agent} timed out after {timeout}s\n")
      raise TimeoutError(f"{agent} timed out after {timeout}s in {workspace}")
    except FileNotFoundError:
      executable = cmd[0]
      raise RuntimeError(
        f"Executable '{executable}' not found. Ensure Docker/Agent is installed and in PATH."
      )
    finally:
      if container_name:
        subprocess.run(
          ["docker", "rm", "-f", container_name],
          stdout=subprocess.DEVNULL,
          stderr=subprocess.DEVNULL,
          check=False,
        )


def verify_functional_correctness(
  workspace: Path,
  clean_test_content: str,
  sample: dict[str, Any],
  timeout: float = 6.0,
  tmp_dir: Path | None = None,
) -> tuple[bool, bool, str, str]:
  """
  Independently evaluates solution.py in workspace.
  Checks:
  1. Side-by-side differential execution on PBE examples against live ground truth (diff_test_solution.py).
  2. Pynguin differential test generation against ground truth using outdir/tmp.
  Returns (is_correct: bool, pbe_passed: bool, pbe_score: str, status_or_error: str).
  """
  sol_file = workspace / "solution.py"
  if not sol_file.exists():
    return (
      False,
      False,
      "0/0",
      "solution.py was not created by agent",
    )

  sol_code = sol_file.read_text(encoding="utf-8")
  gt_code = sample.get("gt_code", "")

  # Stage 1: Run live differential testing on problem examples
  pbe_passed = False
  pbe_score = "0/0"
  env = os.environ.copy()
  env["PYTHONWARNINGS"] = "ignore::SyntaxWarning"

  diff_test_script = make_diff_test_file(sample, sol_code)

  try:
    proc = subprocess.run(
      [sys.executable, "-c", diff_test_script],
      cwd=str(workspace),
      capture_output=True,
      text=True,
      timeout=timeout,
      env=env,
      check=False,
    )

    out = proc.stdout
    m = re.search(r"Summary:\s*(\d+)/(\d+)\s*examples passed", out)
    if m:
      pbe_score = f"{m.group(1)}/{m.group(2)}"
      if m.group(1) == m.group(2) and int(m.group(2)) > 0:
        pbe_passed = True
    elif "__PBE_PASSED__=True" in out and proc.returncode == 0:
      pbe_passed = True
      pbe_score = "All"

    if proc.returncode != 0 or not pbe_passed:
      err_msg = proc.stdout.strip() or proc.stderr.strip()
      return False, False, pbe_score, err_msg or "Differential PBE test mismatch"

  except subprocess.TimeoutExpired:
    return False, False, pbe_score, f"Differential verification timed out ({timeout}s)"
  except (OSError, subprocess.SubprocessError, ValueError) as e:
    return False, False, pbe_score, f"Differential verification error: {e}"

  # Stage 2: Automated Pynguin differential test check if available
  pynguin_result = run_pynguin_verification(
    gt_code, sol_code, timeout=timeout, tmp_dir=tmp_dir
  )
  if pynguin_result and pynguin_result != "PASS":
    return False, pbe_passed, pbe_score, pynguin_result

  return True, pbe_passed, pbe_score, "Passed"


def is_sample_completed(workspace: Path, agent: str = "opencode") -> bool:
  """
  Checks if a task has completed successfully in a previous run.
  Requires result.json and non-empty traj.jsonl.
  """
  result_file = workspace / "result.json"
  traj_file = workspace / "traj.jsonl"
  if not result_file.exists() or not traj_file.exists():
    return False
  try:
    if traj_file.stat().st_size == 0:
      return False
    if agent == "opencode":
      with open(traj_file, "r", encoding="utf-8") as f:
        first_line = f.readline().strip()
      if first_line:
        try:
          data = json.loads(first_line)
          if isinstance(data, dict) and data.get("type") == "error":
            return False
        except json.JSONDecodeError:
          return False
    return True
  except OSError:
    return False


def evaluate_task(
  sample: dict[str, Any],
  agent: str,
  model: str,
  workspace: Path,
  server_port: int,
  timeout: int,
  docker_image: str = "codearc-agent:latest",
  verbose: bool = False,
  opencode_config: str | None = None,
  no_docker: bool = False,
  tmp_dir: Path | None = None,
) -> dict[str, Any]:
  """Runs a single task with an agent (OpenCode or Claude) and evaluates result."""
  sample_id = sample["id"]
  sample_result_file = workspace / "result.json"

  if is_sample_completed(workspace, agent=agent):
    try:
      with open(sample_result_file, "r", encoding="utf-8") as f:
        cached_result = json.load(f)
      if verbose:
        status = "✅ PASS" if cached_result.get("correct") else "❌ FAIL"
        pbe_info = cached_result.get("pbe_score", "")
        print(
          f"[{sample_id}] ⏩ RESUMED {status} (PBE: {pbe_info}) | {cached_result.get('elapsed', 0.0)}s",
          flush=True,
        )
      return cached_result
    except (OSError, json.JSONDecodeError, KeyError):
      pass

  workspace.mkdir(parents=True, exist_ok=True)

  stale = workspace / "solution.py"
  if stale.exists():
    stale.unlink()

  test_content = make_test_file(sample)
  (workspace / "test_solution.py").write_text(test_content, encoding="utf-8")

  ask_oracle_content = make_ask_oracle_client_file(sample, server_port)
  (workspace / "ask_oracle.py").write_text(ask_oracle_content, encoding="utf-8")

  prompt = make_agent_prompt(sample)

  result = {
    "id": sample_id,
    "function_name": sample["function_name"],
    "correct": False,
    "pbe_passed": False,
    "pbe_score": "0/0",
    "error": None,
    "elapsed": 0.0,
    "workspace": str(workspace),
  }

  start_time = time.perf_counter()
  try:
    run_agent(
      prompt=prompt,
      workspace=workspace,
      agent=agent,
      model=model,
      timeout=timeout,
      docker_image=docker_image,
      opencode_config=opencode_config,
      no_docker=no_docker,
    )

    correct, pbe_passed, pbe_score, msg = verify_functional_correctness(
      workspace=workspace,
      clean_test_content=test_content,
      sample=sample,
      tmp_dir=tmp_dir,
    )

    result["correct"] = correct
    result["pbe_passed"] = pbe_passed
    result["pbe_score"] = pbe_score
    result["error"] = None if correct else msg

  except (
    OSError,
    subprocess.SubprocessError,
    TimeoutError,
    RuntimeError,
    ValueError,
  ) as exc:
    result["error"] = str(exc)
    err_path = workspace / "error.txt"
    existing_err = err_path.read_text(encoding="utf-8") if err_path.exists() else ""
    err_path.write_text(f"{existing_err}\nException: {exc}\n".strip(), encoding="utf-8")

  finally:
    result["elapsed"] = round(time.perf_counter() - start_time, 2)
    with open(sample_result_file, "w", encoding="utf-8") as f:
      json.dump(result, f, indent=2)

    if verbose:
      status = "✅ PASS" if result["correct"] else "❌ FAIL"
      pbe_stat = "✅" if result["pbe_passed"] else "❌"
      print(
        f"[{sample_id}] {status} | PBE: {pbe_stat} ({result['pbe_score']}) | {result['elapsed']}s | err: {str(result['error'])[:50]}",
        flush=True,
      )

  return result


def parse_throttle(throttle_str: str | None) -> tuple[int, float] | None:
  """Parses throttle specification 'num:seconds' (e.g. '5:10' for 10s after every 5 tasks)."""
  if not throttle_str:
    return None
  throttle_str = str(throttle_str).strip()
  if ":" in throttle_str:
    parts = throttle_str.split(":", 1)
    try:
      interval = int(parts[0].strip())
      delay = float(parts[1].strip())
      if interval <= 0 or delay <= 0:
        return None
      return interval, delay
    except ValueError:
      raise argparse.ArgumentTypeError(
        f"Invalid throttle format '{throttle_str}'. Expected 'num:seconds' (e.g. '5:10')."
      ) from None
  else:
    try:
      delay = float(throttle_str)
      if delay <= 0:
        return None
      return 1, delay
    except ValueError:
      raise argparse.ArgumentTypeError(
        f"Invalid throttle format '{throttle_str}'. Expected 'num:seconds' (e.g. '5:10')."
      ) from None


def build_argument_parser() -> argparse.ArgumentParser:
  """Builds CLI argument parser."""
  parser = argparse.ArgumentParser(
    description="AI Coding Agent Evaluation Framework for CodeARC Benchmark (OpenCode & Claude Code)"
  )
  parser.add_argument(
    "model",
    type=str,
    help="Model identifier passed to agent (e.g., 'claude-3-7-sonnet', 'opencode/deepseek-v4-pro')",
  )
  parser.add_argument(
    "--agent",
    type=str,
    choices=["opencode", "claude"],
    default="opencode",
    help="Agent CLI runner to evaluate: 'opencode' or 'claude' (default: opencode)",
  )
  parser.add_argument(
    "--num-workers",
    "-j",
    type=int,
    default=1,
    help="Concurrency / parallel worker processes (default: 1)",
  )
  parser.add_argument(
    "--start",
    "-s",
    type=int,
    default=0,
    help="Start index in dataset (default: 0)",
  )
  parser.add_argument(
    "--limit",
    "-n",
    type=int,
    default=None,
    help="Limit evaluation to N tasks (default: all remaining from start)",
  )
  parser.add_argument(
    "--timeout",
    type=int,
    default=300,
    help="Timeout in seconds for agent per task (default: 300s / 5min)",
  )
  parser.add_argument(
    "--outdir",
    "-o",
    type=str,
    default="agents/output",
    help="Output directory to store sample workspaces and result.json (default: agents/output)",
  )
  parser.add_argument(
    "--docker-image",
    type=str,
    default="codearc-agent:latest",
    help="Docker image for isolated task execution (default: codearc-agent:latest)",
  )
  parser.add_argument(
    "--opencode-config",
    type=str,
    default=None,
    help="Path to an opencode.jsonc config file to mount into the container (opencode agent only)",
  )
  parser.add_argument(
    "--throttle",
    type=parse_throttle,
    default=None,
    help="Throttling as 'num:seconds' to sleep 'seconds' after every 'num' tasks (e.g. '5:10')",
  )
  parser.add_argument(
    "--no-docker",
    action="store_true",
    help="Run agent directly on host without Docker container isolation",
  )
  parser.add_argument(
    "--verbose",
    "-v",
    action="store_true",
    help="Print detailed logs per task",
  )
  return parser


def print_startup_banner(
  args: argparse.Namespace,
  model: str,
  total_tasks: int,
  outdir: Path,
  server_port: int,
) -> None:
  """Prints benchmark startup banner."""
  print("=" * 70)
  print(f"🤖 {args.agent.upper()} Agent CodeARC PBE Evaluation (Anonymous Dataset)")
  print(f"   Agent        : {args.agent}")
  print(f"   Model        : {model}")
  print(
    f"   Docker Mode  : {'Disabled (Host)' if args.no_docker else args.docker_image}"
  )
  print(f"   Oracle Server: http://127.0.0.1:{server_port} (Single Master Daemon)")
  if args.opencode_config:
    print(f"   OC Config    : {args.opencode_config}")
  print(f"   Tasks        : {total_tasks} samples")
  print(f"   Workers      : {args.num_workers} parallel workers")
  print(f"   Timeout      : {args.timeout}s per task")
  if args.throttle is not None:
    t_interval, t_delay = args.throttle
    print(f"   Throttle     : Sleep {t_delay}s after every {t_interval} task(s)")
  print(f"   Outdir       : {outdir}")
  print("=" * 70)


def print_progress(
  completed: int,
  total: int,
  passed: int,
  pbe_passed: int,
  failed: int,
  verbose: bool = False,
) -> None:
  """Prints live progress to stdout."""
  if verbose:
    return
  pass_rate = (passed / completed * 100) if completed > 0 else 0.0
  pbe_rate = (pbe_passed / completed * 100) if completed > 0 else 0.0
  print(
    f"\r\033[K[{completed}/{total}] Passed (GT): {passed} ({pass_rate:.1f}%) | "
    f"PBE Passed: {pbe_passed} ({pbe_rate:.1f}%) | Failed: {failed}",
    end="",
    flush=True,
  )


def run_evaluation(
  samples: list[dict[str, Any]],
  args: argparse.Namespace,
  model: str,
  outdir: Path,
  server_port: int,
) -> list[dict[str, Any]]:
  """Runs evaluation across all selected samples using ThreadPoolExecutor."""
  total_tasks = len(samples)
  if total_tasks == 0:
    return []

  results = []
  passed = 0
  pbe_passed = 0
  failed = 0
  executed_count = 0

  workers = max(1, args.num_workers)
  with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
    sample_iter = iter(enumerate(samples))
    future_to_sample = {}

    def submit_next() -> bool:
      nonlocal executed_count
      try:
        _, sample = next(sample_iter)
      except StopIteration:
        return False

      workspace = outdir / sample["id"]
      is_cached = is_sample_completed(workspace, agent=args.agent)

      if not is_cached:
        executed_count += 1
        if (
          args.throttle is not None
          and executed_count > 1
          and (executed_count - 1) % args.throttle[0] == 0
        ):
          time.sleep(args.throttle[1])

      future = executor.submit(
        evaluate_task,
        sample=sample,
        agent=args.agent,
        model=model,
        workspace=workspace,
        server_port=server_port,
        timeout=args.timeout,
        docker_image=args.docker_image,
        verbose=args.verbose,
        opencode_config=args.opencode_config,
        no_docker=args.no_docker,
        tmp_dir=outdir / "tmp",
      )
      future_to_sample[future] = sample
      return True

    for _ in range(min(workers, total_tasks)):
      if not submit_next():
        break

    completed_count = 0
    while future_to_sample:
      done, _ = concurrent.futures.wait(
        future_to_sample.keys(),
        return_when=concurrent.futures.FIRST_COMPLETED,
      )
      for future in done:
        future_to_sample.pop(future)
        res = future.result()
        results.append(res)
        completed_count += 1

        if res["correct"]:
          passed += 1
        else:
          failed += 1

        if res["pbe_passed"]:
          pbe_passed += 1

        print_progress(
          completed_count,
          total_tasks,
          passed,
          pbe_passed,
          failed,
          verbose=args.verbose,
        )
        submit_next()

  if not args.verbose and total_tasks > 0:
    print()

  return results


def print_and_save_summary(
  results: list[dict[str, Any]],
  total_tasks: int,
  args: argparse.Namespace,
  model: str,
  outdir: Path,
) -> None:
  """Prints final metrics and writes lean summary JSON to outdir/result.json."""
  passed = sum(1 for r in results if r.get("correct"))
  pbe_passed = sum(1 for r in results if r.get("pbe_passed"))
  failed = total_tasks - passed
  final_pass_rate = (passed / total_tasks * 100) if total_tasks > 0 else 0.0
  final_pbe_rate = (pbe_passed / total_tasks * 100) if total_tasks > 0 else 0.0
  total_elapsed = sum(r.get("elapsed", 0.0) for r in results)
  avg_elapsed = round(total_elapsed / len(results), 2) if results else 0.0

  print("\n" + "=" * 70)
  print("📊 EVALUATION RESULTS - CodeARC Benchmark (Anonymous Dataset)")
  print(f"   Agent              : {args.agent}")
  print(f"   Model              : {model}")
  print(f"   Total Tasks        : {total_tasks}")
  print(f"   🏆 Full Pass Rate  : {final_pass_rate:.2f}% ({passed}/{total_tasks})")
  print(f"   🧪 PBE Pass Rate   : {final_pbe_rate:.2f}% ({pbe_passed}/{total_tasks})")
  print(f"   ❌ Failed Tasks    : {failed}")
  print(f"   ⏱️ Avg Elapsed      : {avg_elapsed:.2f}s")
  print("=" * 70)

  output_file = outdir / "result.json"
  summary = {
    "benchmark": "CodeARC",
    "dataset": "anonymous",
    "agent": args.agent,
    "model": model,
    "total_tasks": total_tasks,
    "passed": passed,
    "pbe_passed": pbe_passed,
    "failed": failed,
    "pass_rate": round(final_pass_rate, 2),
    "pbe_pass_rate": round(final_pbe_rate, 2),
    "avg_elapsed": avg_elapsed,
  }

  with open(output_file, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

  print(f"💾 Results saved to: {output_file}")


def main():
  try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
  except (AttributeError, io.UnsupportedOperation, OSError):
    pass

  parser = build_argument_parser()
  args = parser.parse_args()

  model = args.model
  dataset = load_dataset()

  start = max(0, args.start)
  samples = (
    dataset[start : start + args.limit] if args.limit is not None else dataset[start:]
  )

  total_tasks = len(samples)
  outdir = Path(args.outdir).resolve()
  outdir.mkdir(parents=True, exist_ok=True)
  tmp_dir = outdir / "tmp"
  tmp_dir.mkdir(parents=True, exist_ok=True)

  cmd_vars = vars(args).copy()
  if cmd_vars.get("throttle") is not None:
    cmd_vars["throttle"] = f"{cmd_vars['throttle'][0]}:{cmd_vars['throttle'][1]}"
  with open(outdir / "command.json", "w", encoding="utf-8") as f:
    json.dump(cmd_vars, f, indent=2)

  # Start Single Master Central Oracle Server with outdir/tmp
  server = CentralOracleServer(
    ("127.0.0.1", 0), dataset, max_queries=20, max_checks=2, tmp_dir=tmp_dir
  )
  server_port = server.server_address[1]
  server_thread = threading.Thread(target=server.serve_forever, daemon=True)
  server_thread.start()

  try:
    print_startup_banner(args, model, total_tasks, outdir, server_port)
    results = run_evaluation(samples, args, model, outdir, server_port)
    print_and_save_summary(results, total_tasks, args, model, outdir)
  finally:
    server.shutdown()
    server.server_close()


if __name__ == "__main__":
  main()
