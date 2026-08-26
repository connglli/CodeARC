# CodeARC Agent Evaluation Framework (Anonymous Dataset)

A self-contained evaluation framework for benchmarking AI coding agents (**OpenCode** and **Claude Code**) on the **CodeARC** benchmark for **Programming-by-Example (PBE)** program synthesis.

Following the design principles of `cruxeval/agents/`, each task is executed in an isolated workspace (inside Docker containers or directly on host), with full agent trajectory logging (`traj.jsonl`), error capturing (`error.txt`), self-testing via `test_solution.py`, test-time oracle querying and differential testing via `ask_oracle.py`, and verification-time differential testing with Pynguin.

---

## 🚀 Quick Start

### 1. Evaluate with Docker (Recommended)

```bash
# Evaluate OpenCode with Claude 3.7 Sonnet on anonymous dataset (first 10 samples)
./agents/run_eval.sh opencode/claude-3-7-sonnet --limit 10

# Evaluate Claude Code CLI directly
./agents/run_eval.sh claude-3-7-sonnet --agent claude --limit 10

# Rebuild Docker sandbox image
./agents/run_eval.sh --rebuild opencode/claude-3-7-sonnet --limit 5
```

### 2. Direct Python Execution (No Docker)

```bash
# Run with Python in virtualenv
./venv/bin/python agents/eval_agent.py opencode/claude-3-7-sonnet \
  --agent opencode \
  --limit 20 \
  --num-workers 4 \
  --outdir agents/output
```

---

## 📁 Workspace Architecture

For each problem in CodeARC (`0` to `1113`), an isolated workspace is created under `agents/output/<sample_id>/`:

```
agents/output/0/
├── test_solution.py     # Self-test script running the 10 initial PBE examples
├── ask_oracle.py        # Oracle tool for input queries and differential check
├── solution.py          # Synthesized implementation produced by the agent
├── traj.jsonl           # Complete streaming trajectory/tool-call logs of the agent
├── error.txt            # Stderr and error trace
└── result.json          # Per-sample evaluation metrics & verification status
```

---

## 🛠️ Test-Time Interaction Features (`ask_oracle.py`)

1. **Initial PBE Examples**: Each workspace starts with 10 visible input-output pairs in `test_solution.py` (unlimited runs).
2. **Oracle Input Query (`ask_oracle.py query <input_arg>`)**:
   - Query the ground truth output for a specific input argument (up to 20 additional queries, 30 total example budget).
   - Only the raw input argument is passed (e.g. `3.5` or `[1, 2, -4, 5]`, not `solution(...)`).
   - Example:
     ```bash
     python3 ask_oracle.py query "3.5"
     python3 ask_oracle.py query "[1, 2, -4, 5]"
     ```
3. **Differential Testing Check (`ask_oracle.py check`)**:
   - Run differential testing (with Pynguin) against the reference implementation to test candidate `solution.py` and receive counterexamples (up to 2 checks allowed).
   - Example:
     ```bash
     python3 ask_oracle.py check
     ```
4. **Verification-Time Differential Testing**: Upon task completion, `verify_functional_correctness` executes full differential testing with Pynguin and all dataset tests.

---

## 🛠️ CLI Options

| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `model` | `str` | *Required* | Model passed to agent (e.g. `opencode/claude-3-7-sonnet`, `claude-3-7-sonnet`) |
| `--agent` | `str` | `opencode` | Agent runner: `opencode` or `claude` |
| `--num-workers` / `-j` | `int` | `1` | Number of concurrent worker tasks |
| `--start` / `-s` | `int` | `0` | Start index in dataset (0-1113) |
| `--limit` / `-n` | `int` | `None` | Limit evaluation to N tasks |
| `--timeout` | `int` | `300` | Agent timeout per task in seconds |
| `--outdir` / `-o` | `str` | `agents/output` | Output directory for workspaces and results |
| `--docker-image` | `str` | `codearc-agent:latest` | Docker image tag for isolated task sandboxes |
| `--opencode-config` | `str` | `None` | Path to custom `opencode.jsonc` configuration file |
| `--throttle` | `str` | `None` | Rate limiter as `num:seconds` (e.g. `5:10` = sleep 10s every 5 tasks) |
| `--no-docker` | `flag` | `False` | Run agent directly on host without Docker |
| `--verbose` / `-v` | `flag` | `False` | Print detailed logs per task |

---

## 📊 Evaluation Metrics

When the benchmark finishes, a consolidated `result.json` and `command.json` are written to the output directory:

- **Full Pass Rate (`pass_rate`)**: Percentage of tasks where the synthesized `solution.py` is functionally equivalent to the ground truth function across all inputs (including Pynguin differential tests).
- **PBE Pass Rate (`pbe_pass_rate`)**: Percentage of tasks where `solution.py` passes 100% of the given PBE input-output examples.
- **Average Elapsed Time (`avg_elapsed`)**: Mean execution time per task.

---

## 🔑 Environment Variables Forwarded

When evaluating via `./agents/run_eval.sh` or Docker, the following API keys and settings are forwarded to the container:

- `OPENAI_API_KEY`, `OPENAI_BASE_URL`
- `ANTHROPIC_API_KEY`, `ANTHROPIC_BASE_URL`
- `OPENROUTER_API_KEY`, `OPENROUTER_BASE_URL`
- `DEEPSEEK_API_KEY`, `DEEPSEEK_BASE_URL`
- `GEMINI_API_KEY`, `GOOGLE_API_KEY`
- `TOGETHER_API_KEY`, `TOGETHER_BASE_URL`
- `OPENCODE_API_KEY`, `OPENCODE_MODEL`, `OPENCODE_BASE_URL`
- `CLAUDE_CODE_SUBAGENT_MODEL`
