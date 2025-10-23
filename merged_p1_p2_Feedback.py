#!/usr/bin/env python3
r"""
full_pipeline_with_feedback.py

Full single-project pipeline (Phase1 -> Phase2 -> Docker -> Execution) with a feedback loop:
- If execution fails and the error indicates package / python-version issues, ask the LLM to propose fixes.
- Apply fixes (updated requirements.txt and/or python version), rebuild and retry.
- Retry up to MAX_ATTEMPTS (10). Save final report.

Important: PHASE1 and PHASE2 prompts are NOT modified (kept as in your working scripts).
This script adds a separate "fix" prompt used only for the feedback loop.

Author: Generated for you
"""

import re
import gc
import json
import time
import shutil
import subprocess
from pathlib import Path
from typing import Optional, Dict, Any

# -------------------------
# Configuration
# -------------------------
MODEL = "codellama:7b"
MAX_ATTEMPTS = 10
DOCKER_BUILD_TIMEOUT = 600
DOCKER_RUN_TIMEOUT = 120

# -------------------------
# Utility helpers
# -------------------------
def clear_gpu_memory_user_safe():
    try:
        import torch
        torch.cuda.empty_cache()
        gc.collect()
        print("✅ GPU memory cleared via torch.cuda.empty_cache()")
    except Exception:
        gc.collect()
        print("⚠️ torch not installed or GPU clear partially effective")


def run_ollama(prompt_text: str, model: str = MODEL, timeout: int = 600) -> str:
    """
    Run ollama locally and return stdout string.
    """
    clear_gpu_memory_user_safe()
    proc = subprocess.run(
        ["ollama", "run", model],
        input=prompt_text,
        text=True,
        capture_output=True,
        encoding="utf-8",
        timeout=timeout
    )
    # prefer stdout, but include stderr if stdout empty
    return (proc.stdout or proc.stderr or "").strip()


def extract_json_block(text: str) -> str:
    m = re.search(r"\{[\s\S]*\}$", text)
    return m.group(0) if m else text


def ensure_json_valid(raw_text: str) -> dict:
    """Try to decode JSON and fall back to a small wrapper with raw_text on failure."""
    try:
        return json.loads(raw_text)
    except Exception:
        cleaned = re.sub(r"```(?:json)?|```", "", raw_text).strip()
        try:
            return json.loads(cleaned)
        except Exception:
            return {"raw_text": raw_text.strip(), "error": "Invalid JSON structure"}


# -------------------------
# Phase 1 & Phase 2 (unchanged prompts)
# -------------------------
def read_source_files(project_path: Path) -> str:
    code_texts = []
    for py in project_path.rglob("*.py"):
        try:
            code_texts.append(py.read_text(encoding="utf-8"))
        except Exception:
            continue
    return "\n\n".join(code_texts)


def phase1_extract_apis(project_dir: Path, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / f"{project_dir.name}_phase1.json"

    source_code = read_source_files(project_dir)
    prompt = f"""
You are an assistant that extracts API calls from Python source code. 
Given a code snippet, output all fully-qualified APIs used 
(package.module.class.method or package.function).Output strictly as valid JSON only.
Do not include any explanation or commentary.

Source code:
{source_code}

Task:
Identify all API calls used in the project, mapped to their parent packages/modules.
Include functions, methods, classes, attributes, etc.
Include both standard library and third-party packages.

Output format (JSON only):
{{
  "pandas": [
    "pandas.DataFrame",
    "pandas.DataFrame.to_numpy",
    "pandas.DataFrame.ix"
  ],
  "numpy": [
    "numpy.array",
    "numpy.linalg.norm"
  ]
}}
"""
    print("🧠 Running Phase 1 LLM for API extraction...")
    raw_out = run_ollama(prompt)
    json_text = extract_json_block(raw_out)
    data = ensure_json_valid(json_text)
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)
    print(f"✅ Phase 1 complete → {output_path}")
    return output_path


def phase2_infer_dependencies(phase1_json_path: Path, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / f"{phase1_json_path.stem.replace('_phase1','')}_phase2.json"

    phase1_data = ensure_json_valid(phase1_json_path.read_text(encoding="utf-8"))
    prompt = f"""
System Prompt:
You are an expert dependency/version and Python version inference assistant.
You must output ONLY valid JSON — no markdown, no code fences, no explanations, no comments.
Output must strictly match the JSON structure below and be human-readable formatted with 2 spaces of indentation.

Given a structured list of extracted API calls grouped by package and standard library (Python stdlib),
infer:
1. The Python version range (min, max) based on the extracted APIs provided.
2. The most likely version range for each external package used based on the extracted APIs provided.
3. requirements.txt (only third-party packages' recommended_requirements_line).

Requirements for Python version:
- Return:
  - min (minimum Python version required)
  - max (maximum Python version allowed, null if none)
  - evidence (list of short statements linking stdlib API usage to Python version addition/removal)
  - notes (optional assumptions)
Requirements for external packages:
- For each package, return:
  - inferred_version_range (string, e.g., ">=1.2.0,<2.0.0")
  - recommended_requirements_line (string, e.g., "pandas>=1.2.0,<2.0.0")
  - evidence (list of short statements mapping APIs to introduction/deprecation versions)
  - confidence (0.0–1.0)
  - notes (optional: conflicting APIs, assumptions, uncertain items)

Additional field:
- requirements.txt: a list of recommended_requirements_line for all dependencies

General rules:
- If you cannot determine a reliable upper or lower bound, indicate null and explain.
- Prefer conservative ranges to ensure the project runs safely.
- Output **only** valid JSON following the schema below.

User Prompt:
extracted APIs: {json.dumps(phase1_data, indent=2)}

Return ONLY the final JSON object — no markdown, no extra text.
Structure:
{{
  "python_version": {{
    "min": "y",
    "max": "x",
    "evidence": ["pathlib.Path added in Python y → min Python y", "time.clock removed in Python x→ max Python z"],
    "notes": ""
  }},
  "dependencies": {{
    "<package>": {{
      "inferred_version_range": ">=X.Y.Z,<A.B.C" or null,
      "recommended_requirements_line": "pkg>=X.Y.Z,<A.B.C" or null,
      "evidence": ["API X introduced, depricated, removed or changed in vM.N.P"],
      "confidence": 0.0,
      "notes": ""
    }}
  }},
  "requirements.txt": [pkg>=X.Y.Z,<A.B.C]
}}
"""
    print("🧠 Running Phase 2 LLM for dependency inference...")
    raw_out = run_ollama(prompt)
    json_text = extract_json_block(raw_out)
    data = ensure_json_valid(json_text)
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)
    print(f"✅ Phase 2 complete → {output_path}")
    return output_path


# -------------------------
# Docker generation helper
# -------------------------
def pick_python_tag(min_v: Optional[str], max_v: Optional[str]) -> str:
    AVAILABLE = ["3.11", "3.10", "3.9", "3.8", "3.7", "3.6"]
    def to_float(v: Optional[str]) -> Optional[float]:
        if not v:
            return None
        nums = re.findall(r"\d+", v)
        if not nums:
            return None
        major = int(nums[0])
        minor = int(nums[1]) if len(nums) > 1 else 0
        return float(f"{major}.{minor}")
    minf = to_float(min_v)
    maxf = to_float(max_v)
    for tag in AVAILABLE:
        tf = float(tag)
        if (minf is None or tf >= minf) and (maxf is None or tf <= maxf):
            return tag
    return AVAILABLE[0]


def generate_docker_from_phase2(project_dir: Path, phase2_json: Path, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    data = ensure_json_valid(phase2_json.read_text(encoding="utf-8"))

    reqs = []
    rlist = data.get("requirements.txt")
    if isinstance(rlist, list) and rlist:
        reqs = [str(x).strip() for x in rlist if x]
    else:
        deps = data.get("dependencies", {})
        for pkg, meta in deps.items():
            if isinstance(meta, dict):
                rec = meta.get("recommended_requirements_line")
                if rec:
                    reqs.append(str(rec).strip())

    py_min = data.get("python_version", {}).get("min")
    py_max = data.get("python_version", {}).get("max")
    py_tag = pick_python_tag(py_min, py_max)

    # write requirements
    req_path = out_dir / "requirements.txt"
    req_path.write_text("\n".join(reqs) + ("\n" if reqs else ""), encoding="utf-8")

    # copy snippet.py if present
    snippet = project_dir / "snippet.py"
    if snippet.exists():
        shutil.copy2(snippet, out_dir / "snippet.py")
    else:
        # fallback: copy all .py files (keeps behavior conservative)
        for py in project_dir.rglob("*.py"):
            shutil.copy2(py, out_dir / py.name)

    # create Dockerfile
    dockerfile = f"""# Auto-generated Dockerfile
FROM python:{py_tag}-slim
WORKDIR /app
COPY . /app
RUN if [ -s /app/requirements.txt ]; then pip install --no-cache-dir -r /app/requirements.txt; fi
CMD ["python", "snippet.py"]
"""
    (out_dir / "Dockerfile").write_text(dockerfile, encoding="utf-8")
    (out_dir / "docker").write_text(dockerfile, encoding="utf-8")
    return out_dir


# -------------------------
# Build & run Docker and capture outputs
# -------------------------
def build_and_run_docker(docker_dir: Path, image_tag: str) -> Dict[str, Any]:
    result = {
        "build_returncode": None,
        "build_stdout": "",
        "build_stderr": "",
        "build_time_sec": None,
        "run_returncode": None,
        "run_stdout": "",
        "run_stderr": "",
        "run_time_sec": None,
    }

    # Build
    start = time.time()
    try:
        build_proc = subprocess.run(
            ["docker", "build", "-t", image_tag, str(docker_dir)],
            capture_output=True, text=True, timeout=DOCKER_BUILD_TIMEOUT
        )
    except subprocess.TimeoutExpired:
        result["build_returncode"] = -1
        result["build_stderr"] = "BUILD_TIMEOUT"
        result["build_time_sec"] = round(time.time() - start, 2)
        return result

    result["build_returncode"] = build_proc.returncode
    result["build_stdout"] = build_proc.stdout
    result["build_stderr"] = build_proc.stderr
    result["build_time_sec"] = round(time.time() - start, 2)

    if build_proc.returncode != 0:
        return result

    # Run
    start_run = time.time()
    try:
        run_proc = subprocess.run(
            ["docker", "run", "--rm", image_tag],
            capture_output=True, text=True, timeout=DOCKER_RUN_TIMEOUT
        )
    except subprocess.TimeoutExpired:
        result["run_returncode"] = -1
        result["run_stderr"] = "RUN_TIMEOUT"
        result["run_time_sec"] = round(time.time() - start_run, 2)
        return result

    result["run_returncode"] = run_proc.returncode
    result["run_stdout"] = run_proc.stdout
    result["run_stderr"] = run_proc.stderr
    result["run_time_sec"] = round(time.time() - start_run, 2)
    return result


# -------------------------
# Feedback LLM: propose fixes
# -------------------------
def llm_propose_fixes(phase2_json_path: Path, build_stderr: str, run_stderr: str, attempt: int) -> Optional[dict]:
    """
    Provide a prompt to the LLM asking to propose updated requirements and python version bounds
    to try to fix package / python-version related failures.

    Returns a dict like:
    {
      "requirements": ["pandas>=1.2.0,<2.0.0", ...],
      "python_min": "3.8",
      "python_max": null,
      "notes": "..."
    }
    or None if the LLM response can't be parsed.
    """
    phase2_data = ensure_json_valid(phase2_json_path.read_text(encoding="utf-8"))
    # Build a compact context
    context_json = json.dumps(phase2_data, indent=2)[:30_000]  # truncate if huge

    prompt = f"""
You are a dependency-fixing assistant. You received a project's Phase-2 inference (JSON) and the
error messages from a Docker build or runtime attempt. Your task: propose conservative fixes
to requirements and/or python version to try to make the snippet run.

Input:
Phase-2 JSON (inferred dependencies & python info):
{context_json}

Build stderr (truncated):
{build_stderr[:8000]}

Run stderr (truncated):
{run_stderr[:8000]}

Attempt number: {attempt}

Task:
- Examine the errors and the inferred dependencies.
- If failures are related to pip/package resolution (e.g., "Could not find a version", "No matching distribution",
  package not available on PyPI, wheel errors, build failures due to missing system libs), propose updated
  requirements lines or alternatives. Example output lines: "pandas>=1.2.0,<2.0.0".
- If failures look like they stem from Python interpreter incompatibility (e.g., SyntaxError due to f-string in older versions,
  "invalid syntax" with print statements, or `time.clock removed` etc.), propose updated python_min / python_max.
- If multiple fixes recommended, list them all. If you cannot determine any fix, return empty list for requirements and null for python bounds.
- Do NOT change the Phase-2 JSON's dependency names; propose requirement lines that match those names or suggest canonical package names.
- Output ONLY valid JSON (no extra text) with this schema:

{{
  "requirements": ["pkgA>=X.Y,<Z", "..."] or [],
  "python_min": "<major.minor>" or null,
  "python_max": "<major.minor>" or null,
  "notes": "short explanatory notes"
}}

Be conservative. Prefer safe upper-bounds (e.g., "<2.0.0") if removing features would break the project.
"""
    print("🧠 Asking LLM for fixes...")
    raw = run_ollama(prompt, timeout=600)
    json_text = extract_json_block(raw)
    parsed = ensure_json_valid(json_text)
    # validate shape
    if not isinstance(parsed, dict):
        return None
    # ensure fields exist
    reqs = parsed.get("requirements", [])
    if not isinstance(reqs, list):
        reqs = []
    python_min = parsed.get("python_min")
    python_max = parsed.get("python_max")
    notes = parsed.get("notes", "")
    return {"requirements": reqs, "python_min": python_min, "python_max": python_max, "notes": notes}


# -------------------------
# Main loop with feedback
# -------------------------
def full_pipeline_with_feedback(project_dir: Path, base_out: Path):
    # create directories
    phase1_dir = base_out / "phase1_output"
    phase2_dir = base_out / "phase2_output"
    docker_build_dir = base_out / "docker_build"
    execution_dir = base_out / "execution_output"
    for d in (phase1_dir, phase2_dir, docker_build_dir, execution_dir):
        d.mkdir(parents=True, exist_ok=True)

    report = {
        "project": project_dir.name,
        "attempts": [],
        "final_status": None,
        "phase1_json": None,
        "phase2_json": None,
        "final_requirements": None,
        "final_python": None,
        "total_time_sec": None
    }

    t_start = time.time()

    # PHASE 1
    phase1_path = phase1_extract_apis(project_dir, phase1_dir)
    report["phase1_json"] = str(phase1_path.resolve())

    # PHASE 2 (initial)
    phase2_path = phase2_infer_dependencies(phase1_path, phase2_dir)
    report["phase2_json"] = str(phase2_path.resolve())

    # Working copy of phase2 that we will modify during feedback
    working_phase2 = phase2_path

    # attempt loop
    for attempt in range(1, MAX_ATTEMPTS + 1):
        attempt_entry = {
            "attempt": attempt,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "image_tag": f"{project_dir.name.lower()}_trial_{attempt}",
            "build": {},
            "run": {},
            "applied_fixes": None
        }
        print(f"\n🔁 Attempt {attempt} — generate docker & run")

        # Prepare docker build folder for this attempt (unique)
        docker_dir_attempt = docker_build_dir / f"attempt_{attempt}"
        if docker_dir_attempt.exists():
            shutil.rmtree(docker_dir_attempt)
        docker_dir_attempt.mkdir(parents=True, exist_ok=True)

        # Generate docker from current phase2 JSON
        generate_docker_from_phase2(project_dir, working_phase2, docker_dir_attempt)

        # Build and run
        result = build_and_run_docker(docker_dir_attempt, attempt_entry["image_tag"])
        attempt_entry["build"] = {
            "returncode": result["build_returncode"],
            "stdout_tail": (result["build_stdout"] or "")[-800:],
            "stderr_tail": (result["build_stderr"] or "")[-1600:],
            "time_sec": result["build_time_sec"]
        }
        attempt_entry["run"] = {
            "returncode": result["run_returncode"],
            "stdout_tail": (result["run_stdout"] or "")[:800],
            "stderr_tail": (result["run_stderr"] or "")[-1600:],
            "time_sec": result["run_time_sec"]
        }

        report["attempts"].append(attempt_entry)

        # Evaluate success
        build_ok = result["build_returncode"] == 0
        run_ok = result["run_returncode"] == 0

        if build_ok and run_ok:
            print("✅ Build and run succeeded on attempt", attempt)
            report["final_status"] = "success"
            # collect final requirements and python from working_phase2
            final_phase2 = ensure_json_valid(working_phase2.read_text(encoding="utf-8"))
            report["final_requirements"] = final_phase2.get("requirements.txt", [])
            report["final_python"] = final_phase2.get("python_version", {})
            break

        # If failure is not related to packages or python, bail out (LLM cannot help)
        build_stderr = result["build_stderr"] or ""
        run_stderr = result["run_stderr"] or ""

        # Heuristic: decide whether to ask LLM for fixes
        package_issue_patterns = [
            r"Could not find a version that satisfies the requirement",
            r"No matching distribution found for",
            r"Failed building wheel for",
            r"ERROR: Could not build wheels for",
            r"ModuleNotFoundError: No module named",
            r"ImportError: No module named",
            r"WARNING: The wheel package is not available",
            r"ERROR: Command errored out with exit status"
        ]
        python_issue_patterns = [
            r"SyntaxError",
            r"invalid syntax",
            r"NameError: name 'print' is not defined",
            r"TypeError: print\(",
            r"AttributeError: 'str' object has no attribute",
            r"IndentationError",
            r"ValueError: bad interpreter:",
            r"ModuleNotFoundError: No module named '.*'",
        ]

        interested = False
        for pat in package_issue_patterns:
            if re.search(pat, build_stderr, re.IGNORECASE) or re.search(pat, run_stderr, re.IGNORECASE):
                interested = True
                break
        if not interested:
            for pat in python_issue_patterns:
                if re.search(pat, build_stderr, re.IGNORECASE) or re.search(pat, run_stderr, re.IGNORECASE):
                    interested = True
                    break

        if not interested:
            print("❌ Failure doesn't look like package/python-version related (LLM fix unlikely). Stopping.")
            report["final_status"] = "failed_non_recoverable"
            break

        # Ask LLM for fixes
        fixes = llm_propose_fixes(working_phase2, build_stderr, run_stderr, attempt)
        if not fixes:
            print("⚠️ LLM did not return usable fixes. Stopping.")
            report["final_status"] = "failed_no_fixes"
            break

        # Apply fixes: update working_phase2 JSON (a copy), regenerate docker and continue next attempt
        attempt_entry["applied_fixes"] = fixes

        # Load working phase2 data, modify fields
        current_phase2 = ensure_json_valid(working_phase2.read_text(encoding="utf-8"))

        # Update requirements.txt if provided
        new_reqs = fixes.get("requirements", [])
        if isinstance(new_reqs, list) and new_reqs:
            current_phase2["requirements.txt"] = new_reqs

        # Update python_version bounds if provided
        python_min = fixes.get("python_min")
        python_max = fixes.get("python_max")
        if "python_version" not in current_phase2:
            current_phase2["python_version"] = {}
        if python_min:
            current_phase2["python_version"]["min"] = python_min
        if python_max:
            current_phase2["python_version"]["max"] = python_max

        # Save an updated working_phase2 file (overwrite)
        working_phase2.write_text(json.dumps(current_phase2, indent=2), encoding="utf-8")
        print(f"🔧 Applied fixes from LLM (attempt {attempt}). Will retry.")

        # loop continues

    # End attempts
    total_time = round(time.time() - t_start, 2)
    report["total_time_sec"] = total_time

    # Save final report to execution_dir
    out_report_path = execution_dir / f"{project_dir.name}_final_report.json"
    out_report_path.parent.mkdir(parents=True, exist_ok=True)
    with out_report_path.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)

    print("\n=== Pipeline finished ===")
    print("Final status:", report["final_status"])
    print("Report saved to:", out_report_path.resolve())

    return out_report_path


# -------------------------
# CLI-style invocation
# -------------------------
if __name__ == "__main__":
    # Change these paths as needed
    PROJECT_DIR = Path(r"C:\Users\sadma\Documents\Thesis\Project LLM\Example_Project")
    BASE_OUT = Path(r"C:\Users\sadma\Documents\Thesis\Project LLM\Example_Run_Merged")

    BASE_OUT.mkdir(parents=True, exist_ok=True)
    print(f"Starting full pipeline with feedback for: {PROJECT_DIR}")
    full_pipeline_with_feedback(PROJECT_DIR, BASE_OUT)
