#!/usr/bin/env python3
"""
full_pipeline_agentic.py

Agentic, LLM-driven full pipeline:
- Phase1: API extraction (LLM)
- Phase2: Dependency & Python inference (LLM)
- Docker generation, build & run
- Agentic feedback loop with Meta-Agent and specialized LLM agents:
    dependency_agent_llm, code_agent_llm, python_agent_llm, system_agent_llm
- Metrics and final JSON report.

Configure MODEL, PROJECT_DIR, BASE_OUT for your environment.
Requires: docker CLI, ollama CLI (or adapt run_ollama to your LLM), Python 3.8+.
"""

import re
import gc
import json
import time
import shutil
import subprocess
from pathlib import Path
from typing import Optional, Dict, Any, List

# -------------------------
# Configuration
# -------------------------
MODEL = "codellama:7b"                   # change to your model/endpoint if needed
MAX_ATTEMPTS = 10
DOCKER_BUILD_TIMEOUT = 600
DOCKER_RUN_TIMEOUT = 120
LLM_TIMEOUT = 600

# -------------------------
# Utility helpers (existing)
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


def run_ollama(prompt_text: str, model: str = MODEL, timeout: int = LLM_TIMEOUT) -> str:
    """
    Run ollama (local) and return stdout string.
    If you use a different LLM runner, adapt this function.
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
# Phase 1 & Phase 2 (unchanged prompts - reuse from your script)
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
  - inferred_version_range (string, e.g., ">=1.2.0,<2.0.0") or null,
  - recommended_requirements_line (string, e.g., "pandas>=1.2.0,<2.0.0") or null,
  - evidence (list of short statements mapping APIs to introduction/deprecation versions),
  - confidence (0.0–1.0),
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
  "requirements.txt": [ "pkg>=X.Y.Z,<A.B.C" ]
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
# Docker generation helpers (unchanged, slightly extended)
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
    """
    Create a docker build folder (out_dir) populated with code and requirements derived from phase2_json.
    Does not overwrite if out_dir has code patches (caller may patch files after this call).
    """
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
# Safety checks
# -------------------------
def safe_requirements_line(s: str) -> bool:
    # allow lines like "pkg", "pkg==1.2.3", "pkg>=1.0.0,<2.0.0"
    if not isinstance(s, str):
        return False
    s = s.strip()
    if len(s) == 0 or len(s) > 200:
        return False
    # forbid shell-like constructs
    for bad in [";","&&","|", "`", "$(", "rm -rf", "curl ", "wget "]:
        if bad in s:
            return False
    return bool(re.match(r"^[A-Za-z0-9_.-]+([<>=!~]=?.+)?$", s))


def safe_code(content: str) -> bool:
    if not isinstance(content, str) or len(content) == 0:
        return False
    # simple forbidden checks
    forbidden = ["rm -rf", "subprocess.Popen(['curl'", "open('/etc", "ssh ", "socket(", "nmap", "os.system("]
    for f in forbidden:
        if f in content:
            return False
    # limit size
    if len(content) > 200_000:
        return False
    return True


# -------------------------
# Meta-Agent (classify failure)
# -------------------------
def meta_agent_decide(build_stderr: str, run_stderr: str, phase2_json: dict) -> str:
    """
    Simple heuristic-based triage with fallback to an LLM-guided decision if ambiguous.
    Returns one of: "dependency", "python", "system", "code", "unknown"
    """
    # heuristics
    combined = (build_stderr or "") + "\n" + (run_stderr or "")
    combined_lower = combined.lower()

    dep_patterns = ["module not found", "no module named", "no matching distribution", "could not find a version", "failed building wheel", "distribution not found"]
    python_patterns = ["syntaxerror", "invalid syntax", "time.clock", "bad interpreter", "print("]
    system_patterns = ["gcc", "fatal error:", "lib", "apt-get", "ld.lld", "unable to find the vcvars", "wheel build failed"]
    code_patterns = ["typeerror", "attributeerror", "nameerror", "indexerror", "keyerror", "valueerror"]

    for p in dep_patterns:
        if re.search(p, combined_lower):
            return "dependency"
    for p in python_patterns:
        if re.search(p, combined_lower):
            return "python"
    for p in system_patterns:
        if re.search(p, combined_lower):
            return "system"
    for p in code_patterns:
        if re.search(p, combined_lower):
            return "code"

    # fallback: ask LLM to classify
    prompt = f"""
You are an assistant that classifies the root cause of a failed Docker build/run for Python code.
Return ONLY a single word among: "dependency", "python", "system", "code", or "unknown".

Build stderr (truncated):
{(build_stderr or '')[:6000]}

Run stderr (truncated):
{(run_stderr or '')[:6000]}

Phase2 summary (truncated):
{json.dumps(phase2_json)[:4000]}

Which of these is the most likely root cause? Return only one word.
"""
    try:
        raw = run_ollama(prompt)
        answer = raw.strip().lower()
        # sanitize
        for choice in ["dependency", "python", "system", "code", "unknown"]:
            if choice in answer:
                return choice
    except Exception:
        pass
    return "unknown"


# -------------------------
# Agent: Dependency (LLM-only)
# -------------------------
def dependency_agent_llm(phase2_json_path: Path, current_requirements_text: str,
                         build_stderr: str, run_stderr: str, attempt: int,
                         max_candidates: int = 3) -> Optional[dict]:
    phase2_text = phase2_json_path.read_text(encoding="utf-8")
    ctx_phase2 = phase2_text[:30000]
    build_err = (build_stderr or "")[:8000]
    run_err = (run_stderr or "")[:8000]
    prompt = f"""
You are a Dependency Repair Agent. Output ONLY valid JSON (no markdown, no fences).

Inputs:
- phase2_json: {ctx_phase2}
- current_requirements_text:
{current_requirements_text}

- build_stderr (truncated):
{build_err}

- run_stderr (truncated):
{run_err}

- attempt: {attempt}

Task:
1. Inspect the phase2_json, current requirements, and errors.
2. Propose up to {max_candidates} conservative candidate fixes as lists of requirements lines.
3. Provide confidence (0.0-1.0) and rationale for each candidate.
4. Optionally propose python_min/python_max.

Output Schema:
{{
  "candidates": [
    {{
      "id": 1,
      "requirements_lines": ["pkgA>=1.2.0,<2.0.0", "pkgB==0.9.1"],
      "change_type": ["add","modify","pin"],
      "confidence": 0.0,
      "rationale": ["evidence line 1"]
    }}
  ],
  "recommended_python_min": null,
  "recommended_python_max": null,
  "notes": ""
}}
"""
    print("🧠 dependency_agent_llm: asking LLM for candidate requirement fixes...")
    raw = run_ollama(prompt)
    json_text = extract_json_block(raw)
    parsed = ensure_json_valid(json_text)
    if not isinstance(parsed, dict) or "candidates" not in parsed:
        return None
    # sanitize candidates
    good = []
    for c in parsed.get("candidates", [])[:max_candidates]:
        if not isinstance(c, dict):
            continue
        reqs = c.get("requirements_lines", [])
        if not isinstance(reqs, list) or len(reqs) == 0:
            continue
        sanitized = [r.strip() for r in reqs if isinstance(r, str) and safe_requirements_line(r)]
        if not sanitized:
            continue
        c["requirements_lines"] = sanitized
        try:
            c["confidence"] = float(c.get("confidence", 0.0))
        except Exception:
            c["confidence"] = 0.0
        good.append(c)
    parsed["candidates"] = good
    return parsed


# -------------------------
# Agent: Code (LLM-only)
# -------------------------
def code_agent_llm(project_dir: Path, error_log: str, file_list: Optional[list] = None,
                   python_hint: Optional[str] = None, requirements_text: str = "") -> Optional[dict]:
    code_blob_parts = []
    files = file_list or [str(p.relative_to(project_dir)) for p in project_dir.rglob("*.py")]
    max_chars = 20000
    for fname in files:
        try:
            content = (project_dir / fname).read_text(encoding="utf-8")
        except Exception:
            continue
        snippet = f"--- {fname} ---\n{content}\n"
        code_blob_parts.append(snippet)
        if sum(len(p) for p in code_blob_parts) > max_chars:
            break
    code_blob = "\n".join(code_blob_parts)[:max_chars]
    prompt = f"""
You are a Code Repair Agent. Output ONLY valid JSON (no markdown, no fences).

Inputs:
- error_log (truncated):
{(error_log or '')[:8000]}

- code_files and content:
{code_blob}

- environment_hint:
Python version: {python_hint}
Requirements: {requirements_text[:2000]}

Task:
1. Diagnose the error.
2. Propose up to 2 minimal candidate patches, each returning full modified file content.
3. If new dependencies are needed, mention them in notes.

Schema:
{{
  "candidates": [
    {{
      "id": 1,
      "modified_files": {{
         "path/to/file.py": "full new file content here"
      }},
      "diff_summary": ["file.py: added import X"],
      "confidence": 0.0,
      "rationale": ["line 1 of evidence"]
    }}
  ],
  "notes": ""
}}
"""
    print("🧠 code_agent_llm: asking LLM for code patch candidates...")
    raw = run_ollama(prompt)
    json_text = extract_json_block(raw)
    parsed = ensure_json_valid(json_text)
    if not isinstance(parsed, dict) or "candidates" not in parsed:
        return None
    cands = []
    for c in parsed.get("candidates", [])[:2]:
        if not isinstance(c, dict):
            continue
        mods = c.get("modified_files", {})
        if not isinstance(mods, dict) or not mods:
            continue
        ok = True
        for path, content in list(mods.items()):
            if not isinstance(content, str) or not safe_code(content):
                ok = False
        if not ok:
            continue
        try:
            c["confidence"] = float(c.get("confidence", 0.0))
        except Exception:
            c["confidence"] = 0.0
        cands.append(c)
    parsed["candidates"] = cands
    return parsed


# -------------------------
# Agent: Python version (LLM-only)
# -------------------------
def python_agent_llm(build_stderr: str, run_stderr: str, phase2_json: dict) -> Optional[dict]:
    prompt = f"""
You are a Python-Version Repair Agent. Output ONLY valid JSON (no markdown, no fences).

Inputs:
- build_stderr (truncated):
{(build_stderr or '')[:4000]}

- run_stderr (truncated):
{(run_stderr or '')[:4000]}

- phase2 summary (truncated):
{json.dumps(phase2_json)[:4000]}

Task:
1. Determine if the failure is due to Python interpreter incompatibility.
2. If so, recommend python_min and/or python_max (string like "3.10" or null).
3. Provide a short rationale and confidence (0.0-1.0).

Schema:
{{
  "recommended_python_min": "3.10" or null,
  "recommended_python_max": null or "3.11",
  "confidence": 0.0,
  "rationale": "short explanation"
}}
"""
    print("🧠 python_agent_llm: asking LLM for python version suggestion...")
    raw = run_ollama(prompt)
    json_text = extract_json_block(raw)
    parsed = ensure_json_valid(json_text)
    if not isinstance(parsed, dict):
        return None
    try:
        parsed["confidence"] = float(parsed.get("confidence", 0.0))
    except Exception:
        parsed["confidence"] = 0.0
    return parsed


# -------------------------
# Agent: System (LLM-only)
# -------------------------
def system_agent_llm(build_stderr: str, run_stderr: str) -> Optional[dict]:
    prompt = f"""
You are a System-Dependency Repair Agent. Output ONLY valid JSON (no markdown, no fences).

Inputs:
- build_stderr (truncated):
{(build_stderr or '')[:6000]}

- run_stderr (truncated):
{(run_stderr or '')[:6000]}

Task:
1. Identify missing system-level packages (apt packages) required during pip builds or runtime.
2. Provide a short list of apt-get install lines to add into the Dockerfile before pip install.
3. Provide confidence (0.0-1.0) and rationale.

Schema:
{{
  "apt_lines": ["apt-get update && apt-get install -y pkg1 pkg2", "..."],
  "confidence": 0.0,
  "rationale": "short reason"
}}
"""
    print("🧠 system_agent_llm: asking LLM for system package suggestions...")
    raw = run_ollama(prompt)
    json_text = extract_json_block(raw)
    parsed = ensure_json_valid(json_text)
    if not isinstance(parsed, dict):
        return None
    try:
        parsed["confidence"] = float(parsed.get("confidence", 0.0))
    except Exception:
        parsed["confidence"] = 0.0
    # quick sanitize of apt_lines
    apt_lines = parsed.get("apt_lines", [])
    if not isinstance(apt_lines, list):
        parsed["apt_lines"] = []
    else:
        cleaned = []
        for l in apt_lines:
            if isinstance(l, str) and "apt-get" in l and len(l) < 1000:
                cleaned.append(l.strip())
        parsed["apt_lines"] = cleaned
    return parsed


# -------------------------
# Helpers: apply fixes & patching
# -------------------------
def apply_requirements_to_phase2(phase2_path: Path, new_reqs: List[str]):
    cur = ensure_json_valid(phase2_path.read_text(encoding="utf-8"))
    cur["requirements.txt"] = new_reqs
    phase2_path.write_text(json.dumps(cur, indent=2), encoding="utf-8")


def apply_python_bounds_to_phase2(phase2_path: Path, python_min: Optional[str], python_max: Optional[str]):
    cur = ensure_json_valid(phase2_path.read_text(encoding="utf-8"))
    if "python_version" not in cur:
        cur["python_version"] = {}
    if python_min:
        cur["python_version"]["min"] = python_min
    if python_max:
        cur["python_version"]["max"] = python_max
    phase2_path.write_text(json.dumps(cur, indent=2), encoding="utf-8")


def insert_apt_lines_in_dockerfile(docker_dir: Path, apt_lines: List[str]):
    df = (docker_dir / "Dockerfile").read_text(encoding="utf-8")
    # insert apt lines after WORKDIR /app line or before pip install
    insertion = "\n".join([f"RUN {l}" for l in apt_lines]) + "\n"
    if "RUN if [ -s /app/requirements.txt ]; then pip install" in df:
        df = df.replace("RUN if [ -s /app/requirements.txt ]; then pip install --no-cache-dir -r /app/requirements.txt; fi",
                        insertion + "RUN if [ -s /app/requirements.txt ]; then pip install --no-cache-dir -r /app/requirements.txt; fi")
    else:
        df = df + "\n" + insertion
    (docker_dir / "Dockerfile").write_text(df, encoding="utf-8")


def replace_python_base_image_in_dockerfile(docker_dir: Path, new_python_tag: str):
    dockerfile_path = docker_dir / "Dockerfile"
    df = dockerfile_path.read_text(encoding="utf-8")
    df = re.sub(r"FROM python:\d+\.\d+-slim", f"FROM python:{new_python_tag}-slim", df)
    dockerfile_path.write_text(df, encoding="utf-8")


def apply_code_patch_to_docker_build(docker_dir: Path, modified_files: Dict[str, str]):
    """
    Write modified file contents into docker build folder.
    modified_files keys are relative paths like 'snippet.py' or 'module/foo.py'
    """
    for rel_path, content in modified_files.items():
        dest = docker_dir / rel_path
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(content, encoding="utf-8")


# -------------------------
# Metrics summarizer
# -------------------------
def compute_metrics_from_report(report: dict) -> dict:
    attempts = report.get("attempts", [])
    total_attempts = len(attempts)
    success_attempt = None
    for a in attempts:
        if a.get("build", {}).get("returncode") == 0 and a.get("run", {}).get("returncode") == 0:
            success_attempt = a["attempt"]
            break
    metrics = {
        "total_attempts": total_attempts,
        "success_attempt": success_attempt,
        "attempts_to_success": (success_attempt if success_attempt is not None else None),
        "total_time_sec": report.get("total_time_sec"),
        "avg_build_time_sec": None,
        "avg_run_time_sec": None,
        "avg_chosen_confidence": None,
        "fix_type_counts": {},
    }
    build_times = []
    run_times = []
    confidences = []
    fix_counts = {}
    for a in attempts:
        b = a.get("build", {})
        r = a.get("run", {})
        if b.get("time_sec") is not None:
            build_times.append(b["time_sec"])
        if r.get("time_sec") is not None:
            run_times.append(r["time_sec"])
        applied = a.get("applied_fixes") or {}
        agent = a.get("applied_agent")
        if agent:
            fix_counts[agent] = fix_counts.get(agent, 0) + 1
        conf = None
        try:
            conf = float(applied.get("chosen_confidence", 0.0)) if isinstance(applied, dict) else None
        except Exception:
            conf = None
        if conf is not None:
            confidences.append(conf)
    metrics["avg_build_time_sec"] = round(sum(build_times)/len(build_times), 2) if build_times else None
    metrics["avg_run_time_sec"] = round(sum(run_times)/len(run_times), 2) if run_times else None
    metrics["avg_chosen_confidence"] = round(sum(confidences)/len(confidences), 3) if confidences else None
    metrics["fix_type_counts"] = fix_counts
    return metrics


# -------------------------
# Main loop with meta-agent + specialized agents
# -------------------------
def full_pipeline_agentic(project_dir: Path, base_out: Path):
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

    # provide current requirements text helper
    def read_current_requirements_text():
        try:
            cur = ensure_json_valid(working_phase2.read_text(encoding="utf-8"))
            r = cur.get("requirements.txt", [])
            if isinstance(r, list):
                return "\n".join(r) + ("\n" if r else "")
            return str(r)
        except Exception:
            return ""

    # attempt loop
    for attempt in range(1, MAX_ATTEMPTS + 1):
        attempt_entry = {
            "attempt": attempt,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "image_tag": f"{project_dir.name.lower()}_trial_{attempt}",
            "build": {},
            "run": {},
            "applied_agent": None,
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
            final_phase2 = ensure_json_valid(working_phase2.read_text(encoding="utf-8"))
            report["final_requirements"] = final_phase2.get("requirements.txt", [])
            report["final_python"] = final_phase2.get("python_version", {})
            break

        # Collect errors
        build_stderr = result["build_stderr"] or ""
        run_stderr = result["run_stderr"] or ""

        # Decide if it's recoverable (dependency/python/system/code) using meta-agent
        phase2_json = ensure_json_valid(working_phase2.read_text(encoding="utf-8"))
        decision = meta_agent_decide(build_stderr, run_stderr, phase2_json)
        print(f"🧭 Meta-Agent decision: {decision}")

        if decision == "unknown":
            print("❌ Meta-Agent could not determine a recoverable cause. Stopping.")
            report["final_status"] = "failed_unknown"
            break

        # choose and call the specialized agent
        applied = {"chosen_candidate": None, "chosen_confidence": None, "notes": None}
        attempt_entry["applied_agent"] = decision

        if decision == "dependency":
            cur_reqs_text = read_current_requirements_text()
            dep_out = dependency_agent_llm(working_phase2, cur_reqs_text, build_stderr, run_stderr, attempt)
            if not dep_out or not dep_out.get("candidates"):
                print("⚠️ Dependency agent returned no candidates. Stopping.")
                report["final_status"] = "failed_no_fixes"
                attempt_entry["applied_fixes"] = dep_out
                break
            # selection policy: pick highest confidence candidate with confidence >= 0.5 else pick first
            candidates = sorted(dep_out["candidates"], key=lambda x: x.get("confidence", 0.0), reverse=True)
            chosen = candidates[0]
            applied["chosen_candidate"] = chosen.get("id")
            applied["chosen_confidence"] = chosen.get("confidence", 0.0)
            applied["notes"] = chosen.get("rationale", [])
            # apply to phase2
            apply_requirements_to_phase2(working_phase2, chosen["requirements_lines"])
            attempt_entry["applied_fixes"] = {"requirements": chosen["requirements_lines"], "rationale": chosen.get("rationale", [])}

        elif decision == "code":
            cur_reqs_text = read_current_requirements_text()
            code_out = code_agent_llm(project_dir, run_stderr or build_stderr, file_list=None, python_hint=None, requirements_text=cur_reqs_text)
            if not code_out or not code_out.get("candidates"):
                print("⚠️ Code agent returned no candidates. Stopping.")
                report["final_status"] = "failed_no_fixes"
                attempt_entry["applied_fixes"] = code_out
                break
            candidates = sorted(code_out["candidates"], key=lambda x: x.get("confidence", 0.0), reverse=True)
            chosen = candidates[0]
            applied["chosen_candidate"] = chosen.get("id")
            applied["chosen_confidence"] = chosen.get("confidence", 0.0)
            applied["notes"] = chosen.get("rationale", [])
            # apply code patches to the docker build folder before rebuilding
            apply_code_patch_to_docker_build(docker_dir_attempt, chosen["modified_files"])
            attempt_entry["applied_fixes"] = {"modified_files": list(chosen["modified_files"].keys()), "diff_summary": chosen.get("diff_summary", []), "rationale": chosen.get("rationale", [])}

        elif decision == "python":
            py_out = python_agent_llm(build_stderr, run_stderr, phase2_json)
            if not py_out:
                print("⚠️ Python agent returned no suggestion. Stopping.")
                report["final_status"] = "failed_no_fixes"
                attempt_entry["applied_fixes"] = py_out
                break
            applied["chosen_confidence"] = py_out.get("confidence", 0.0)
            applied["notes"] = py_out.get("rationale", "")
            # apply python bounds to phase2 and then change Dockerfile in the attempt folder
            minp = py_out.get("recommended_python_min")
            maxp = py_out.get("recommended_python_max")
            if minp or maxp:
                apply_python_bounds_to_phase2(working_phase2, minp, maxp)
                # regenerate dockerfile for this attempt folder (overwrite)
                generate_docker_from_phase2(project_dir, working_phase2, docker_dir_attempt)
            attempt_entry["applied_fixes"] = {"python_min": minp, "python_max": maxp, "rationale": py_out.get("rationale", "")}

        elif decision == "system":
            sys_out = system_agent_llm(build_stderr, run_stderr)
            if not sys_out:
                print("⚠️ System agent returned no suggestion. Stopping.")
                report["final_status"] = "failed_no_fixes"
                attempt_entry["applied_fixes"] = sys_out
                break
            applied["chosen_confidence"] = sys_out.get("confidence", 0.0)
            applied["notes"] = sys_out.get("rationale", "")
            # apply apt lines to Dockerfile in attempt folder
            if sys_out.get("apt_lines"):
                insert_apt_lines_in_dockerfile(docker_dir_attempt, sys_out.get("apt_lines"))
            attempt_entry["applied_fixes"] = {"apt_lines": sys_out.get("apt_lines", []), "rationale": sys_out.get("rationale", "")}

        else:
            print("❌ Unknown decision from meta-agent; stopping.")
            report["final_status"] = "failed_unknown"
            break

        # record applied summary
        attempt_entry["applied_fixes"] = attempt_entry.get("applied_fixes") or {}
        attempt_entry["applied_fixes"]["meta"] = applied
        print(f"🔧 Applied fixes from {decision} agent. Will retry (attempt {attempt + 1} if any).")

        # loop continues, next iteration will rebuild using updated phase2 or patched docker_dir
        # note: We intentionally don't immediately rebuild within same iteration; next loop iteration regenerates build folder

    # End attempts
    total_time = round(time.time() - t_start, 2)
    report["total_time_sec"] = total_time

    # Save final report to execution_dir
    out_report_path = execution_dir / f"{project_dir.name}_final_report.json"
    out_report_path.parent.mkdir(parents=True, exist_ok=True)
    # add computed metrics
    report["metrics"] = compute_metrics_from_report(report)
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
    # Edit these paths to point to your project and desired output folder
    PROJECT_DIR = Path(r"C:\Users\sadma\Documents\Thesis\Project LLM\Example_Project")
    BASE_OUT = Path(r"C:\Users\sadma\Documents\Thesis\Project LLM\Example_Run_Agentic")

    BASE_OUT.mkdir(parents=True, exist_ok=True)
    print(f"Starting full agentic pipeline for: {PROJECT_DIR}")
    final = full_pipeline_agentic(PROJECT_DIR, BASE_OUT)
    print("Done. Final report:", final)
