#!/usr/bin/env python3
"""
run_dataset_agentic.py

Batch-run an agentic feedback pipeline for a dataset of pre-generated Dockerized Python projects.
- Scans ROOT_DIR for project folders (each must contain snippet.py, requirements.txt, Dockerfile).
- For each project: attempts to build & run Docker image; on failure, invokes Meta-Agent and specialized agents
  (Dependency, Python, System, Code) to propose LLM-based fixes, applies fixes to a per-attempt build folder,
  and retries up to MAX_ATTEMPTS.
- Produces per-project JSON reports and a CSV with columns: project, metrics (JSON serialized).

Configure ROOT_DIR and BASE_OUT. Requires docker CLI and ollama CLI (or modify run_ollama).
"""
import re
import gc
import json
import time
import shutil
import subprocess
import csv
import uuid
from pathlib import Path
from typing import Optional, Dict, Any, List

# -------------------------
# Configuration
# -------------------------
MODEL = "codellama:7b"                   # adapt to your LLM runner/model
MAX_ATTEMPTS = 10
DOCKER_BUILD_TIMEOUT = 600
DOCKER_RUN_TIMEOUT = 120
LLM_TIMEOUT = 600

# Root dataset directory (change to your dataset path)
ROOT_DIR = Path(r"C:\Users\sadma\Documents\Thesis\Project LLM\Docker_HG2.9k_Dataset")

# Output directory for reports and CSV
BASE_OUT = Path(r"C:\Users\sadma\Documents\Thesis\Project LLM\Docker_HG2.9k_agent_Results")
BASE_OUT.mkdir(parents=True, exist_ok=True)

CSV_PATH = BASE_OUT / "hg2_9k_results.csv"
PER_PROJECT_DIR = BASE_OUT / "per_project_reports"
PER_PROJECT_DIR.mkdir(parents=True, exist_ok=True)

CLEANUP_IMAGES = True  # set False to keep Docker images for debugging

# -------------------------
# Utility helpers
# -------------------------
def clear_gpu_memory_user_safe():
    try:
        import torch
        torch.cuda.empty_cache()
        gc.collect()
    except Exception:
        gc.collect()


def run_ollama(prompt_text: str, model: str = MODEL, timeout: int = LLM_TIMEOUT) -> str:
    """
    Run ollama (local) and return stdout string.
    Adapt this if using another LLM runner (OpenAI, local server, etc.).
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
    try:
        return json.loads(raw_text)
    except Exception:
        cleaned = re.sub(r"```(?:json)?|```", "", raw_text).strip()
        try:
            return json.loads(cleaned)
        except Exception:
            return {"raw_text": raw_text.strip(), "error": "Invalid JSON structure"}


# -------------------------
# Safety checks
# -------------------------
def safe_requirements_line(s: str) -> bool:
    if not isinstance(s, str):
        return False
    s = s.strip()
    if len(s) == 0 or len(s) > 200:
        return False
    for bad in [";","&&","|", "`", "$(", "rm -rf", "curl ", "wget "]:
        if bad in s:
            return False
    return bool(re.match(r"^[A-Za-z0-9_.-]+([<>=!~]=?.+)?$", s))


def safe_code(content: str) -> bool:
    if not isinstance(content, str) or len(content) == 0:
        return False
    forbidden = ["rm -rf", "subprocess.Popen(['curl'", "open('/etc", "ssh ", "socket(", "nmap", "os.system("]
    for f in forbidden:
        if f in content:
            return False
    if len(content) > 200_000:
        return False
    return True


# -------------------------
# Docker build/run helpers
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

    start = time.time()
    try:
        build_proc = subprocess.run(
            ["docker", "build", "-t", image_tag, str(docker_dir)],
            capture_output=True, text=True, timeout=DOCKER_BUILD_TIMEOUT
        )
    except subprocess.TimeoutExpired:
        result.update({"build_returncode": -1, "build_stderr": "BUILD_TIMEOUT", "build_time_sec": round(time.time() - start, 2)})
        return result
    except Exception as e:
        result.update({"build_returncode": -2, "build_stderr": f"BUILD_EXCEPTION: {e}", "build_time_sec": round(time.time() - start, 2)})
        return result

    result["build_returncode"] = build_proc.returncode
    result["build_stdout"] = build_proc.stdout
    result["build_stderr"] = build_proc.stderr
    result["build_time_sec"] = round(time.time() - start, 2)

    if build_proc.returncode != 0:
        return result

    start_run = time.time()
    try:
        run_proc = subprocess.run(
            ["docker", "run", "--rm", image_tag],
            capture_output=True, text=True, timeout=DOCKER_RUN_TIMEOUT
        )
    except subprocess.TimeoutExpired:
        result.update({"run_returncode": -1, "run_stderr": "RUN_TIMEOUT", "run_time_sec": round(time.time() - start_run, 2)})
        return result
    except Exception as e:
        result.update({"run_returncode": -2, "run_stderr": f"RUN_EXCEPTION: {e}", "run_time_sec": round(time.time() - start_run, 2)})
        return result

    result["run_returncode"] = run_proc.returncode
    result["run_stdout"] = run_proc.stdout
    result["run_stderr"] = run_proc.stderr
    result["run_time_sec"] = round(time.time() - start_run, 2)
    return result


def docker_remove_image(tag: str):
    try:
        subprocess.run(["docker", "rmi", "-f", tag], capture_output=True, text=True, timeout=30)
    except Exception:
        pass


# -------------------------
# Meta-Agent classification
# -------------------------
def meta_agent_decide(build_stderr: str, run_stderr: str, phase2_json: dict) -> str:
    combined = (build_stderr or "") + "\n" + (run_stderr or "")
    combined_lower = combined.lower()

    dep_patterns = ["module not found", "no module named", "no matching distribution", "could not find a version",
                    "failed building wheel", "distribution not found"]
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

    # fallback: light LLM classification
    prompt = f"""
You are an assistant that classifies the root cause of a failed Docker build/run for Python code.
Return ONLY one of: "dependency", "python", "system", "code", or "unknown".

Build stderr (truncated):
{(build_stderr or '')[:6000]}

Run stderr (truncated):
{(run_stderr or '')[:6000]}

Phase2 summary (truncated):
{json.dumps(phase2_json)[:4000]}

Return one word.
"""
    try:
        raw = run_ollama(prompt)
        ans = raw.strip().lower()
        for choice in ["dependency", "python", "system", "code", "unknown"]:
            if choice in ans:
                return choice
    except Exception:
        pass
    return "unknown"


# -------------------------
# Agent implementations (LLM-driven)
# These versions accept project-level inputs (no strict dependency on Phase2 files).
# -------------------------
def dependency_agent_llm_project(project_dir: Path, build_stderr: str, run_stderr: str, attempt: int, max_candidates: int = 3) -> Optional[dict]:
    """
    Dependency agent that uses the project's requirements.txt and optional phase2 json if present.
    Returns parsed JSON with 'candidates' list similar to earlier schema.
    """
    # try to locate a phase2 json in project (pattern *_phase2.json)
    phase2_path = None
    for p in project_dir.glob("*_phase2.json"):
        phase2_path = p
        break
    phase2_text = ""
    if phase2_path and phase2_path.exists():
        phase2_text = phase2_path.read_text(encoding="utf-8")[:30000]
    else:
        # build minimal phase2 summary from requirements.txt if available
        req_text = ""
        if (project_dir / "requirements.txt").exists():
            req_text = (project_dir / "requirements.txt").read_text(encoding="utf-8")[:2000]
        phase2_text = json.dumps({"requirements.txt": req_text})

    current_requirements_text = ""
    try:
        current_requirements_text = (project_dir / "requirements.txt").read_text(encoding="utf-8")
    except Exception:
        current_requirements_text = ""

    prompt = f"""
You are a Dependency Repair Agent. Output ONLY valid JSON (no markdown, no fences).

Inputs:
- phase2_json (or summary): {phase2_text}
- current_requirements_text:
{current_requirements_text}

- build_stderr (truncated):
{(build_stderr or '')[:8000]}

- run_stderr (truncated):
{(run_stderr or '')[:8000]}

- attempt: {attempt}

Task:
1. Propose up to {max_candidates} conservative candidate fixes as lists of requirements lines.
2. Provide confidence (0.0-1.0) and short rationale for each.
3. If no fix is possible, return empty candidates list.

Output schema:
{{ "candidates": [ {{ "id": 1, "requirements_lines": ["pkg>=1.2.0"], "change_type":["add"], "confidence":0.0, "rationale":["..."] }} ], "notes": "" }}
"""
    try:
        raw = run_ollama(prompt)
    except Exception as e:
        print("Dependency agent LLM error:", e)
        return None

    json_text = extract_json_block(raw)
    parsed = ensure_json_valid(json_text)
    if not isinstance(parsed, dict) or "candidates" not in parsed:
        return None

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


def python_agent_llm_project(build_stderr: str, run_stderr: str, project_dir: Path) -> Optional[dict]:
    prompt = f"""
You are a Python-Version Repair Agent. Output ONLY valid JSON.

Inputs:
- build_stderr (truncated):
{(build_stderr or '')[:4000]}

- run_stderr (truncated):
{(run_stderr or '')[:4000]}

- project hint: files in project directory (listing):
{json.dumps([p.name for p in project_dir.iterdir() if p.is_file()])}

Task:
1. If failure looks like interpreter incompatibility, recommend python_min and/or python_max (e.g., "3.10").
2. Provide confidence (0.0-1.0) and short rationale.

Schema:
{{ "recommended_python_min": "3.10" or null, "recommended_python_max": null or "3.11", "confidence": 0.0, "rationale": "..." }}
"""
    try:
        raw = run_ollama(prompt)
    except Exception as e:
        print("Python agent LLM error:", e)
        return None
    parsed = ensure_json_valid(extract_json_block(raw))
    if not isinstance(parsed, dict):
        return None
    try:
        parsed["confidence"] = float(parsed.get("confidence", 0.0))
    except Exception:
        parsed["confidence"] = 0.0
    return parsed


def system_agent_llm_project(build_stderr: str, run_stderr: str, project_dir: Path) -> Optional[dict]:
    prompt = f"""
You are a System-Dependency Repair Agent. Output ONLY valid JSON.

Inputs:
- build_stderr (truncated):
{(build_stderr or '')[:6000]}

- run_stderr (truncated):
{(run_stderr or '')[:6000]}

- project hint: file listing:
{json.dumps([p.name for p in project_dir.iterdir() if p.is_file()])}

Task:
1. Identify missing system packages needed to build or run.
2. Provide apt-get lines to insert into Dockerfile (before pip install).
3. Provide confidence and rationale.

Schema:
{{ "apt_lines": ["apt-get update && apt-get install -y pkg1 pkg2"], "confidence":0.0, "rationale":"..." }}
"""
    try:
        raw = run_ollama(prompt)
    except Exception as e:
        print("System agent LLM error:", e)
        return None
    parsed = ensure_json_valid(extract_json_block(raw))
    if not isinstance(parsed, dict):
        return None
    parsed.setdefault("apt_lines", [])
    try:
        parsed["confidence"] = float(parsed.get("confidence", 0.0))
    except Exception:
        parsed["confidence"] = 0.0
    # sanitize apt_lines
    cleaned = []
    for l in parsed.get("apt_lines", []):
        if isinstance(l, str) and "apt-get" in l and len(l) < 1000:
            cleaned.append(l.strip())
    parsed["apt_lines"] = cleaned
    return parsed


def code_agent_llm_project(project_dir: Path, build_stderr: str, run_stderr: str, file_list: Optional[list] = None) -> Optional[dict]:
    # prepare code blob from project files (limited)
    files = file_list or [str(p.name) for p in project_dir.glob("*.py")]
    code_blob_parts = []
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
You are a Code Repair Agent. Output ONLY valid JSON.

Inputs:
- build_stderr (truncated):
{(build_stderr or '')[:4000]}

- run_stderr (truncated):
{(run_stderr or '')[:4000]}

- code_files and content:
{code_blob}

Task:
1. Propose up to 2 minimal candidate patches (full file contents) that likely fix the error.
2. Return modified_files as mapping "filename": "full new content", plus confidence and rationale.

Schema:
{{ "candidates": [ {{ "id":1, "modified_files":{{ "snippet.py": "..." }}, "diff_summary":["..."], "confidence":0.0, "rationale":["..."] }} ], "notes":"" }}
"""
    try:
        raw = run_ollama(prompt)
    except Exception as e:
        print("Code agent LLM error:", e)
        return None
    parsed = ensure_json_valid(extract_json_block(raw))
    if not isinstance(parsed, dict) or "candidates" not in parsed:
        return None
    good = []
    for c in parsed.get("candidates", [])[:2]:
        mods = c.get("modified_files", {})
        if not isinstance(mods, dict) or not mods:
            continue
        ok = True
        for path, content in mods.items():
            if not isinstance(content, str) or not safe_code(content):
                ok = False
        if not ok:
            continue
        try:
            c["confidence"] = float(c.get("confidence", 0.0))
        except Exception:
            c["confidence"] = 0.0
        good.append(c)
    parsed["candidates"] = good
    return parsed


# -------------------------
# Apply fix helpers
# -------------------------
def insert_apt_lines_in_dockerfile(dockerfile_path: Path, apt_lines: List[str]):
    df = dockerfile_path.read_text(encoding="utf-8")
    insertion = "\n".join([f"RUN {l}" for l in apt_lines]) + "\n"
    if "RUN if [ -s /app/requirements.txt ]; then pip install" in df:
        df = df.replace("RUN if [ -s /app/requirements.txt ]; then pip install --no-cache-dir -r /app/requirements.txt; fi",
                        insertion + "RUN if [ -s /app/requirements.txt ]; then pip install --no-cache-dir -r /app/requirements.txt; fi")
    else:
        # attempt to insert after WORKDIR or before CMD
        if "WORKDIR" in df:
            df = df.replace("WORKDIR /app\n", "WORKDIR /app\n" + insertion)
        else:
            df = insertion + df
    dockerfile_path.write_text(df, encoding="utf-8")


def replace_python_tag_in_dockerfile(dockerfile_path: Path, new_tag: str):
    df = dockerfile_path.read_text(encoding="utf-8")
    new_df = re.sub(r"FROM python:\d+\.\d+-slim", f"FROM python:{new_tag}-slim", df)
    if new_df == df:
        # fallback: replace any FROM python:... occurrence
        new_df = re.sub(r"FROM python:[^\s]+", f"FROM python:{new_tag}-slim", df)
    dockerfile_path.write_text(new_df, encoding="utf-8")


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
        applied_agent = a.get("applied_agent")
        if applied_agent:
            fix_counts[applied_agent] = fix_counts.get(applied_agent, 0) + 1
        applied = a.get("applied_fixes") or {}
        try:
            conf = float(applied.get("chosen_confidence", 0.0))
            confidences.append(conf)
        except Exception:
            pass
    avg_build = round(sum(build_times)/len(build_times), 2) if build_times else None
    avg_run = round(sum(run_times)/len(run_times), 2) if run_times else None
    avg_conf = round(sum(confidences)/len(confidences), 3) if confidences else 0.0

    metrics = {
        "total_attempts": total_attempts,
        "success_attempt": success_attempt,
        "attempts_to_success": success_attempt if success_attempt is not None else None,
        "total_time_sec": report.get("total_time_sec"),
        "avg_build_time_sec": avg_build,
        "avg_run_time_sec": avg_run,
        "avg_chosen_confidence": avg_conf,
        "fix_type_counts": fix_counts
    }
    return metrics


# -------------------------
# Main project processing (agentic)
# -------------------------
def process_project_agentic(project_dir: Path, out_root: Path) -> Optional[Path]:
    """
    Run full agentic build/run loop for a single project directory.
    Returns path to per-project JSON report (or None if skipped).
    """
    # verify required files
    for f in ("snippet.py", "requirements.txt", "Dockerfile"):
        if not (project_dir / f).exists():
            print(f"Skipping {project_dir.name}: missing {f}")
            return None

    out_project_dir = out_root / project_dir.name
    out_project_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "project": project_dir.name,
        "attempts": [],
        "final_status": None,
        "total_time_sec": None
    }
    t_start = time.time()

    # We'll keep a working copy of the project per attempt to apply fixes to build context
    # but we won't overwrite original project files
    for attempt in range(1, MAX_ATTEMPTS + 1):
        attempt_entry = {
            "attempt": attempt,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "image_tag": f"{project_dir.name.lower()}_trial_{attempt}_{uuid.uuid4().hex[:8]}",
            "build": {},
            "run": {},
            "applied_agent": None,
            "applied_fixes": None
        }
        print(f"[{project_dir.name}] Attempt {attempt} starting...")

        # create attempt build folder as a copy of original project dir
        build_folder = out_project_dir / f"attempt_{attempt}_context"
        if build_folder.exists():
            shutil.rmtree(build_folder)
        shutil.copytree(project_dir, build_folder)

        # build & run
        result = build_and_run_docker(build_folder, attempt_entry["image_tag"])
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

        # success?
        build_ok = result["build_returncode"] == 0
        run_ok = result["run_returncode"] == 0
        if build_ok and run_ok:
            report["final_status"] = "success"
            print(f"[{project_dir.name}] Success on attempt {attempt}")
            # cleanup image optionally
            if CLEANUP_IMAGES:
                docker_remove_image(attempt_entry["image_tag"])
            break

        # failure -> meta-agent decides
        build_stderr = result["build_stderr"] or ""
        run_stderr = result["run_stderr"] or ""
        # attempt to find a phase2 json in original project dir for extra context
        phase2_json = {}
        for p in project_dir.glob("*_phase2.json"):
            try:
                phase2_json = ensure_json_valid(p.read_text(encoding="utf-8"))
                break
            except Exception:
                phase2_json = {}
        decision = meta_agent_decide(build_stderr, run_stderr, phase2_json)
        print(f"[{project_dir.name}] Meta-Agent decision: {decision}")

        if decision == "unknown":
            report["final_status"] = "failed_unknown"
            print(f"[{project_dir.name}] Unknown failure type; stopping further attempts.")
            # cleanup image optionally
            if CLEANUP_IMAGES:
                docker_remove_image(attempt_entry["image_tag"])
            break

        attempt_entry["applied_agent"] = decision
        applied_summary = {"chosen_candidate": None, "chosen_confidence": None, "notes": None}

        # call appropriate agent, apply top candidate to build_folder
        if decision == "dependency":
            dep_out = dependency_agent_llm_project(project_dir, build_stderr, run_stderr, attempt)
            if not dep_out or not dep_out.get("candidates"):
                print(f"[{project_dir.name}] Dependency agent returned no candidates. Stopping.")
                report["final_status"] = "failed_no_fixes"
                attempt_entry["applied_fixes"] = dep_out
                if CLEANUP_IMAGES:
                    docker_remove_image(attempt_entry["image_tag"])
                break
            # choose best candidate by confidence
            candidates = sorted(dep_out["candidates"], key=lambda x: x.get("confidence", 0.0), reverse=True)
            chosen = candidates[0]
            applied_summary["chosen_candidate"] = chosen.get("id")
            applied_summary["chosen_confidence"] = chosen.get("confidence", 0.0)
            applied_summary["notes"] = chosen.get("rationale", [])
            # apply requirements to build folder
            req_path = build_folder / "requirements.txt"
            try:
                req_path.write_text("\n".join(chosen["requirements_lines"]) + "\n", encoding="utf-8")
            except Exception:
                # fallback: create requirements if missing
                req_path.parent.mkdir(parents=True, exist_ok=True)
                req_path.write_text("\n".join(chosen["requirements_lines"]) + "\n", encoding="utf-8")
            attempt_entry["applied_fixes"] = {"requirements": chosen["requirements_lines"], "rationale": chosen.get("rationale", [])}

        elif decision == "python":
            py_out = python_agent_llm_project(build_stderr, run_stderr, project_dir)
            if not py_out:
                print(f"[{project_dir.name}] Python agent returned no suggestion. Stopping.")
                report["final_status"] = "failed_no_fixes"
                attempt_entry["applied_fixes"] = py_out
                if CLEANUP_IMAGES:
                    docker_remove_image(attempt_entry["image_tag"])
                break
            applied_summary["chosen_confidence"] = py_out.get("confidence", 0.0)
            applied_summary["notes"] = py_out.get("rationale", "")
            new_min = py_out.get("recommended_python_min")
            new_max = py_out.get("recommended_python_max")
            # apply python change by editing Dockerfile in build folder
            dockerfile_path = build_folder / "Dockerfile"
            if dockerfile_path.exists() and new_min:
                replace_python_tag_in_dockerfile(dockerfile_path, new_min)
            attempt_entry["applied_fixes"] = {"python_min": new_min, "python_max": new_max, "rationale": py_out.get("rationale", "")}

        elif decision == "system":
            sys_out = system_agent_llm_project(build_stderr, run_stderr, project_dir)
            if not sys_out:
                print(f"[{project_dir.name}] System agent returned no suggestion. Stopping.")
                report["final_status"] = "failed_no_fixes"
                attempt_entry["applied_fixes"] = sys_out
                if CLEANUP_IMAGES:
                    docker_remove_image(attempt_entry["image_tag"])
                break
            applied_summary["chosen_confidence"] = sys_out.get("confidence", 0.0)
            applied_summary["notes"] = sys_out.get("rationale", "")
            apt_lines = sys_out.get("apt_lines", [])
            dockerfile_path = build_folder / "Dockerfile"
            if dockerfile_path.exists() and apt_lines:
                insert_apt_lines_in_dockerfile(dockerfile_path, apt_lines)
            attempt_entry["applied_fixes"] = {"apt_lines": apt_lines, "rationale": sys_out.get("rationale", "")}

        elif decision == "code":
            code_out = code_agent_llm_project(project_dir, build_stderr, run_stderr)
            if not code_out or not code_out.get("candidates"):
                print(f"[{project_dir.name}] Code agent returned no candidates. Stopping.")
                report["final_status"] = "failed_no_fixes"
                attempt_entry["applied_fixes"] = code_out
                if CLEANUP_IMAGES:
                    docker_remove_image(attempt_entry["image_tag"])
                break
            candidates = sorted(code_out["candidates"], key=lambda x: x.get("confidence", 0.0), reverse=True)
            chosen = candidates[0]
            applied_summary["chosen_candidate"] = chosen.get("id")
            applied_summary["chosen_confidence"] = chosen.get("confidence", 0.0)
            applied_summary["notes"] = chosen.get("rationale", [])
            # write modified files into build_folder
            modified_files = chosen.get("modified_files", {})
            for relpath, content in modified_files.items():
                dest = build_folder / relpath
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_text(content, encoding="utf-8")
            attempt_entry["applied_fixes"] = {"modified_files": list(modified_files.keys()), "diff_summary": chosen.get("diff_summary", []), "rationale": chosen.get("rationale", [])}

        else:
            # should not happen due to earlier check, but handle defensively
            report["final_status"] = "failed_unknown"
            if CLEANUP_IMAGES:
                docker_remove_image(attempt_entry["image_tag"])
            break

        # record meta summary and continue to next attempt (which will create new build folder)
        attempt_entry["applied_fixes"] = attempt_entry.get("applied_fixes") or {}
        attempt_entry["applied_fixes"]["meta"] = applied_summary
        print(f"[{project_dir.name}] Applied fixes from {decision} agent (confidence {applied_summary.get('chosen_confidence')}). Retrying...")

        # cleanup image of failed attempt optionally
        if CLEANUP_IMAGES:
            docker_remove_image(attempt_entry["image_tag"])

        # loop continues -> next attempt will copy original project again and apply new fixes in its build context

    # finalize timings and metrics
    total_time = round(time.time() - t_start, 2)
    report["total_time_sec"] = total_time
    report["metrics"] = compute_metrics_from_report(report)

    # write per-project final report
    report_path = out_project_dir / f"{project_dir.name}_final_report.json"
    with report_path.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)

    print(f"[{project_dir.name}] Finished. Final status: {report.get('final_status')}. Report saved to {report_path}")
    return report_path


# -------------------------
# Batch processing
# -------------------------
def process_dataset_agentic(root: Path, out_root: Path, csv_path: Path):
    if not root.exists():
        raise FileNotFoundError(f"Dataset root not found: {root}")

    projects = [p for p in sorted(root.iterdir()) if p.is_dir()]
    results = []

    for proj in projects:
        try:
            print(f"\n=== Processing project: {proj.name} ===")
            report_path = process_project_agentic(proj, out_root)
            if report_path is None:
                print(f"Skipped {proj.name}")
                continue
            with report_path.open("r", encoding="utf-8") as fh:
                report = json.load(fh)
            # keep only metrics in CSV
            metrics = report.get("metrics", {})
            results.append({"project": proj.name, "metrics": metrics})
        except Exception as e:
            print(f"Error processing project {proj.name}: {e}")
            # record a minimal failure metric
            results.append({"project": proj.name, "metrics": {
                "total_attempts": 0,
                "success_attempt": None,
                "attempts_to_success": None,
                "total_time_sec": None,
                "avg_build_time_sec": None,
                "avg_run_time_sec": None,
                "avg_chosen_confidence": 0.0,
                "fix_type_counts": {}
            }})

    # write CSV with project and metrics JSON string
    with csv_path.open("w", encoding="utf-8", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["project", "metrics"])
        for r in results:
            writer.writerow([r["project"], json.dumps(r["metrics"])])

    print(f"\nBatch complete. Processed {len(results)} projects. CSV saved to: {csv_path}")
    return results


# -------------------------
# CLI entry
# -------------------------
if __name__ == "__main__":
    start = time.time()
    print("Starting agentic batch run on dataset root:", ROOT_DIR)
    processed = process_dataset_agentic(ROOT_DIR, PER_PROJECT_DIR, CSV_PATH)
    print(f"Total projects processed: {len(processed)}. Time elapsed: {round(time.time() - start, 2)}s")
