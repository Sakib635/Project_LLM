#!/usr/bin/env python3
"""
=== FULL PIPELINE: Phase 1 → Phase 2 → Docker Build & Execution Test ===

Performs:
  1. API extraction via Codellama (Phase 1)
  2. Dependency inference (Phase 2)
  3. Dockerfile + requirements generation
  4. Docker build & run execution
  5. JSON summary with timings and outputs

Author: Sadman Jashim Sakib
"""

import os
import re
import gc
import json
import time
import shutil
import subprocess
from pathlib import Path


# ===============================================================
#                   UTILITY FUNCTIONS
# ===============================================================

def clear_gpu_memory_user_safe():
    """Free GPU memory used by torch (if available)."""
    try:
        import torch
        torch.cuda.empty_cache()
        gc.collect()
        print("✅ GPU memory cleared via torch.cuda.empty_cache()")
    except Exception:
        gc.collect()
        print("⚠️ torch not installed or GPU clear partially effective")


def run_ollama(prompt_text: str, model: str = "codellama:7b") -> str:
    """Run Ollama model locally and return stdout."""
    clear_gpu_memory_user_safe()
    proc = subprocess.run(
        ["ollama", "run", model],
        input=prompt_text,
        text=True,
        capture_output=True,
        encoding="utf-8",
        timeout=600
    )
    return proc.stdout.strip()


def read_source_files(project_path: Path) -> str:
    """Concatenate all Python files in the project."""
    all_code = []
    for py in project_path.rglob("*.py"):
        try:
            all_code.append(py.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"⚠️ Could not read {py}: {e}")
    return "\n\n".join(all_code)


def extract_json_block(text: str) -> str:
    """Extract JSON-like portion from LLM response."""
    m = re.search(r"\{[\s\S]*\}$", text)
    return m.group(0) if m else text


def ensure_json_valid(raw_text: str) -> dict:
    """Try to load valid JSON; fallback to {'raw_text': ...}."""
    try:
        return json.loads(raw_text)
    except Exception:
        # Try to sanitize common issues
        cleaned = re.sub(r"```(?:json)?|```", "", raw_text).strip()
        try:
            return json.loads(cleaned)
        except Exception:
            return {"raw_text": raw_text.strip(), "error": "Invalid JSON structure"}


# ===============================================================
#                   PHASE 1 – API Extraction
# ===============================================================

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
    json.dump(data, output_path.open("w", encoding="utf-8"), indent=2)
    print(f"✅ Phase 1 complete → {output_path}")
    return output_path


# ===============================================================
#                   PHASE 2 – Dependency Inference
# ===============================================================

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

    json.dump(data, output_path.open("w", encoding="utf-8"), indent=2)
    print(f"✅ Phase 2 complete → {output_path}")
    return output_path


# ===============================================================
#                   DOCKER GENERATION
# ===============================================================

def pick_python_tag(min_v, max_v):
    tags = ["3.11", "3.10", "3.9", "3.8"]
    def num(v):
        if not v: return None
        m = re.findall(r"\d+", v)
        return float(f"{m[0]}.{m[1]}") if len(m) >= 2 else float(m[0])
    minf = num(min_v)
    maxf = num(max_v)
    for t in tags:
        tf = float(t)
        if (not minf or tf >= minf) and (not maxf or tf <= maxf):
            return t
    return "3.11"


def generate_docker(project_dir: Path, phase2_json: Path, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    data = ensure_json_valid(phase2_json.read_text(encoding="utf-8"))

    reqs = data.get("requirements.txt", [])
    if not reqs:
        deps = data.get("dependencies", {})
        for k, v in deps.items():
            if isinstance(v, dict) and v.get("recommended_requirements_line"):
                reqs.append(v["recommended_requirements_line"])

    py_info = data.get("python_version", {})
    py_tag = pick_python_tag(py_info.get("min"), py_info.get("max"))

    (out_dir / "requirements.txt").write_text("\n".join(reqs) or "", encoding="utf-8")
    snippet = project_dir / "snippet.py"
    if snippet.exists():
        shutil.copy2(snippet, out_dir / "snippet.py")

    dockerfile = f"""# Auto-generated Dockerfile
FROM python:{py_tag}-slim
WORKDIR /app
COPY . /app
RUN pip install --no-cache-dir -r requirements.txt || true
CMD ["python", "snippet.py"]
"""
    (out_dir / "Dockerfile").write_text(dockerfile, encoding="utf-8")
    print(f"✅ Dockerfile created (Python {py_tag}) at {out_dir}")
    return out_dir


# ===============================================================
#                   EXECUTION TEST
# ===============================================================

def run_docker_execution(docker_dir: Path, project_name: str, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / f"{project_name}_execution_report.json"

    image_tag = f"single_{project_name.lower()}"
    result = {
        "project": project_name,
        "status": None,
        "build_time_sec": None,
        "run_time_sec": None,
        "output": "",
        "error": ""
    }

    try:
        start_build = time.time()
        build = subprocess.run(["docker", "build", "-t", image_tag, str(docker_dir)],
                               capture_output=True, text=True, timeout=600)
        result["build_time_sec"] = round(time.time() - start_build, 2)

        if build.returncode != 0:
            result["status"] = "build_failed"
            result["error"] = build.stderr[-400:]
        else:
            start_run = time.time()
            run = subprocess.run(["docker", "run", "--rm", image_tag],
                                 capture_output=True, text=True, timeout=120)
            result["run_time_sec"] = round(time.time() - start_run, 2)
            result["output"] = run.stdout[-400:]
            result["error"] = run.stderr[-400:]
            result["status"] = "success" if run.returncode == 0 else "runtime_error"

    except subprocess.TimeoutExpired:
        result["status"] = "timeout"
        result["error"] = "Execution timed out"
    except Exception as e:
        result["status"] = "exception"
        result["error"] = str(e)

    json.dump(result, report_path.open("w", encoding="utf-8"), indent=2)
    print(f"📄 Execution report saved: {report_path}")
    return report_path


# ===============================================================
#                   MAIN PIPELINE
# ===============================================================

if __name__ == "__main__":
    project_dir = Path(r"C:\Users\sadma\Documents\Thesis\Project LLM\Example_Project")
    base_out = Path(r"C:\Users\sadma\Documents\Thesis\Project LLM\Example_Run_Merged")

    phase1_dir = base_out / "phase1_output"
    phase2_dir = base_out / "phase2_output"
    docker_dir = base_out / "docker_build"
    exec_dir = base_out / "execution_output"

    print(f"\n🚀 Starting full pipeline for project: {project_dir.name}\n")
    t0 = time.time()

    phase1_path = phase1_extract_apis(project_dir, phase1_dir)
    phase2_path = phase2_infer_dependencies(phase1_path, phase2_dir)
    docker_project_dir = generate_docker(project_dir, phase2_path, docker_dir)
    exec_report = run_docker_execution(docker_project_dir, project_dir.name, exec_dir)

    total_time = round(time.time() - t0, 2)
    print("\n✅ Pipeline finished successfully!")
    print(f"Total time: {total_time}s")
    print(f"Phase 1 JSON: {phase1_path}")
    print(f"Phase 2 JSON: {phase2_path}")
    print(f"Docker folder: {docker_project_dir}")
    print(f"Execution report: {exec_report}")
