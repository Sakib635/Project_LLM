import json
import subprocess
from pathlib import Path
import gc
import re

def clear_gpu_memory_user_safe():
    """Free GPU memory used by torch (if installed) and collect garbage."""
    try:
        import torch
        torch.cuda.empty_cache()
        gc.collect()
        print("✅ GPU memory cleared via torch.cuda.empty_cache()")
    except Exception:
        gc.collect()
        print("⚠️ torch not installed or GPU clear partially effective")

def run_llm_phase2(prompt_text: str, output_path: Path):
    """Send Phase-2 prompt to Codellama and save only the JSON output."""
    clear_gpu_memory_user_safe()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        proc = subprocess.run(
            ["ollama", "run", "codellama:7b"],
            input=prompt_text,
            text=True,
            capture_output=True,
            check=True,
            encoding="utf-8"
        )

        raw_output = proc.stdout.strip()

        # Extract only JSON portion
        jmatch = re.search(r"\{[\s\S]*\}$", raw_output)
        if jmatch:
            raw_output = jmatch.group(0)

        # Try to validate and pretty print
        try:
            json_obj = json.loads(raw_output)
            raw_output = json.dumps(json_obj, indent=2)
        except json.JSONDecodeError:
            print("⚠️ Warning: output not valid JSON; saved raw output")

        output_path.write_text(raw_output, encoding="utf-8")
        print(f"✅ Phase-2 JSON saved at {output_path}")

    except subprocess.CalledProcessError as e:
        err_output = (e.stderr or e.stdout or str(e))
        if isinstance(err_output, bytes):
            err_output = err_output.decode("utf-8", errors="ignore")
        print(f"❌ Ollama call failed:\n{err_output}")

    except Exception as e:
        print(f"❌ Unexpected error: {e}")

# ---------------- Example usage ----------------

phase1_json_path = Path(r"phase1_output/0a2ac74d800a2eff9540_phase1.json")
phase2_output_path = Path(r"phase2_output/project_phase2.json")

with phase1_json_path.open("r", encoding="utf-8") as f:
    phase1_data = json.load(f)

# 🧠 STRICT system prompt to enforce pure JSON output
prompt_phase2 = f"""
System Prompt:
You are an expert dependency/version and Python version inference assistant.
You must output ONLY valid JSON — no markdown, no code fences, no explanations, no comments.
Output must strictly match the JSON schema below and be human-readable formatted with 2 spaces of indentation.

Given  a structured list of extracted API calls grouped by package and standard library (Python stdlib),
infer:
1. The Python version range (min, max) based on stdlib APIs.
2. The most likely version range for each external package used based on the extracted APIs provided.
3. requirements.txt (only third-party packages' recommended lines).

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
Schema:
{{
  "python_version": {{
    "min": "3.4",
    "max": "3.7",
    "evidence": ["pathlib.Path added in Python 3.4 → min Python 3.4", "time.clock removed in Python 3.8 → max Python 3.7"],
    "notes": ""
  }},
  "dependencies": {{
    "<package>": {{
      "inferred_version_range": ">=X.Y.Z,<A.B.C" or null,
      "recommended_requirements_line": "pkg>=X.Y.Z,<A.B.C" or null,
      "evidence": [       
        "API X introduced in vM.N.P",
        "API Y removed in vU.V.W",
        "API Z deprecated in vU.V.Q"],
      "confidence": 0.0–1.0,
      "notes": ""
    }}
  }},
  "requirements.txt": [
    "only third-party packages lines here"
  ]
}}

"""

run_llm_phase2(prompt_phase2, phase2_output_path)
