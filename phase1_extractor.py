import os
import json
import subprocess
from pathlib import Path
import gc
from datetime import datetime

# -----------------------------
# GPU memory helper
# -----------------------------
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

# -----------------------------
# Read all Python source files
# -----------------------------
def read_source_files(project_path: Path) -> str:
    """Read all .py files in project recursively and combine into a single string."""
    code_texts = []
    for py_file in project_path.rglob("*.py"):
        try:
            code_texts.append(py_file.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"⚠️ Skipping {py_file}: {e}")
            continue
    return "\n\n".join(code_texts)

# -----------------------------
# Run Codellama 7B via Ollama
# -----------------------------
def run_llm_phase1(prompt_text: str, output_path: Path):
    """Send prompt to local Codellama and save output JSON."""
    clear_gpu_memory_user_safe()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        proc = subprocess.run(
            ["ollama", "run", "codellama:7b"],
            input=prompt_text,
            text=True,
            capture_output=True,
            check=True,
            encoding="utf-8"  # <<< force UTF-8 decoding
        )
        raw_output = proc.stdout.strip()

        # Optional: extract JSON substring and pretty-print
        import re
        jmatch = re.search(r"\{[\s\S]*\}$", raw_output)
        if jmatch:
            try:
                json_obj = json.loads(jmatch.group(0))
                raw_output = json.dumps(json_obj, indent=2)
            except Exception:
                pass

        output_path.write_text(raw_output, encoding="utf-8")
        print(f"✅ Phase-1 JSON saved at {output_path}")

    except subprocess.CalledProcessError as e:
        # Force UTF-8 decoding of stderr/stdout in case of errors
        err_output = (e.stderr or e.stdout or str(e))
        if isinstance(err_output, bytes):
            err_output = err_output.decode("utf-8", errors="ignore")
        print(f"❌ Ollama call failed:\n{err_output}")

    except Exception as e:
        print(f"❌ Unexpected error: {e}")


# -----------------------------
# Main execution
# -----------------------------
if __name__ == "__main__":
    project_dir = Path(r"C:\Users\sadma\Documents\Thesis\Project LLM\HG2.9K\0b92b9ca837a0f5474c732876220db78")
    phase1_out = Path("phase1_output") / f"{project_dir.name}_phase1.json"

    print(f"Scanning project: {project_dir}")
    source_code = read_source_files(project_dir)

    prompt = f"""
You are an assistant that extracts API calls from Python source code. 
Given a code snippet, output all fully-qualified APIs used 
(package.module.class.method or package.function).

Source code:
{source_code}


Task:
Identify all API calls used in the project, mapped to their parent packages/modules.
Include functions, methods, classes, attributes, etc.
Include both standard library and third-party packages.

Output in a structured format.
Output format (JSON):
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

    run_llm_phase1(prompt, phase1_out)
