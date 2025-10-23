import os
import json
import subprocess
from pathlib import Path
import gc
from datetime import datetime
import re
from tqdm import tqdm

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
            encoding="utf-8"
        )
        raw_output = proc.stdout.strip()

        # 🔒 Strictly extract only JSON content
        jmatch = re.search(r"\{[\s\S]*\}$", raw_output)
        if not jmatch:
            raise ValueError("❌ No valid JSON structure found in LLM output.")
        
        json_str = jmatch.group(0)
        json_obj = json.loads(json_str)  # validate JSON

        # Reformat & save clean JSON only
        clean_json = json.dumps(json_obj, indent=2)
        output_path.write_text(clean_json, encoding="utf-8")
        tqdm.write(f"✅ Clean JSON saved: {output_path.name}")

    except subprocess.CalledProcessError as e:
        err_output = (e.stderr or e.stdout or str(e))
        if isinstance(err_output, bytes):
            err_output = err_output.decode("utf-8", errors="ignore")
        tqdm.write(f"❌ Ollama call failed:\n{err_output}")

    except Exception as e:
        tqdm.write(f"❌ Unexpected error while processing {output_path.name}: {e}")

# -----------------------------
# Main execution (resume-safe)
# -----------------------------
if __name__ == "__main__":
    projects_root = Path(r"C:/Users/sadma/Documents/Thesis/Project LLM/HG2.9K")
    output_root = Path("phase1_output_HG2.9k")
    output_root.mkdir(exist_ok=True)

    project_dirs = [p for p in projects_root.iterdir() if p.is_dir()]
    completed_projects = {f.stem.replace("_phase1", "") for f in output_root.glob("*_phase1.json")}

    tqdm.write(f"🔍 Found {len(project_dirs)} total projects.")
    tqdm.write(f"✅ {len(completed_projects)} already processed. {len(project_dirs) - len(completed_projects)} remaining.\n")

    remaining_projects = [p for p in project_dirs if p.name not in completed_projects]

    for project_dir in tqdm(remaining_projects, desc="Processing projects", unit="proj", initial=len(completed_projects), total=len(project_dirs)):
        phase1_out = output_root / f"{project_dir.name}_phase1.json"

        if phase1_out.exists():
            tqdm.write(f"⏭️ Skipping {project_dir.name} (already processed)")
            continue

        tqdm.write(f"\n📂 Scanning project: {project_dir.name}")
        source_code = read_source_files(project_dir)

        if not source_code.strip():
            tqdm.write(f"⚠️ No Python files found in {project_dir}")
            continue

        # 🧠 Prompt kept *unchanged* — but ensures LLM returns JSON only
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

Output strictly as valid JSON only.
Do not include any explanation or commentary.

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
        run_llm_phase1(prompt, phase1_out)

    print("\n🎯 All projects processed successfully (strict JSON mode).")
