import json
import subprocess
from pathlib import Path
import gc
import re
import time
import csv

# ========== MEMORY CLEANUP ==========
def clear_gpu_memory_user_safe():
    """Free GPU memory used by torch (if available) and collect garbage."""
    try:
        import torch
        torch.cuda.empty_cache()
        gc.collect()
        print("✅ GPU memory cleared via torch.cuda.empty_cache()")
    except Exception:
        gc.collect()
        print("⚠️ torch not installed or GPU clear partially effective")


# ========== LLM CALL ==========
def run_llm_phase2(prompt_text: str, output_path: Path):
    """Send Phase-2 prompt to Codellama and save only the JSON output. Returns inference time (seconds)."""
    clear_gpu_memory_user_safe()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    start_time = time.time()

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
        duration = round(time.time() - start_time, 2)

        # Extract only JSON portion
        jmatch = re.search(r"\{[\s\S]*\}$", raw_output)
        if jmatch:
            raw_output = jmatch.group(0)

        # Validate and pretty print JSON
        try:
            json_obj = json.loads(raw_output)
            raw_output = json.dumps(json_obj, indent=2)
        except json.JSONDecodeError:
            print("⚠️ Output not valid JSON; saving raw output")

        output_path.write_text(raw_output, encoding="utf-8")
        print(f"✅ Phase-2 JSON saved at {output_path}")
        return duration

    except subprocess.CalledProcessError as e:
        err_output = (e.stderr or e.stdout or str(e))
        if isinstance(err_output, bytes):
            err_output = err_output.decode("utf-8", errors="ignore")
        print(f"❌ Ollama call failed:\n{err_output}")

    except Exception as e:
        print(f"❌ Unexpected error: {e}")

    return None


# ========== AUTOMATION ==========
def process_all_phase1_files(
    phase1_dir: Path,
    phase2_dir: Path,
    csv_path: Path,
    resume: bool = True,
    sleep_between: float = 2.0
):
    """
    Iterate over all Phase-1 JSON files and run Phase-2 inference.
    Automatically resumes where it left off if resume=True.
    Logs summary data to CSV.
    """

    all_files = sorted(phase1_dir.glob("*.json"))
    print(f"🔍 Found {len(all_files)} Phase-1 files.")

    # Prepare CSV file header if not exist
    if not csv_path.exists():
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Project Name", "Number of Packages", "Number of APIs", "Time Taken (seconds)"])

    for i, file_path in enumerate(all_files, start=1):
        rel_name = file_path.stem
        out_path = phase2_dir / f"{rel_name}_phase2.json"

        # Resume: skip if already processed
        if resume and out_path.exists():
            print(f"⏭️ Skipping {rel_name} (already processed)")
            continue

        try:
            with file_path.open("r", encoding="utf-8") as f:
                phase1_data = json.load(f)

            # Count packages & APIs
            num_packages = len(phase1_data)
            num_apis = sum(len(v) for v in phase1_data.values() if isinstance(v, list))

            # Build LLM prompt
            prompt_phase2 = f"""
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
            print(f"\n🚀 [{i}/{len(all_files)}] Running Phase-2 for: {rel_name}")
            duration = run_llm_phase2(prompt_phase2, out_path)

            # Write summary row
            if duration is not None:
                with csv_path.open("a", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow([rel_name, num_packages, num_apis, duration])

            print("🕓 Waiting a bit before next file...\n")
            time.sleep(sleep_between)

        except Exception as e:
            print(f"❌ Error processing {file_path}: {e}")


# ========== MAIN ==========
if __name__ == "__main__":
    phase1_dir = Path(r"C:\Users\sadma\Documents\Thesis\Project LLM\phase1_output_HG2.9k")
    phase2_dir = Path(r"C:\Users\sadma\Documents\Thesis\Project LLM\phase2_output_HG2.9k")
    csv_path = Path("phase2_summary.csv")

    process_all_phase1_files(phase1_dir, phase2_dir, csv_path)
