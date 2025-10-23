#!/usr/bin/env python3
"""
Execution Test Runner for Dockerized Python Gist Projects

- Scans all project folders containing snippet.py, requirements.txt, and Dockerfile
  under ROOT.
- Builds a Docker image for each project.
- Runs the container and captures output.
- Stores incremental results in RESULTS_DIR in JSON and CSV formats.
- Supports resuming: skips projects already recorded in summary files.
- Tracks build and run duration for each project.
"""

import subprocess
from pathlib import Path
import json
import csv
import time

# === CONFIGURATION ===
ROOT = Path(r"C:\Users\sadma\Documents\Thesis\Project LLM\Docker_HG2.9k_Dataset")
RESULTS_DIR = Path(r"C:\Users\sadma\Documents\Thesis\Project LLM\Docker_HG2.9k_Execution_Output")

JSON_SUMMARY = RESULTS_DIR / "docker_execution_summary.json"
CSV_SUMMARY = RESULTS_DIR / "docker_execution_summary.csv"

# === SETUP RESULTS FOLDER ===
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# === LOAD EXISTING RESULTS FOR RESUME CAPABILITY ===
summary = []
processed_projects = set()

if JSON_SUMMARY.exists():
    try:
        with open(JSON_SUMMARY, "r", encoding="utf-8") as jf:
            summary = json.load(jf)
            processed_projects = {entry["project"] for entry in summary}
            print(f"Resuming execution: {len(processed_projects)} projects already recorded.")
    except Exception as e:
        print(f"⚠️ Failed to read existing JSON summary: {e}")
        summary = []
        processed_projects = set()

# === SCAN PROJECTS ===
projects = [p for p in ROOT.iterdir() if p.is_dir()]
total = len(projects)

print(f"Found {total} projects. Starting Docker execution tests...\n")

for idx, proj in enumerate(projects, start=1):
    name = proj.name

    if name in processed_projects:
        print(f"[{idx}/{total}] ⏭ Skipping {name} (already processed)")
        continue

    dockerfile = proj / "Dockerfile"
    snippet = proj / "snippet.py"

    print(f"[{idx}/{total}] 🧱 Processing: {name}")

    if not dockerfile.exists() or not snippet.exists():
        print(f"⚠️ Skipping {name}: Missing Dockerfile or snippet.py")
        entry = {
            "project": name,
            "status": "missing_files",
            "output": "",
            "error": "Missing Dockerfile or snippet.py",
            "build_time_sec": None,
            "run_time_sec": None
        }
        summary.append(entry)
        processed_projects.add(name)
        continue

    build_time = None
    run_time = None

    try:
        image_tag = f"gisttest_{name.lower()}"

        # === Build Docker image ===
        build_cmd = ["docker", "build", "-t", image_tag, str(proj)]
        print(f"   🔨 Building Docker image...")
        start_build = time.time()
        build_result = subprocess.run(build_cmd, capture_output=True, text=True, timeout=600)
        build_time = round(time.time() - start_build, 2)

        if build_result.returncode != 0:
            print(f"❌ Build failed for {name} ({build_time}s)")
            entry = {
                "project": name,
                "status": "build_failed",
                "output": build_result.stdout[-400:].strip(),
                "error": build_result.stderr[-400:].strip(),
                "build_time_sec": build_time,
                "run_time_sec": None
            }
            summary.append(entry)
            processed_projects.add(name)
            continue

        # === Run Docker container ===
        run_cmd = ["docker", "run", "--rm", image_tag]
        print(f"   ▶ Running container...")
        start_run = time.time()
        run_result = subprocess.run(run_cmd, capture_output=True, text=True, timeout=120)
        run_time = round(time.time() - start_run, 2)

        if run_result.returncode == 0:
            print(f"✅ Success for {name} (build: {build_time}s, run: {run_time}s)")
            entry = {
                "project": name,
                "status": "success",
                "output": run_result.stdout[:400].strip(),
                "error": run_result.stderr[:400].strip(),
                "build_time_sec": build_time,
                "run_time_sec": run_time
            }
        else:
            print(f"⚠️ Runtime error for {name} (build: {build_time}s, run: {run_time}s)")
            entry = {
                "project": name,
                "status": "runtime_error",
                "output": run_result.stdout[:400].strip(),
                "error": run_result.stderr[-400:].strip(),
                "build_time_sec": build_time,
                "run_time_sec": run_time
            }

        summary.append(entry)
        processed_projects.add(name)

    except subprocess.TimeoutExpired:
        print(f"⏱️ Timeout for {name}")
        entry = {
            "project": name,
            "status": "timeout",
            "output": "",
            "error": "Execution exceeded timeout limit",
            "build_time_sec": build_time,
            "run_time_sec": run_time
        }
        summary.append(entry)
        processed_projects.add(name)

    except Exception as e:
        print(f"💥 Exception for {name}: {e}")
        entry = {
            "project": name,
            "status": "exception",
            "output": "",
            "error": str(e),
            "build_time_sec": build_time,
            "run_time_sec": run_time
        }
        summary.append(entry)
        processed_projects.add(name)

    # === SAVE RESULTS AFTER EACH PROJECT ===
    try:
        with open(JSON_SUMMARY, "w", encoding="utf-8") as jf:
            json.dump(summary, jf, indent=2)
        with open(CSV_SUMMARY, "w", newline="", encoding="utf-8") as cf:
            writer = csv.DictWriter(cf, fieldnames=["project", "status", "output", "error", "build_time_sec", "run_time_sec"])
            writer.writeheader()
            writer.writerows(summary)
    except Exception as e:
        print(f"⚠️ Failed to save summary files: {e}")

# === FINAL SUMMARY ===
print("\n=== 🧾 Execution Test Completed ===")
print(f"Total projects scanned: {total}")
print(f"Results saved in: {RESULTS_DIR}")
print(f"JSON summary: {JSON_SUMMARY}")
print(f"CSV summary:  {CSV_SUMMARY}")
