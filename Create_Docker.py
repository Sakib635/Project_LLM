#!/usr/bin/env python3
r"""
prepare_docker_dataset.py

- Reads Phase-2 JSON files from:
    C:\Users\sadma\Documents\Thesis\Project LLM\phase2_output_HG2.9k
  (files named like projectName_phase1_phase2.json or projectName_phase2.json)

- For each Phase-2 JSON, extracts requirements (preferred 'requirements.txt' list;
  fallback: collect dependencies.*.recommended_requirements_line)

- Copies snippet.py from:
    C:\Users\sadma\Documents\Thesis\Project LLM\HG2.9K\<projectName>\snippet.py

- Creates folder:
    Docker_HG2.9k_Dataset\<projectName>\
  containing:
    - snippet.py
    - requirements.txt
    - Dockerfile
    - docker (duplicate of Dockerfile; per your spec)

Notes:
- The script chooses a Docker Python base image tag (3.11,3.10,3.9,3.8,3.7,3.6)
  that fits inside the inferred python_version min/max.
- If no compatible Python tag is found, script uses the latest available tag (3.11).
"""

from pathlib import Path
import json
import shutil
import re

# === CONFIGURATION ===
PHASE2_DIR = Path(r"C:/Users/sadma/Documents/Thesis/Project LLM/phase2_output_HG2.9k")
PROJECTS_ROOT = Path(r"C:/Users/sadma/Documents/Thesis/Project LLM/HG2.9K")
OUTPUT_ROOT = Path("Docker_HG2.9k_Dataset")

# Python base tags available for Docker images (prefer newer)
AVAILABLE_PY_TAGS = ["3.11", "3.10", "3.9", "3.8", "3.7", "3.6"]

# Filenames expected in project source folders
SNIPPET_FILENAME = "snippet.py"


# === UTILITIES ===
def normalize_version_str(v: str):
    """Return float representation for quick comparison (major.min),
    or None if parsing fails. Example '3.7.9' -> 3.7
    """
    if not v:
        return None
    try:
        parts = re.findall(r"\d+", str(v))
        if not parts:
            return None
        major = int(parts[0])
        minor = int(parts[1]) if len(parts) > 1 else 0
        return float(f"{major}.{minor}")
    except Exception:
        return None


def pick_python_tag(min_v, max_v):
    """Pick the highest AVAILABLE_PY_TAG that satisfies min_v <= tag <= max_v."""
    min_f = normalize_version_str(min_v) if min_v else None
    max_f = normalize_version_str(max_v) if max_v else None

    for tag in AVAILABLE_PY_TAGS:
        tag_f = normalize_version_str(tag)
        if not tag_f:
            continue
        if min_f and tag_f < min_f:
            continue
        if max_f and tag_f > max_f:
            continue
        return tag
    return AVAILABLE_PY_TAGS[0]  # fallback to newest


def extract_requirements_from_phase2_json(data):
    """Return a list of requirement lines (strings). Fallback to dependency recommended lines."""
    reqs = []
    rlist = data.get("requirements.txt")
    if isinstance(rlist, list) and rlist:
        reqs = [str(x).strip() for x in rlist if x]
    if not reqs:
        deps = data.get("dependencies", {})
        for pkg, meta in deps.items():
            if isinstance(meta, dict):
                rec = meta.get("recommended_requirements_line")
                if rec:
                    reqs.append(str(rec).strip())
    seen, deduped = set(), []
    for r in reqs:
        if r not in seen:
            seen.add(r)
            deduped.append(r)
    return deduped


def project_name_from_phase2_filename(stem: str, projects_root: Path):
    """Heuristic to find matching project folder name for a given phase2 file stem."""
    orig = stem
    s = re.sub(r"_phase1_phase2$|_phase1$|_phase2$", "", stem)

    cand = projects_root / s
    if cand.exists() and cand.is_dir():
        return s

    for d in projects_root.iterdir():
        if not d.is_dir():
            continue
        if d.name == s:
            return d.name
        if stem.startswith(d.name + "_") or stem.startswith(d.name + "-"):
            return d.name
        if d.name.startswith(s + "_") or d.name.startswith(s + "-"):
            return d.name

    cand2 = projects_root / orig
    if cand2.exists() and cand2.is_dir():
        return orig

    return None


# === MAIN ===
def main():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    phase2_files = sorted(PHASE2_DIR.glob("*.json"))
    if not phase2_files:
        print(f"⚠️ No phase-2 JSON files found in {PHASE2_DIR}")
        return

    created_count = 0
    skipped_no_reqs = 0
    missing_snippet = 0
    unresolved_name = 0

    print(f"Found {len(phase2_files)} phase-2 JSON files. Processing...")

    for p in phase2_files:
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"⚠️ Failed to parse JSON {p.name}: {e}")
            continue

        stem = p.stem
        proj_name = project_name_from_phase2_filename(stem, PROJECTS_ROOT)
        if not proj_name:
            print(f"⚠️ Could not resolve project folder for phase2 file: {p.name} (stem={stem})")
            unresolved_name += 1
            continue

        reqs = extract_requirements_from_phase2_json(data)
        if not reqs:
            print(f"⏭️ Skipping {proj_name}: no requirements inferred (empty).")
            skipped_no_reqs += 1
            continue

        candidate_snippet = PROJECTS_ROOT / proj_name / SNIPPET_FILENAME
        if not candidate_snippet.exists():
            print(f"⚠️ Missing snippet.py for project {proj_name} at expected path: {candidate_snippet}")
            missing_snippet += 1
            continue

        py_min = data.get("python_version", {}).get("min")
        py_max = data.get("python_version", {}).get("max")
        chosen_tag = pick_python_tag(py_min, py_max)

        out_proj_dir = OUTPUT_ROOT / proj_name
        out_proj_dir.mkdir(parents=True, exist_ok=True)

        req_path = out_proj_dir / "requirements.txt"
        req_path.write_text("\n".join(reqs).strip() + "\n", encoding="utf-8")

        dest_snip = out_proj_dir / "snippet.py"
        shutil.copy2(candidate_snippet, dest_snip)

        dockerfile_content = f"""# Auto-generated Dockerfile for project: {proj_name}
FROM python:{chosen_tag}-slim

WORKDIR /app

COPY snippet.py /app/snippet.py
COPY requirements.txt /app/requirements.txt

# Optional system packages:
# RUN apt-get update && apt-get install -y --no-install-recommends build-essential gcc

RUN if [ -s /app/requirements.txt ]; then pip install --no-cache-dir -r /app/requirements.txt; fi

CMD ["python", "snippet.py"]
"""

        (out_proj_dir / "Dockerfile").write_text(dockerfile_content, encoding="utf-8")
        (out_proj_dir / "docker").write_text(dockerfile_content, encoding="utf-8")

        print(f"✅ Created: {out_proj_dir}  (Python {chosen_tag})  packages:{len(reqs)}")
        created_count += 1

    print("\n=== Summary ===")
    print(f"✅ Projects processed: {created_count}")
    print(f"⏭️ Skipped (no requirements): {skipped_no_reqs}")
    print(f"⚠️ Missing snippet.py: {missing_snippet}")
    print(f"❓ Unresolved name matches: {unresolved_name}")
    print("📁 Output directory:", OUTPUT_ROOT.resolve())


if __name__ == "__main__":
    main()
