"""Build the two deliverable ZIPs for the UROP repo corrections.

Run with the project's .venv python. Writes both zips under a
`review_bundles/` directory at the repo root (git-ignored path chosen at
call time, not committed).
"""
import fnmatch
import hashlib
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = REPO_ROOT / "review_bundles"
OUT_DIR.mkdir(exist_ok=True)

EXCLUDE_PATTERNS = ["__pycache__", "*.pyc", ".pytest_cache", "*.egg-info"]


def should_skip(path: Path) -> bool:
    parts = path.parts
    for pat in EXCLUDE_PATTERNS:
        if any(fnmatch.fnmatch(p, pat) for p in parts) or fnmatch.fnmatch(path.name, pat):
            return True
    return False


def add_tree(zf: zipfile.ZipFile, src_dir: Path, arc_prefix: str):
    if not src_dir.exists():
        print(f"  WARN missing dir, skipped: {src_dir}")
        return
    n = 0
    for p in sorted(src_dir.rglob("*")):
        if p.is_dir():
            continue
        if should_skip(p):
            continue
        rel = p.relative_to(src_dir)
        arcname = f"{arc_prefix}/{rel.as_posix()}" if arc_prefix else rel.as_posix()
        zf.write(p, arcname)
        n += 1
    print(f"  added {n} files from {src_dir} -> {arc_prefix or '.'}")


def add_file(zf: zipfile.ZipFile, src_file: Path, arcname: str):
    if not src_file.exists():
        print(f"  WARN missing file, skipped: {src_file}")
        return
    zf.write(src_file, arcname)
    print(f"  added file {src_file} -> {arcname}")


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_review_zip():
    out_path = OUT_DIR / "urop_review_bundle.zip"
    print(f"\n=== Building review ZIP: {out_path} ===")
    with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for d in ["src", "scripts", "notebooks", "configs", "tests", "requirements", "docs"]:
            add_tree(zf, REPO_ROOT / d, d)
        for f in ["pyproject.toml", "README.md", "CLAUDE.md", ".gitignore", ".env.example"]:
            add_file(zf, REPO_ROOT / f, f)
        # Provenance text (no vendored code, no raw datasets) + small manifests/report only
        add_file(zf, REPO_ROOT / "external" / "PROVENANCE.md", "external/PROVENANCE.md")
        add_file(zf, REPO_ROOT / "data" / "tau_bench_raw" / "PROVENANCE.md", "data/tau_bench_raw/PROVENANCE.md")
        for f in ["conversion_report.json", "smoke_sample_manifest.json",
                  "pilot_sample_manifest.json", "split_manifest.json"]:
            add_file(zf, REPO_ROOT / "data" / "tau_bench" / f, f"data/tau_bench/{f}")
    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"Review ZIP: {out_path} ({size_mb:.2f} MB), sha256={sha256_of(out_path)}")
    return out_path


def build_kaggle_bundle():
    out_path = OUT_DIR / "urop_kaggle_private_bundle.zip"
    print(f"\n=== Building private Kaggle bundle: {out_path} ===")
    with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as zf:
        # Our corrected project code
        add_tree(zf, REPO_ROOT / "src", "src")
        add_tree(zf, REPO_ROOT / "requirements", "requirements")
        # Vendored AutoTraceGT (no confirmed license -- private bundle only, never public repo)
        add_tree(zf, REPO_ROOT / "external" / "autotracegt_upstream", "external/autotracegt_upstream")
        # Only the specific tau-bench subtree the notebook's own path check and
        # extract_retail_tool_schemas() actually read (ast-parsed, not imported).
        add_tree(
            zf,
            REPO_ROOT / "external" / "tau-bench" / "tau_bench" / "envs" / "retail" / "tools",
            "external/tau-bench/tau_bench/envs/retail/tools",
        )
        # Real smoke sample + its manifest (exactly what Section 5/7's REQUIRED_PATHS checks)
        add_file(zf, REPO_ROOT / "data" / "tau_bench" / "tau_bench_train_3.jsonl",
                  "data/tau_bench/tau_bench_train_3.jsonl")
        add_file(zf, REPO_ROOT / "data" / "tau_bench" / "smoke_sample_manifest.json",
                  "data/tau_bench/smoke_sample_manifest.json")
        add_file(zf, REPO_ROOT / "data" / "tau_bench" / "conversion_report.json",
                  "data/tau_bench/conversion_report.json")
        # Provenance
        add_file(zf, REPO_ROOT / "external" / "PROVENANCE.md", "external/PROVENANCE.md")
        add_file(zf, REPO_ROOT / "data" / "tau_bench_raw" / "PROVENANCE.md",
                  "data/tau_bench_raw/PROVENANCE.md")
    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"Kaggle bundle: {out_path} ({size_mb:.2f} MB), sha256={sha256_of(out_path)}")
    return out_path


if __name__ == "__main__":
    r = build_review_zip()
    k = build_kaggle_bundle()
    print("\nDone.")
    print(f"Review ZIP:  {r}")
    print(f"Kaggle ZIP:  {k}")
