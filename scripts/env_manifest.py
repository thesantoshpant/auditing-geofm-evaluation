"""Record the environment used for the headline grid.

Produces data/results/environment.json with installed versions of the
reproducibility-critical packages plus the AnySat torch.hub cache state
(branch or sha if available, and the approximate cache-fetch timestamp
derived from cache-directory mtime -- not an authoritative fetch time,
just the best available proxy when torch.hub strips git metadata).

If a package is not installed, its entry is null (not a failure -- the
environment may not have every model extra installed).
"""
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "data" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def pkg_version(name):
    try:
        from importlib.metadata import version, PackageNotFoundError
        try:
            return version(name)
        except PackageNotFoundError:
            return None
    except ImportError:
        return None


def anysat_cache_state():
    """Return (branch_or_sha, fetch_timestamp_iso) from the torch.hub cache.
    torch.hub strips git metadata when it unzips a github archive, so when the
    user fetched 'gastruc/anysat' with default branch, only the branch name is
    recoverable, not the resolved commit sha. We also return the cache mtime
    so a reproducer can look up which commit was HEAD at that time on GitHub."""
    import datetime
    cache = Path.home() / ".cache" / "torch" / "hub"
    if not cache.exists():
        return (None, None)
    candidates = [p for p in cache.glob("gastruc_anysat_*") if p.is_dir()]
    if not candidates:
        return (None, None)
    parts = candidates[0].name.split("_")
    branch_or_sha = parts[-1] if len(parts) >= 3 else None
    try:
        mtime = candidates[0].stat().st_mtime
        ts = datetime.datetime.fromtimestamp(mtime, tz=datetime.timezone.utc).isoformat()
    except OSError:
        ts = None
    return (branch_or_sha, ts)


def main():
    pkgs = [
        "torch", "torchvision", "numpy", "pandas", "scipy", "scikit-learn",
        "rasterio", "shapely", "geopandas", "pyproj",
        "terratorch", "timm", "claymodel",
        "huggingface-hub", "matplotlib",
    ]
    branch, fetch_ts = anysat_cache_state()
    manifest = {
        "python": sys.version.split()[0],
        "platform": sys.platform,
        "packages": {p: pkg_version(p) for p in pkgs},
        "anysat_torch_hub": {
            "branch_or_sha": branch,
            "cache_mtime_utc_approx": fetch_ts,
            "exact_commit_recoverable": False if branch == "main" else True,
            "recovery_note": ("torch.hub strips git metadata when it unzips a "
                              "GitHub archive. If branch_or_sha is 'main', the "
                              "exact commit at fetch time is not recoverable "
                              "from the cache. cache_mtime_utc_approx is the "
                              "cache directory's mtime, an approximate proxy "
                              "for when AnySat was fetched; a reviewer can look "
                              "up gastruc/anysat HEAD on GitHub at roughly that "
                              "timestamp to bound the commit."),
        },
        "note": ("Captured at run time on the compute node used for the headline grid. "
                 "The 'packages' dict above is the source of truth for "
                 "reproducibility; pyproject.toml specifies looser ranges. "
                 "claymodel is pinned in pyproject.toml to commit "
                 "f14e698f3c237cabf8d28dec669a362d66625381 and was imported "
                 "from that commit."),
    }
    out = RESULTS_DIR / "environment.json"
    out.write_text(json.dumps(manifest, indent=2))
    print(f"Wrote data/results/{out.name}")
    for k, v in manifest["packages"].items():
        print(f"  {k:<22} {v}")
    print(f"  anysat (torch.hub)    {manifest['anysat_torch_hub']['branch_or_sha']}  "
          f"(cached {manifest['anysat_torch_hub']['cache_mtime_utc_approx']})")


if __name__ == "__main__":
    main()
