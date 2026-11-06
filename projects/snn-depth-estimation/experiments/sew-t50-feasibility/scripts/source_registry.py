#!/usr/bin/env python3
import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--base-commit", required=True)
    parser.add_argument("--upstream-commit", required=True)
    parser.add_argument("--source-diff")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    project = Path(args.project_root).resolve()
    output = Path(args.output).resolve()
    files = {}
    for path in sorted(project.rglob("*")):
        if not path.is_file() or path == output:
            continue
        relative = str(path.relative_to(project))
        if "__pycache__" in path.parts or path.suffix == ".pyc" or path.name == ".DS_Store":
            continue
        files[relative] = {"bytes": path.stat().st_size, "sha256": digest(path)}

    source_diff = None
    if args.source_diff:
        patch = Path(args.source_diff).resolve()
        source_diff = {
            "path": str(patch),
            "bytes": patch.stat().st_size,
            "sha256": digest(patch),
        }
    registry = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "project_root": str(project),
        "base_commit": args.base_commit,
        "upstream_commit": args.upstream_commit,
        "source_diff": source_diff,
        "files": files,
    }
    output.write_text(json.dumps(registry, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
