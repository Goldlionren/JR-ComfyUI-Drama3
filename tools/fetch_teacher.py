"""Install the fixed Fish Audio checkpoint. Inference never calls this utility."""
from argparse import ArgumentParser
import json
import os
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--dest", type=Path, default=root / ".runtime" / "teacher")
    args = parser.parse_args()
    target = args.dest.resolve()
    manifest = json.loads((root / "resources" / "teacher.json").read_text("utf-8"))
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("HF_HOME", str(root / ".runtime" / "hf-cache"))
    from huggingface_hub import snapshot_download

    snapshot_download(repo_id=manifest["repository"], revision=manifest["revision"],
                      allow_patterns=list(manifest["files"]), local_dir=target,
                      max_workers=4, token=False)
    import sys
    sys.path.insert(0, str(root))
    from jr_teacher import validate_teacher
    validate_teacher(target)
    print(f"Verified official teacher {manifest['revision']} at {target}", flush=True)


if __name__ == "__main__":
    main()
