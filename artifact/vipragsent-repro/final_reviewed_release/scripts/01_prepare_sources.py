from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from vipragsent.data.ingest_public import download_public_datasets, public_download_plan
from vipragsent.data.ingest_visobert import build_unified_records, discover_source_files, scan_local_export
from vipragsent.utils.env import env_bool, load_env_file
from vipragsent.utils.hashing import sha256_file
from vipragsent.utils.io import dump_json, ensure_dir, write_jsonl


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare local/public data sources without training.")
    parser.add_argument("--data-root", default=str(ROOT / "data"))
    parser.add_argument("--local-visobert-export", default=None)
    parser.add_argument("--dry-run", action="store_true", help="Scan and report without writing interim data.")
    parser.add_argument("--download-public", action="store_true", help="Download public dataset mirrors; license review is still required.")
    parser.add_argument("--skip-aivivn", action="store_true", help="Skip Kaggle AIVIVN download.")
    parser.add_argument("--limit", type=int, default=None, help="Optional ingest record limit.")
    args = parser.parse_args()

    load_env_file(ROOT / ".env")
    data_root = Path(args.data_root)
    local_root = Path(args.local_visobert_export or os.getenv("LOCAL_VISOBERT_EXPORT", r"D:\hf_cache\exports\visobert_12000_api"))
    manifest_dir = ensure_dir(data_root / "manifest")

    scan = scan_local_export(local_root)
    files = discover_source_files(local_root)
    checksums = [{"path": str(path), "checksum": sha256_file(path), "bytes": path.stat().st_size} for path in files]
    network_allowed = env_bool("ENABLE_NETWORK_DOWNLOAD", True)
    public_plan = public_download_plan(enable_network=args.download_public and network_allowed)

    report = {
        "status": "dry_run" if args.dry_run else "prepared",
        "local_visobert": scan,
        "checksums_count": len(checksums),
        "public_fallbacks": public_plan,
    }
    dump_json(manifest_dir / "checksums.json", {"schema_version": "1.0", "files": checksums})
    dump_json(manifest_dir / "source_registry.json", report)

    if not args.dry_run and local_root.exists():
        output_path = data_root / "interim" / "cleaned_social.jsonl"
        count = write_jsonl(output_path, build_unified_records(local_root, limit=args.limit))
        report["interim_output"] = str(output_path)
        report["records_written"] = count

    if args.download_public and not args.dry_run:
        if not network_allowed:
            report["public_download"] = {"status": "skipped_network_disabled"}
        else:
            public_output = download_public_datasets(
                data_root / "raw",
                include_aivivn=not args.skip_aivivn,
                token=os.getenv("HF_TOKEN") or None,
            )
            public_path = data_root / "interim" / "public_datasets_unified.jsonl"
            public_count = write_jsonl(public_path, public_output["records"])
            report["public_download"] = {
                "status": public_output["status"],
                "datasets": public_output["datasets"],
                "unified_output": str(public_path),
                "records_written": public_count,
            }

    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
