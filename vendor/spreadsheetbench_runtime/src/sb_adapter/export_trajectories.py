from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from .log_parser import parse_agent_logs, save_records


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _validated_evaluation(path: Path, selected_ids: list[str]) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"evaluation file not found: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid evaluation JSON: {path}: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("format") != "spreadsheetbench_adapter_evaluation_v1":
        raise ValueError("evaluation JSON has the wrong or missing format identity")
    summary = payload.get("summary")
    if not isinstance(summary, dict):
        raise ValueError("evaluation JSON must contain a summary object")
    if summary.get("evaluation_mode") != "libreoffice_recalc":
        raise ValueError("evaluation was not produced with LibreOffice recalculation")
    manifest = summary.get("run_manifest")
    if not isinstance(manifest, dict) or not manifest.get("protocol_sha256"):
        raise ValueError("evaluation is not bound to a run manifest")
    checks = manifest.get("checks")
    if not isinstance(checks, dict) or not checks or not all(checks.values()):
        raise ValueError("evaluation run-manifest checks are missing or failed")
    rows = payload.get("results")
    if not isinstance(rows, list):
        raise ValueError("evaluation JSON must contain a results list")
    ids = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or row.get("id") is None:
            raise ValueError(f"evaluation row {index} lacks an id")
        if not isinstance(row.get("hard_score"), (int, float)) or not isinstance(
            row.get("soft_score"), (int, float)
        ):
            raise ValueError(f"evaluation row {index} lacks explicit verifier scores")
        test_cases = row.get("test_cases")
        if not isinstance(test_cases, list) or any(
            not isinstance(case, dict) or not isinstance(case.get("passed"), bool)
            for case in test_cases
        ):
            raise ValueError(f"evaluation row {index} lacks explicit passed fields")
        ids.append(str(row["id"]))
    if len(ids) != len(set(ids)):
        raise ValueError("evaluation contains duplicate task IDs")
    missing = [task_id for task_id in selected_ids if task_id not in set(ids)]
    extra = sorted(set(ids).difference(selected_ids))
    if missing or extra:
        raise ValueError(
            f"evaluation slice mismatch: missing={missing[:8]}, extra={extra[:8]}"
        )
    if summary.get("total_instances") != len(selected_ids):
        raise ValueError("evaluation summary denominator does not match the selected slice")
    return {
        "format": payload["format"],
        "run_manifest": manifest,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export benchmark logs and verifier results as ordered trajectories."
    )
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--eval-file", type=Path, required=True)
    parser.add_argument("--run-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start-idx", type=int, required=True)
    parser.add_argument("--end-idx", type=int, required=True)
    args = parser.parse_args()

    dataset_file = args.data_path / "dataset.json"
    dataset = json.loads(dataset_file.read_text(encoding="utf-8"))
    selected_ids = [
        str(item["id"]) for item in dataset[args.start_idx : args.end_idx]
    ]
    evaluation_identity = _validated_evaluation(args.eval_file, selected_ids)
    if not args.run_manifest.is_file():
        raise FileNotFoundError(f"run manifest not found: {args.run_manifest}")
    run_manifest = json.loads(args.run_manifest.read_text(encoding="utf-8"))
    run_manifest_sha = _sha256(args.run_manifest)
    evaluation_run = evaluation_identity["run_manifest"]
    if (
        evaluation_run.get("sha256") != run_manifest_sha
        or evaluation_run.get("protocol_sha256") != run_manifest.get("protocol_sha256")
        or run_manifest.get("instance_ids") != selected_ids
    ):
        raise ValueError("evaluation, run manifest, and selected slice are not the same run")
    records = parse_agent_logs(args.log_dir, results_file=args.eval_file)
    by_id = {}
    for record in records:
        if record.task_id in by_id:
            raise ValueError(f"multiple trajectories found for task {record.task_id}")
        by_id[record.task_id] = record
    missing = [instance_id for instance_id in selected_ids if instance_id not in by_id]
    extra = sorted(set(by_id).difference(selected_ids))
    if missing or extra:
        raise ValueError(
            f"trajectory slice mismatch: missing={missing[:8]}, extra={extra[:8]}"
        )
    ordered = [by_id[instance_id] for instance_id in selected_ids]
    wrong_protocol = [
        record.task_id
        for record in ordered
        if record.runtime_metadata.get("protocol_sha256")
        != run_manifest.get("protocol_sha256")
    ]
    if wrong_protocol:
        raise ValueError(
            f"trajectory logs are not bound to the source run: {wrong_protocol[:8]}"
        )
    missing_scores = [record.task_id for record in ordered if record.verifier_score is None]
    if missing_scores:
        raise ValueError(
            f"verifier score missing for {len(missing_scores)} tasks: {missing_scores[:8]}"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = args.output.with_name(f".{args.output.name}.{os.getpid()}.tmp")
    save_records(ordered, temporary_output)
    os.replace(temporary_output, args.output)
    manifest = {
        "format": "ordered_spreadsheetbench_trajectories_v1",
        "dataset_sha256": _sha256(dataset_file),
        "evaluation_sha256": _sha256(args.eval_file),
        "evaluation_format": evaluation_identity["format"],
        "source_run_manifest_sha256": run_manifest_sha,
        "source_protocol_sha256": run_manifest["protocol_sha256"],
        "start_idx": args.start_idx,
        "end_idx": args.end_idx,
        "record_count": len(ordered),
        "task_ids": selected_ids,
        "records_sha256": _sha256(args.output),
    }
    manifest_path = args.output.with_suffix(".manifest.json")
    temporary_manifest = manifest_path.with_name(
        f".{manifest_path.name}.{os.getpid()}.tmp"
    )
    temporary_manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    os.replace(temporary_manifest, manifest_path)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
