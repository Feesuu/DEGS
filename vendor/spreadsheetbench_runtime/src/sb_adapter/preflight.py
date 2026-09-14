from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import tempfile
import urllib.request
from pathlib import Path

from .evaluate import _recalculate_workbook
from .transport import validate_service_url


ROOT = Path(__file__).resolve().parents[2]


def _workbook_count(task_dir: Path, patterns: tuple[str, ...]) -> int:
    matches = {
        path.resolve()
        for pattern in patterns
        for path in task_dir.glob(pattern)
        if path.is_file()
    }
    return len(matches)


def _probe_libreoffice(soffice: str, workbook: Path) -> dict:
    with tempfile.TemporaryDirectory(prefix="sb_adapter_lo_preflight_") as tmp:
        root = Path(tmp)
        source = root / "input.xlsx"
        source.write_bytes(workbook.read_bytes())
        converted = Path(
            _recalculate_workbook(
                str(source),
                str(root / "recalculated"),
                "preflight",
                soffice=soffice,
                timeout_seconds=60,
            )
        )
        return {
            "conversion_exit_code": 0,
            "conversion_output_created": converted.is_file(),
            "conversion_message": "sandboxed LibreOffice conversion succeeded",
        }


def _request_json(
    url: str,
    api_key: str,
    timeout: float,
    *,
    payload: dict | None = None,
) -> dict:
    body = json.dumps(payload).encode("utf-8") if payload is not None else None
    request = urllib.request.Request(
        url,
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST" if payload is not None else "GET",
    )
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _probe_models(base_url: str, api_key: str, timeout: float) -> dict:
    payload = _request_json(
        base_url.rstrip("/") + "/models",
        api_key,
        timeout,
    )
    return {
        "status": "ok",
        "model_ids": [str(item.get("id")) for item in payload.get("data", [])],
    }


def _probe_generation(
    base_url: str,
    api_key: str,
    model: str,
    timeout: float,
) -> dict:
    report = _probe_models(base_url, api_key, timeout)
    if model not in report["model_ids"]:
        raise RuntimeError(
            f"requested generation model {model!r} is not served: {report['model_ids']}"
        )
    payload = _request_json(
        base_url.rstrip("/") + "/chat/completions",
        api_key,
        timeout,
        payload={
            "model": model,
            "messages": [{"role": "user", "content": "Reply with ready."}],
            "max_tokens": 16,
            "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": False},
        },
    )
    choices = payload.get("choices") or []
    content = (
        choices[0].get("message", {}).get("content")
        if choices and isinstance(choices[0], dict)
        else None
    )
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError("chat completion returned no non-empty message content")
    response_model = payload.get("model")
    if response_model is not None and str(response_model) != model:
        raise RuntimeError(
            f"chat response model mismatch: requested={model!r}, response={response_model!r}"
        )
    report.update(
        {
            "model": model,
            "chat_nonempty": True,
            "chat_content_chars": len(content),
            "response_model": response_model or model,
        }
    )
    return report


def _probe_embedding(
    base_url: str,
    api_key: str,
    model: str,
    timeout: float,
) -> dict:
    report = _probe_models(base_url, api_key, timeout)
    if model not in report["model_ids"]:
        raise RuntimeError(
            f"requested embedding model {model!r} is not served: {report['model_ids']}"
        )
    payload = _request_json(
        base_url.rstrip("/") + "/embeddings",
        api_key,
        timeout,
        payload={"model": model, "input": ["SpreadsheetBench adapter preflight"]},
    )
    rows = payload.get("data") or []
    vector = rows[0].get("embedding") if rows and isinstance(rows[0], dict) else None
    if (
        not isinstance(vector, list)
        or not vector
        or not all(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
            for value in vector
        )
    ):
        raise RuntimeError("embedding endpoint returned no finite numeric vector")
    report.update(
        {
            "model": model,
            "embedding_finite": True,
            "embedding_dimension": len(vector),
        }
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the packaged adapter runtime.")
    parser.add_argument(
        "--data-path",
        type=Path,
        default=ROOT / "data" / "spreadsheetbench_verified_400",
    )
    parser.add_argument(
        "--records",
        type=Path,
        default=ROOT / "data" / "frozen_train_0_200" / "ordered_records.json",
    )
    parser.add_argument("--generation-base-url")
    parser.add_argument("--generation-model", default="Qwen3.5-9B-AWQ")
    parser.add_argument("--generation-key-env", default="YD5_API_KEY")
    parser.add_argument("--embedding-base-url")
    parser.add_argument("--embedding-model", default="Qwen3-Embedding-8B")
    parser.add_argument("--embedding-key-env", default="EMBED_API_KEY")
    parser.add_argument("--network-timeout", type=float, default=10.0)
    args = parser.parse_args()

    dataset_file = args.data_path / "dataset.json"
    dataset = json.loads(dataset_file.read_text(encoding="utf-8"))
    records = json.loads(args.records.read_text(encoding="utf-8"))
    dataset_ids = [str(item["id"]) for item in dataset]
    record_ids = [str(item["task_id"]) for item in records]
    task_dirs = []
    invalid_workbooks = []
    first_input = None
    for item in dataset:
        task_dir = args.data_path / str(item.get("spreadsheet_path", item["id"]))
        task_dirs.append(task_dir)
        input_matches = _workbook_count(
            task_dir,
            ("*_input.xlsx", "*_init.xlsx", "initial.xlsx", "input.xlsx"),
        )
        golden_matches = _workbook_count(
            task_dir,
            ("*_answer.xlsx", "*_golden.xlsx", "golden.xlsx"),
        )
        if input_matches != 1 or golden_matches != 1:
            invalid_workbooks.append(
                {
                    "id": str(item["id"]),
                    "inputs": input_matches,
                    "goldens": golden_matches,
                }
            )
        if first_input is None:
            candidates = [
                path
                for pattern in ("*_input.xlsx", "*_init.xlsx", "initial.xlsx", "input.xlsx")
                for path in task_dir.glob(pattern)
            ]
            if candidates:
                first_input = candidates[0]

    soffice = shutil.which("soffice") or "/usr/lib/libreoffice/program/soffice"
    libreoffice = {"executable": soffice, "available": Path(soffice).is_file()}
    if libreoffice["available"]:
        result = subprocess.run(
            [soffice, "--version"], capture_output=True, text=True, check=True
        )
        libreoffice["version"] = result.stdout.strip()
        if first_input is not None:
            try:
                libreoffice.update(_probe_libreoffice(soffice, first_input))
            except Exception as exc:
                libreoffice.update(
                    {
                        "conversion_exit_code": None,
                        "conversion_output_created": False,
                        "conversion_message": f"{type(exc).__name__}: {exc}",
                    }
                )

    report = {
        "dataset": {
            "count": len(dataset),
            "unique_ids": len(dataset_ids) == len(set(dataset_ids)),
            "task_directory_count": sum(path.is_dir() for path in task_dirs),
            "tasks_with_exactly_one_input_and_golden": len(dataset) - len(invalid_workbooks),
            "invalid_workbook_tasks": invalid_workbooks,
        },
        "frozen_train_0_200": {
            "count": len(records),
            "exact_dataset_prefix": record_ids == dataset_ids[:200],
            "success_count": sum(item.get("success") is True for item in records),
            "failure_count": sum(item.get("success") is False for item in records),
        },
        "split": {
            "train": [0, 200],
            "heldout": [200, 400],
            "overlap": sorted(set(dataset_ids[:200]).intersection(dataset_ids[200:400])),
        },
        "libreoffice": libreoffice,
    }
    for label, base_url, key_env, model, probe in (
        (
            "generation",
            args.generation_base_url,
            args.generation_key_env,
            args.generation_model,
            _probe_generation,
        ),
        (
            "embedding",
            args.embedding_base_url,
            args.embedding_key_env,
            args.embedding_model,
            _probe_embedding,
        ),
    ):
        if not base_url:
            continue
        key = os.getenv(key_env, "EMPTY")
        try:
            validate_service_url(base_url)
            report[label] = probe(base_url, key, model, args.network_timeout)
        except Exception as exc:
            report[label] = {"status": "error", "error": f"{type(exc).__name__}: {exc}"}

    required_checks = (
        report["dataset"]["count"] == 400,
        report["dataset"]["unique_ids"],
        report["dataset"]["task_directory_count"] == 400,
        report["dataset"]["tasks_with_exactly_one_input_and_golden"] == 400,
        report["frozen_train_0_200"]["count"] == 200,
        report["frozen_train_0_200"]["exact_dataset_prefix"],
        not report["split"]["overlap"],
        report["libreoffice"]["available"],
        report["libreoffice"].get("conversion_exit_code") == 0,
        report["libreoffice"].get("conversion_output_created") is True,
    )
    requested_service_checks = [
        report[label].get("status") == "ok"
        for label, base_url in (
            ("generation", args.generation_base_url),
            ("embedding", args.embedding_base_url),
        )
        if base_url
    ]
    report["ok"] = all((*required_checks, *requested_service_checks))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
