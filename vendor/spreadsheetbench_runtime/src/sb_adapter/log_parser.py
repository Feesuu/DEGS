from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from .schemas import RawTrajectoryRecord, TrajectoryStep

ROLE_HEADER_RE = re.compile(r"^## \[(\d+)\] ([A-Z]+)\s*$", re.MULTILINE)
PROTOCOL_SHA_RE = re.compile(r"^\*\*Run Protocol SHA256\*\*:\s*([0-9a-f]{64})\s*$", re.MULTILINE)


def load_results_index(results_file: str | Path | None) -> dict[str, dict[str, Any]]:
    """Load benchmark runner/evaluation result files across common layouts.

    The returned keys include task id, explicit instance id, and trial-qualified
    keys when a result row contains seed/trial/repeat metadata.  Rows may come
    from runner results or evaluate_with_official outputs.
    """
    if not results_file:
        return {}
    path = Path(results_file)
    if not path.exists():
        return {}
    paths = [path]
    if path.is_dir():
        paths = sorted(p for p in path.rglob("*.json") if p.is_file())
    index: dict[str, dict[str, Any]] = {}
    for result_path in paths:
        try:
            payload = json.loads(result_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        rows: list[Any]
        if isinstance(payload, dict):
            rows = payload.get("results") or payload.get("records") or []
            if not rows and any(k in payload for k in ("id", "task_id", "instance_id")):
                rows = [payload]
        elif isinstance(payload, list):
            rows = payload
        else:
            rows = []
        for item in rows:
            if not isinstance(item, dict):
                continue
            item_id = item.get("id") or item.get("task_id") or item.get("instance_id")
            if item_id is None:
                continue
            keys = {str(item_id)}
            trial_index, seed, _ = _infer_trial_metadata(result_path, str(item_id), item)
            keys.add(_result_lookup_key(str(item_id), trial_index, seed))
            for key in keys:
                # Prefer a trial-qualified row over an unqualified duplicate.
                index[key] = item
    return index


def parse_agent_logs(log_dir: str | Path, *, results_file: str | Path | None = None) -> list[RawTrajectoryRecord]:
    log_dir = Path(log_dir)
    results_index = load_results_index(results_file)
    records: list[RawTrajectoryRecord] = []
    seen: set[str] = set()
    for path in sorted(log_dir.rglob("*.md")):
        record = parse_markdown_log(path, results_index=results_index)
        if record is not None and record.trajectory_id not in seen:
            seen.add(record.trajectory_id)
            records.append(record)
    return records


def parse_markdown_log(path: str | Path, *, results_index: dict[str, dict[str, Any]] | None = None) -> RawTrajectoryRecord | None:
    path = Path(path)
    text = path.read_text(encoding="utf-8", errors="replace")
    messages = _parse_md_messages(text)
    if not messages:
        return None
    task_message = _first_user_task(messages)
    task_context = _parse_task_context(task_message)
    task_id = _logged_markdown_instance_id(text) or _infer_task_id(path, task_context)
    trial_index, seed, version_id = _infer_trial_metadata(path, task_id, {})
    result = _lookup_result(results_index or {}, task_id, trial_index, seed)
    # Result metadata may be more informative than path metadata.
    trial_index, seed, version_id = _infer_trial_metadata(path, task_id, result, fallback_trial=trial_index, fallback_seed=seed, fallback_version=version_id)
    steps = _messages_to_steps(messages)
    final_response = _infer_final_response(messages)
    success, verifier_score, verifier_feedback = _result_fields(result)
    trajectory_id = _trajectory_id(task_id, trial_index, seed, version_id)
    return RawTrajectoryRecord(
        trajectory_id=trajectory_id,
        task_id=task_id,
        trial_index=trial_index,
        instruction=task_context.get("instruction") or result.get("instruction") or "",
        instruction_type=task_context.get("instruction_type") or result.get("instruction_type") or "",
        answer_position=task_context.get("answer_position") or result.get("answer_position") or "",
        spreadsheet_path=task_context.get("spreadsheet_path") or result.get("spreadsheet_path") or "",
        output_path=task_context.get("output_path") or result.get("output_path") or "",
        final_response=final_response,
        success=success,
        verifier_score=verifier_score,
        verifier_feedback=verifier_feedback,
        steps=steps,
        runtime_metadata={
            "log_file": str(path),
            "log_format": "markdown",
            "seed": seed,
            "version_id": version_id,
            "protocol_sha256": (
                PROTOCOL_SHA_RE.search(text).group(1)
                if PROTOCOL_SHA_RE.search(text)
                else None
            ),
        },
        service_metadata={},
        extra={"runner_result": result},
    )


def save_records(records: list[RawTrajectoryRecord], output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps([r.to_dict() for r in records], ensure_ascii=False, indent=2), encoding="utf-8")


def load_records(path: str | Path) -> list[RawTrajectoryRecord]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return [RawTrajectoryRecord.from_dict(x) for x in payload]


def _parse_md_messages(text: str) -> list[dict[str, str]]:
    matches = list(ROLE_HEADER_RE.finditer(text))
    messages: list[dict[str, str]] = []
    for i, match in enumerate(matches):
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        content = text[start:end]
        content = re.sub(r"\n---\s*$", "", content.strip(), flags=re.S)
        messages.append({"role": match.group(2).lower(), "content": content.strip()})
    return messages


def _first_user_task(messages: list[dict[str, str]]) -> str:
    for msg in messages:
        if msg.get("role") == "user" and "### instruction" in msg.get("content", ""):
            content = msg["content"]
            if content.startswith("Task:"):
                content = content[len("Task:"):].strip()
            return content
    for msg in messages:
        if msg.get("role") == "user":
            return msg.get("content", "")
    return ""


def _parse_task_context(text: str) -> dict[str, str]:
    fields = ["working_directory", "instruction", "spreadsheet_path", "spreadsheet_content", "instruction_type", "answer_position", "output_path"]
    result: dict[str, str] = {}
    for field in fields:
        pattern = rf"^### {re.escape(field)}\s*$"
        m = re.search(pattern, text, flags=re.M)
        if not m:
            continue
        start = m.end()
        next_m = re.search(r"^### ", text[start:], flags=re.M)
        end = start + next_m.start() if next_m else len(text)
        value = text[start:end].strip()
        value = re.sub(r"\n---[\s\S]*$", "", value).strip()
        result[field] = value
    return result


def _logged_markdown_instance_id(text: str) -> str | None:
    match = re.search(r"^\*\*Instance ID JSON\*\*: (.+)$", text, flags=re.M)
    if not match:
        return None
    try:
        value = json.loads(match.group(1))
    except json.JSONDecodeError:
        return None
    return str(value) if value else None


def _infer_task_id(path: Path, task_context: dict[str, str]) -> str:
    stem = path.stem
    # Adapter log filenames use the agent name followed by the task id.
    for prefix in ("preloaded_experience_agent_", "cli_only_agent_"):
        if stem.startswith(prefix):
            remainder = stem[len(prefix):]
            # Strip common trial/seed suffixes without stripping task ids like 13-1.
            remainder = re.sub(r"(?:__|_)(?:trial|repeat|run)[-_]?\d+.*$", "", remainder)
            remainder = re.sub(r"(?:__|_)seed[-_]?\d+.*$", "", remainder)
            return remainder
    spreadsheet_path = task_context.get("spreadsheet_path", "")
    if spreadsheet_path:
        parts = [p for p in re.split(r"[/\\]", spreadsheet_path) if p]
        if len(parts) >= 2:
            return parts[-2]
    return stem


def _infer_trial_metadata(
    path: Path,
    task_id: str,
    payload: dict[str, Any],
    *,
    fallback_trial: int | None = None,
    fallback_seed: str | None = None,
    fallback_version: str | None = None,
) -> tuple[int, str | None, str]:
    trial = _first_int(payload, ["trial_index", "trial", "repeat", "run_index", "sample_index"], fallback_trial)
    seed_value = payload.get("seed", payload.get("run_seed", payload.get("shuffle_seed", fallback_seed)))
    seed = None if seed_value is None else str(seed_value)
    text = "/".join(path.parts[-4:])
    for pattern in (r"trial[-_]?([0-9]+)", r"repeat[-_]?([0-9]+)", r"run[-_]?([0-9]+)"):
        m = re.search(pattern, text, flags=re.I)
        if m and fallback_trial is None and not any(k in payload for k in ("trial_index", "trial", "repeat", "run_index", "sample_index")):
            trial = int(m.group(1))
            break
    if seed is None:
        m = re.search(r"seed[-_]?([0-9]+)", text, flags=re.I)
        if m:
            seed = m.group(1)
    if trial is None:
        trial = 0
    version = str(payload.get("version_id") or fallback_version or _short_hash(str(path.resolve())))
    return int(trial), seed, version


def _first_int(payload: dict[str, Any], keys: list[str], fallback: int | None) -> int | None:
    for key in keys:
        if key in payload and payload[key] is not None:
            try:
                return int(payload[key])
            except Exception:
                continue
    return fallback


def _lookup_result(index: dict[str, dict[str, Any]], task_id: str, trial_index: int, seed: str | None) -> dict[str, Any]:
    return index.get(_result_lookup_key(task_id, trial_index, seed)) or index.get(task_id) or {}


def _result_lookup_key(task_id: str, trial_index: int, seed: str | None) -> str:
    return f"{task_id}::trial{trial_index}::seed{seed or 'none'}"


def _trajectory_id(task_id: str, trial_index: int, seed: str | None, version_id: str) -> str:
    return f"{task_id}::trial{trial_index}::seed{seed or 'none'}::log{version_id}"


def _short_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]


def _messages_to_steps(messages: list[dict[str, str]]) -> list[TrajectoryStep]:
    steps: list[TrajectoryStep] = []
    step_id = 0
    for i, msg in enumerate(messages):
        if msg.get("role") != "assistant":
            continue
        raw = msg.get("content", "")
        action = _extract_action(raw)
        obs = ""
        if i + 1 < len(messages) and messages[i + 1].get("role") == "user":
            nxt = messages[i + 1].get("content", "")
            if nxt.strip().startswith("Observation:"):
                obs = nxt.strip()[len("Observation:"):].strip()
        if action or obs or raw:
            steps.append(TrajectoryStep(step_id=step_id, raw_model_output=raw, action=action, observation=obs, tool_name="bash" if "bash" in action else None, action_valid=bool(action)))
            step_id += 1
    return steps


def _extract_action(raw: str) -> str:
    if "ACTION: TASK_COMPLETE" in raw:
        return "ACTION: TASK_COMPLETE"
    m = re.search(r"Action:\s*(\{[\s\S]*?\})\s*$", raw.strip())
    if not m:
        m = re.search(r"Action:\s*(\{[\s\S]*?\})", raw)
    if not m:
        return ""
    block = m.group(1).strip()
    try:
        obj = json.loads(block)
        if isinstance(obj, dict):
            if obj.get("name") and isinstance(obj.get("arguments"), dict):
                return json.dumps(obj, ensure_ascii=False)
    except Exception:
        pass
    return block


def _infer_final_response(messages: list[dict[str, str]]) -> str | None:
    for msg in reversed(messages):
        if msg.get("role") == "assistant":
            return msg.get("content", "")
    return None


def _result_fields(result: dict[str, Any]) -> tuple[bool, float | None, str | None]:
    if not result:
        return False, None, None
    test_cases = result.get("test_cases") or result.get("cases") or []
    soft_score = _numeric(result.get("soft_score"))
    hard_score = _numeric(result.get("hard_score"))
    if soft_score is None or hard_score is None:
        raise ValueError("verifier result lacks explicit soft_score/hard_score")
    if test_cases:
        if any(not isinstance(case.get("passed"), bool) for case in test_cases):
            raise ValueError("verifier test case lacks an explicit passed boolean")
        msgs = [str(x.get("message") or x.get("error") or "") for x in test_cases if str(x.get("message") or x.get("error") or "").strip()]
    else:
        msgs = []
    success = hard_score >= 1.0
    feedback_parts = []
    if result.get("error"):
        feedback_parts.append(str(result.get("error")))
    if result.get("message"):
        feedback_parts.append(str(result.get("message")))
    feedback_parts.extend(msgs[:20])
    feedback = "; ".join(x for x in feedback_parts if x) or None
    return success, float(soft_score), feedback


def _numeric(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None
