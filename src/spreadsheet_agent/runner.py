"""
SpreadsheetBench Runner - Task execution logic.

This module handles loading benchmark data, running agents on tasks,
and collecting results. It is agnostic to the specific agent implementation.
"""

import json
import os
from pathlib import Path
import shutil
import tempfile
import traceback
from dataclasses import dataclass, field

from .agents.base import BaseSpreadsheetAgent, AgentContext


def _resolve_within(root: str | Path, *parts: str) -> str:
    resolved_root = Path(root).resolve()
    candidate = resolved_root.joinpath(*parts).resolve()
    try:
        candidate.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError(f"dataset path escapes configured root: {candidate}") from exc
    return str(candidate)


def _preserve_regular_output(source: str, destination: str) -> bool:
    """Copy a produced artifact without treating its existence as agent success."""
    if not os.path.isfile(source):
        return False
    target = Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", dir=target.parent
    )
    temporary = Path(temporary_name)
    try:
        with open(source, "rb") as input_handle, os.fdopen(
            descriptor, "wb"
        ) as output_handle:
            shutil.copyfileobj(input_handle, output_handle)
            output_handle.flush()
            os.fsync(output_handle.fileno())
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()
    return True


@dataclass
class BenchmarkInstance:
    """A single benchmark instance."""
    id: str
    instruction: str
    spreadsheet_path: str
    instruction_type: str = ""
    answer_position: str = ""
    metadata: dict = field(default_factory=dict)


@dataclass
class TestCaseResult:
    """Result for a single test case."""
    input_file: str
    output_file: str
    success: bool
    agent_completed: bool = False
    output_preserved: bool = False
    agent_answer: str = ""
    turns: int = 0
    error: str = ""
    failure_kind: str = ""


@dataclass
class InstanceResult:
    """Result for a benchmark instance (may have multiple test cases)."""
    id: str
    instruction: str
    success: bool
    test_cases: list[TestCaseResult] = field(default_factory=list)
    error: str = ""


def get_spreadsheet_content(file_path: str, max_rows: int = 5) -> str:
    """Get spreadsheet content in the format expected by SpreadsheetBench."""
    try:
        import openpyxl
        wb = openpyxl.load_workbook(file_path, data_only=True)
        ws = wb.active

        lines = []
        for i, row in enumerate(ws.iter_rows(values_only=True), 1):
            if i > max_rows:
                lines.append(f"... ({ws.max_row - max_rows} more rows)")
                break
            # Format as tuple-like representation
            row_values = [str(cell) if cell is not None else "" for cell in row]
            lines.append(str(tuple(row_values)))

        wb.close()
        return "\n".join(lines)
    except Exception as e:
        return f"[Could not read spreadsheet: {e}]"


def get_workbook_structure(
    file_path: str | Path,
    *,
    max_sample_rows: int = 6,
    max_cells_per_row: int = 32,
    max_chars: int = 12000,
) -> str:
    """Return a bounded, coordinate-aware summary of every worksheet."""
    import openpyxl

    workbook = openpyxl.load_workbook(file_path, data_only=False)
    try:
        lines = [
            "Workbook sheets: "
            + json.dumps(workbook.sheetnames, ensure_ascii=False)
        ]
        for worksheet in workbook.worksheets:
            details = [f"used_range={worksheet.calculate_dimension()}"]
            if worksheet.sheet_state != "visible":
                details.append(f"state={worksheet.sheet_state}")
            tables = [
                f"{table.name}:{table.ref}"
                for table in worksheet.tables.values()
            ]
            if tables:
                details.append("tables=" + ",".join(tables))
            if worksheet.auto_filter.ref:
                details.append(f"filter={worksheet.auto_filter.ref}")
            merged = [str(item) for item in worksheet.merged_cells.ranges]
            if merged:
                details.append("merged=" + ",".join(merged[:12]))
            lines.append(f"Sheet {worksheet.title}: " + "; ".join(details))

            sampled = 0
            scan_limit = min(worksheet.max_row or 0, 50)
            for row_index in range(1, scan_limit + 1):
                cells = []
                for column_index in range(1, (worksheet.max_column or 0) + 1):
                    cell = worksheet.cell(row=row_index, column=column_index)
                    if cell.value is None:
                        continue
                    value = str(cell.value).replace("\r", " ").replace("\n", " ")
                    if len(value) > 120:
                        value = value[:117] + "..."
                    cells.append(f"{cell.coordinate}={value}")
                    if len(cells) >= max_cells_per_row:
                        cells.append("...")
                        break
                if not cells:
                    continue
                lines.append(f"  row {row_index}: " + " | ".join(cells))
                sampled += 1
                if sampled >= max_sample_rows:
                    break
            if sampled == 0:
                lines.append("  (no non-empty cells in first 50 rows)")
            if len("\n".join(lines)) >= max_chars:
                lines.append("... (workbook structure summary truncated)")
                break
        return "\n".join(lines)[:max_chars]
    finally:
        workbook.close()


class SpreadsheetBenchRunner:
    """
    Runner for executing agents on SpreadsheetBench.

    Loads the packaged ``dataset.json`` and runs one isolated task at a time.
    """

    def __init__(
        self,
        agent: BaseSpreadsheetAgent,
        data_path: str,
        output_dir: str = "outputs/spreadsheetbench",
        working_dir: str | None = None,
        workbook_structure_preflight: bool = False,
    ):
        """
        Initialize the runner.

        Args:
            agent: The agent to run
            data_path: Path to the packaged SpreadsheetBench data directory
            output_dir: Directory to save output spreadsheets
            working_dir: Working directory for agent (default: system temp folder)
        """
        self.agent = agent
        self.data_path = data_path
        self.output_dir = output_dir
        self.workbook_structure_preflight = bool(workbook_structure_preflight)
        if working_dir is not None:
            working_dir = os.path.abspath(working_dir)
            os.makedirs(working_dir, exist_ok=True)
        self._custom_working_dir = working_dir
        self.working_dir: str | None = working_dir

    def load_data(self) -> list[BenchmarkInstance]:
        """Load benchmark instances in the stored dataset order."""
        data_file = _resolve_within(self.data_path, "dataset.json")
        if not os.path.isfile(data_file):
            raise FileNotFoundError(f"dataset.json not found in {self.data_path}")
        print(f"Loading data from: {data_file}")
        with open(data_file, "r", encoding="utf-8") as f:
            data_list = json.load(f)
        instances = []
        for data in data_list:
            instance_id = str(data["id"])
            instances.append(BenchmarkInstance(
                id=instance_id,
                instruction=data["instruction"],
                spreadsheet_path=str(data.get("spreadsheet_path", instance_id)),
                instruction_type=data.get("instruction_type", ""),
                answer_position=data.get("answer_position", ""),
                metadata={k: v for k, v in data.items()
                          if k not in ("id", "instruction", "spreadsheet_path",
                                       "instruction_type", "answer_position")},
            ))

        return instances

    def _find_spreadsheet_dir(self, instance: BenchmarkInstance) -> str | None:
        """Resolve the dataset-declared task directory without fallbacks."""
        path = _resolve_within(self.data_path, instance.spreadsheet_path)
        return path if os.path.isdir(path) else None

    def _find_input_files(self, spreadsheet_dir: str) -> list[str]:
        """Return regular input workbooks for Verified or full SpreadsheetBench."""
        files = os.listdir(spreadsheet_dir)
        matches = sorted(
            filename
            for filename in files
            if (
                filename.endswith("_input.xlsx")
                or filename.endswith("_init.xlsx")
                or filename == "initial.xlsx"
            )
        )
        return matches

    def run_test_case(
        self,
        instance: BenchmarkInstance,
        input_file: str,
        *,
        runtime_instance_id: str | None = None,
        retrieval_id: str | None = None,
    ) -> TestCaseResult:
        """Run one named case with distinct runtime and retrieval identities."""
        spreadsheet_dir = self._find_spreadsheet_dir(instance)
        if spreadsheet_dir is None:
            raise FileNotFoundError(
                f"Spreadsheet directory not found for {instance.id}"
            )
        if input_file not in self._find_input_files(spreadsheet_dir):
            raise ValueError(f"input workbook is absent for {instance.id}: {input_file}")
        return self._run_test_case(
            instance=instance,
            spreadsheet_dir=spreadsheet_dir,
            input_file=input_file,
            runtime_instance_id=runtime_instance_id,
            retrieval_id=retrieval_id,
        )

    def run_instance(self, instance: BenchmarkInstance) -> InstanceResult:
        """Run the agent on a single benchmark instance."""
        spreadsheet_dir = self._find_spreadsheet_dir(instance)

        if spreadsheet_dir is None:
            return InstanceResult(
                id=instance.id,
                instruction=instance.instruction,
                success=False,
                error=f"Spreadsheet directory not found for {instance.id}",
            )

        input_files = self._find_input_files(spreadsheet_dir)

        if not input_files:
            return InstanceResult(
                id=instance.id,
                instruction=instance.instruction,
                success=False,
                error=f"No input files found in {spreadsheet_dir}",
            )

        result = InstanceResult(
            id=instance.id,
            instruction=instance.instruction,
            success=True,
        )

        for input_file in input_files:
            test_result = self._run_test_case(
                instance=instance,
                spreadsheet_dir=spreadsheet_dir,
                input_file=input_file,
            )
            result.test_cases.append(test_result)
            if not test_result.success:
                result.success = False

        return result

    def _run_test_case(
        self,
        instance: BenchmarkInstance,
        spreadsheet_dir: str,
        input_file: str,
        *,
        runtime_instance_id: str | None = None,
        retrieval_id: str | None = None,
    ) -> TestCaseResult:
        """Run a single test case."""
        input_path = os.path.join(spreadsheet_dir, input_file)
        if Path(input_path).is_symlink() or not Path(input_path).is_file():
            raise ValueError(f"input workbook must be a regular non-symlink file: {input_path}")

        # Determine output filename based on input naming convention
        if input_file.endswith("_input.xlsx"):
            output_file = input_file[: -len("_input.xlsx")] + "_output.xlsx"
        elif "_init.xlsx" in input_file:
            # Verified format: 1_13-1_init.xlsx -> 1_13-1_output.xlsx
            output_file = input_file.replace("_init.xlsx", "_output.xlsx")
        else:
            base = os.path.splitext(input_file)[0]
            output_file = f"{base}_output.xlsx"

        # Setup output path
        output_subdir = _resolve_within(self.output_dir, instance.spreadsheet_path)
        os.makedirs(output_subdir, exist_ok=True)
        final_output_path = os.path.join(output_subdir, output_file)
        if os.path.exists(final_output_path):
            os.remove(final_output_path)

        # Create task-specific subdirectory within working_dir
        # Use instance id and input file base name to create unique subdirectory
        input_base = os.path.splitext(input_file)[0]
        task_subdir_name = f"{runtime_instance_id or instance.id}_{input_base}".replace("/", "_").replace("\\", "_")
        if self.working_dir is None:
            raise RuntimeError("runner requires an explicit working directory")
        task_working_dir = _resolve_within(self.working_dir, task_subdir_name)
        task_path = Path(task_working_dir)
        if task_path.exists() or task_path.is_symlink():
            raise FileExistsError(f"task working directory already exists: {task_path}")
        task_path.mkdir(mode=0o700)

        # Copy input to task-specific working directory
        work_input = os.path.join(task_working_dir, "input.xlsx")
        work_output = os.path.join(task_working_dir, "output.xlsx")
        shutil.copy(input_path, work_input)

        # Remove any existing output
        if os.path.exists(work_output):
            os.remove(work_output)

        # Get spreadsheet content
        spreadsheet_content = (
            get_workbook_structure(work_input)
            if self.workbook_structure_preflight
            else get_spreadsheet_content(work_input)
        )

        # Create context
        context = AgentContext(
            working_dir=task_working_dir,
            input_file=work_input,
            output_file=work_output,
            instruction=instance.instruction,
            spreadsheet_content=spreadsheet_content,
            instruction_type=instance.instruction_type,
            answer_position=instance.answer_position,
            instance_id=runtime_instance_id or instance.id,
            retrieval_id=retrieval_id or "",
        )

        # Run agent
        try:
            agent_result = self.agent.run(context)

            agent_completed = bool(
                agent_result.get("agent_completed", agent_result.get("success"))
            )
            output_preserved = _preserve_regular_output(
                work_output, final_output_path
            )

            if not bool(agent_result.get("success")):
                return TestCaseResult(
                    input_file=input_file,
                    output_file=output_file,
                    success=False,
                    agent_completed=agent_completed,
                    output_preserved=output_preserved,
                    agent_answer=agent_result.get("answer", ""),
                    turns=agent_result.get("turns", 0),
                    error=(
                        agent_result.get("error")
                        or "Agent did not complete the task successfully"
                    ),
                    failure_kind=str(
                        agent_result.get("failure_kind") or "agent_terminal_failure"
                    ),
                )
            if output_preserved:
                return TestCaseResult(
                    input_file=input_file,
                    output_file=output_file,
                    success=True,
                    agent_completed=agent_completed,
                    output_preserved=True,
                    agent_answer=agent_result.get("answer", ""),
                    turns=agent_result.get("turns", 0),
                )
            else:
                return TestCaseResult(
                    input_file=input_file,
                    output_file=output_file,
                    success=False,
                    agent_completed=agent_completed,
                    output_preserved=False,
                    agent_answer=agent_result.get("answer", ""),
                    turns=agent_result.get("turns", 0),
                    error="Output file was not created",
                    failure_kind="missing_output",
                )

        except Exception as e:
            output_preserved = _preserve_regular_output(
                work_output, final_output_path
            )
            return TestCaseResult(
                input_file=input_file,
                output_file=output_file,
                success=False,
                agent_completed=False,
                output_preserved=output_preserved,
                error=f"{e}\n{traceback.format_exc()}",
                failure_kind="runner_exception",
            )

