from __future__ import annotations

from pathlib import Path
import shutil

from openpyxl import Workbook

from spreadsheet_agent.runner import BenchmarkInstance, SpreadsheetBenchRunner


class FailingAgentWithOutput:
    def run(self, context):
        shutil.copyfile(context.input_file, context.output_file)
        return {
            "success": False,
            "agent_completed": False,
            "output_exists": True,
            "answer": "",
            "turns": 30,
            "error": "Max turns exceeded",
        }


class RaisingAgentWithOutput:
    def run(self, context):
        shutil.copyfile(context.input_file, context.output_file)
        raise RuntimeError("synthetic runtime failure")


def test_failed_agent_output_is_preserved_without_marking_agent_success(
    tmp_path: Path,
) -> None:
    data_path = tmp_path / "data"
    task_path = data_path / "synthetic-task"
    task_path.mkdir(parents=True)
    workbook = Workbook()
    workbook.active["A1"] = "synthetic"
    workbook.save(task_path / "initial.xlsx")

    output_dir = tmp_path / "outputs"
    runner = SpreadsheetBenchRunner(
        agent=FailingAgentWithOutput(),
        data_path=str(data_path),
        output_dir=str(output_dir),
        working_dir=str(tmp_path / "work"),
    )

    result = runner.run_instance(
        BenchmarkInstance(
            id="synthetic-task",
            instruction="Create the requested artifact.",
            spreadsheet_path="synthetic-task",
        )
    )

    assert result.success is False
    assert len(result.test_cases) == 1
    test_case = result.test_cases[0]
    assert test_case.success is False
    assert test_case.agent_completed is False
    assert test_case.output_preserved is True
    assert test_case.error == "Max turns exceeded"
    assert (output_dir / "synthetic-task" / "initial_output.xlsx").is_file()


def test_output_is_preserved_even_when_agent_raises(tmp_path: Path) -> None:
    data_path = tmp_path / "data"
    task_path = data_path / "synthetic-task"
    task_path.mkdir(parents=True)
    workbook = Workbook()
    workbook.active["A1"] = "synthetic"
    workbook.save(task_path / "initial.xlsx")

    output_dir = tmp_path / "outputs"
    runner = SpreadsheetBenchRunner(
        agent=RaisingAgentWithOutput(),
        data_path=str(data_path),
        output_dir=str(output_dir),
        working_dir=str(tmp_path / "work"),
    )

    result = runner.run_instance(
        BenchmarkInstance(
            id="synthetic-task",
            instruction="Create the requested artifact.",
            spreadsheet_path="synthetic-task",
        )
    )

    test_case = result.test_cases[0]
    assert result.success is False
    assert test_case.agent_completed is False
    assert test_case.output_preserved is True
    assert "synthetic runtime failure" in test_case.error
    assert (output_dir / "synthetic-task" / "initial_output.xlsx").is_file()
