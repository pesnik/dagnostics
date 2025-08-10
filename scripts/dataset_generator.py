import json
from typing import List, Optional

import typer
from typer import Option

from dagnostics.analysis.analyzer import DAGAnalyzer
from dagnostics.cli.utils import initialize_components_for_notifications
from dagnostics.core.models import LogEntry, TaskInstance


class DatasetGenerator:
    def __init__(self):
        pass

    def get_failed_tasks(self, analyzer: DAGAnalyzer, ndays: int) -> List[TaskInstance]:
        failed_tasks = analyzer.airflow_client.get_failed_tasks(ndays * 24 * 60)

        if not failed_tasks:
            typer.echo("No failed tasks found.")
            return []

        typer.echo(f"Found {len(failed_tasks)} failed tasks.")
        return failed_tasks

    def generate_dataset(self, analyzer: DAGAnalyzer, failed_tasks: List[TaskInstance]):
        training_data = []
        for task in failed_tasks:
            try:
                # Get all tries for this task instance
                typer.echo(
                    f"🔍 Fetching tries for {task.dag_id}.{task.task_id} (run: {task.run_id})..."
                )
                task_tries = analyzer.airflow_client.get_task_tries(
                    task.dag_id, task.task_id, task.run_id
                )

                # Filter only failed tries
                failed_tries = [
                    try_instance
                    for try_instance in task_tries
                    if try_instance.state == "failed" and try_instance.try_number > 0
                ]

                if not failed_tries:
                    typer.echo(
                        f"⚠️  No failed tries found for {task.dag_id}.{task.task_id} (run: {task.run_id})"
                    )
                    continue

                for failed_try in failed_tries:
                    try:
                        typer.echo(
                            f"📝 Analyzing {task.dag_id}.{task.task_id} (run: {task.run_id}, try: {failed_try.try_number})..."
                        )
                        error_candidates, error_message = self.get_dataset(
                            analyzer,
                            failed_try.dag_id,
                            failed_try.task_id,
                            failed_try.run_id,
                            failed_try.try_number,
                        )
                        if error_candidates:
                            error_messages = "\n".join(
                                [error.message for error in error_candidates]
                            )
                            print(error_messages, error_message)
                            training_data.append(
                                {"candidates": error_messages, "error": error_message}
                            )
                    except Exception as e:
                        print(e)
            except Exception as e:
                print(e)

        with open("training_data.jsonl", "w") as f:
            for train_data in training_data:
                json.dump(train_data, f)
                f.write("\n")

    def get_dataset(
        self,
        analyzer: DAGAnalyzer,
        dag_id: str,
        task_id: str,
        run_id: str,
        try_number: int,
    ) -> tuple[Optional[List[LogEntry]], str]:
        _, candidates, error_message = analyzer.extract_task_error_for_sms(
            dag_id, task_id, run_id, try_number
        )

        return candidates, error_message


app = typer.Typer()


@app.command()
def main(
    config_file: Optional[str] = Option(
        "config/config.yaml",
        "--config",
        "-c",
        help="Path to configuration file (default: searches standard locations)",
    ),
    llm_provider: str = Option(
        "gemini",
        "--llm",
        "-l",
        help="LLM provider to use (ollama, openai, anthropic, gemini)",
    ),
):
    _, analyzer = initialize_components_for_notifications(config_file, llm_provider)

    generator = DatasetGenerator()
    failed_tasks = generator.get_failed_tasks(analyzer, 1)
    generator.generate_dataset(analyzer, failed_tasks)


if __name__ == "__main__":
    app()
