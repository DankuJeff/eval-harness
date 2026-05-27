"""
Step 9 — W&B logging integration.

WandbLogger is initialized once at the entry point (cli.py or api/server.py)
and passed into runners as a dependency. Never instantiated inside a runner.
Call finish() at the entry point after all results are logged.
"""

import os
import wandb


class WandbLogger:
    def __init__(self, run_name: str | None = None):
        # Temporarily remove WANDB_ENTITY so the SDK resolves the entity from
        # the API key rather than from the env var. The env var value may not
        # match the actual account slug, causing upsertBucket failures.
        _entity = os.environ.pop("WANDB_ENTITY", None)
        try:
            self._run = wandb.init(
                project=os.environ["WANDB_PROJECT"],
                name=run_name,
                reinit="finish_previous",
            )
        finally:
            if _entity is not None:
                os.environ["WANDB_ENTITY"] = _entity

    def log_run(self, results: list[dict]) -> None:
        """Log one result object per eval case to W&B."""
        for result in results:
            self._run.log(
                {
                    "model": result["model"],
                    "model_id": result["model_id"],
                    "prompt_version": result["prompt_version"],
                    "task_type": result["task_type"],
                    "case_id": result["case_id"],
                    "mode": result["mode"],
                    "pass_rate": result["pass_rate"],
                    "timestamp": result["timestamp"],
                    **{
                        f"metric/{k}": v
                        for k, v in result["metric_scores"].items()
                    },
                }
            )

    def finish(self) -> None:
        wandb.finish()
