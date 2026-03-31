from __future__ import annotations

import argparse
import json
import os
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from lm_eval import simple_evaluate
from lm_eval.models.api_models import TemplateAPI
from lm_eval.utils import handle_non_serializable


ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class ApiEndpointConfig:
    name: str
    description: str | None
    url: str
    model_id: str
    endpoint_type: str
    api_key_name: str
    default_request_params: dict[str, Any]


class StreamingSampleList(list):
    def __init__(self, task_name: str | None, output_dir: Path):
        super().__init__()
        self.task_name = task_name or "unknown-task"
        self.output_dir = output_dir
        self.stream_path = output_dir / f"samples_{self.task_name}.jsonl"

    def append(self, item: Any) -> None:
        sample_dump = json.dumps(
            item,
            default=handle_non_serializable,
            ensure_ascii=False,
        )
        with self.stream_path.open("a", encoding="utf-8") as handle:
            handle.write(sample_dump)
            handle.write("\n")
        super().append(item)


def append_jsonl(path: Path, payload: Any) -> None:
    dumped = json.dumps(
        payload,
        default=handle_non_serializable,
        ensure_ascii=False,
    )
    with path.open("a", encoding="utf-8") as handle:
        handle.write(dumped)
        handle.write("\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run lm-eval tasks from model_configs.json using credentials from .env."
    )
    parser.add_argument("--task", help="lm-eval task name")
    parser.add_argument(
        "--config",
        help="Model config name from model_configs.json",
    )
    parser.add_argument(
        "--model-configs",
        type=Path,
        help="Path to model_configs.json",
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        help="Path to a dotenv-style env file",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directory to write meta.json and results.json into",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=8,
        help="Number of concurrent API requests for lm-eval",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=900,
        help="Per-request timeout in seconds",
    )
    parser.add_argument(
        "--limit",
        type=parse_limit,
        help="Optional lm-eval limit. Accepts an integer count or a fraction like 0.1",
    )
    parser.add_argument(
        "--log-samples",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include per-sample outputs in results.json and stream them to JSONL as they are produced",
    )
    parser.add_argument(
        "--predict-only",
        action="store_true",
        help="Skip metric computation and only save generations",
    )
    parser.add_argument(
        "--show-configs",
        action="store_true",
        help="Print available config names and exit",
    )
    return parser.parse_args()


def parse_limit(raw: str) -> int | float:
    if "." in raw:
        return float(raw)
    return int(raw)


def load_env_file(path: Path) -> dict[str, str]:
    if not path.exists():
        raise FileNotFoundError(f"Missing env file: {path}")

    values: dict[str, str] = {}
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        values[key.strip()] = strip_quotes(value.strip())
    return values


def strip_quotes(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def load_model_configs(path: Path) -> list[ApiEndpointConfig]:
    payload = json.loads(path.read_text())
    configs: list[ApiEndpointConfig] = []

    for raw_config in payload.get("configs", []):
        endpoint = raw_config["target"]["api_endpoint"]
        configs.append(
            ApiEndpointConfig(
                name=raw_config["name"],
                description=raw_config.get("description"),
                url=endpoint["url"],
                model_id=endpoint["model_id"],
                endpoint_type=endpoint["type"],
                api_key_name=endpoint["api_key_name"],
                default_request_params=endpoint.get("default_request_params", {}),
            )
        )

    if not configs:
        raise ValueError(f"No configs found in {path}")
    return configs


def get_config(configs: list[ApiEndpointConfig], name: str) -> ApiEndpointConfig:
    for config in configs:
        if config.name == name:
            return config
    available = ", ".join(sorted(config.name for config in configs))
    raise ValueError(f"Unknown config '{name}'. Available configs: {available}")


def resolve_api_key(
    config: ApiEndpointConfig, env_values: dict[str, str], env_path: Path
) -> str:
    api_key = env_values.get(config.api_key_name) or os.environ.get(config.api_key_name)
    if not api_key:
        raise ValueError(
            f"Missing {config.api_key_name} in {env_path} or the process environment"
        )

    os.environ[config.api_key_name] = api_key
    # lm-eval's OpenAI-compatible backends look specifically at OPENAI_API_KEY.
    os.environ["OPENAI_API_KEY"] = api_key
    return api_key


def build_model_args(config: ApiEndpointConfig, args: argparse.Namespace) -> dict[str, Any]:
    if config.endpoint_type != "chat":
        raise ValueError(f"Unsupported endpoint type '{config.endpoint_type}' for {config.name}")

    return {
        "model": config.model_id,
        "base_url": config.url,
        "tokenizer_backend": "none",
        "tokenized_requests": False,
        "num_concurrent": args.concurrency,
        "timeout": args.timeout,
    }


def build_gen_kwargs(config: ApiEndpointConfig) -> dict[str, Any]:
    gen_kwargs = deepcopy(config.default_request_params)
    max_tokens = gen_kwargs.pop("max_tokens", None)
    if max_tokens is not None:
        gen_kwargs["max_gen_toks"] = max_tokens

    if "do_sample" not in gen_kwargs:
        temperature = float(gen_kwargs.get("temperature", 0) or 0)
        top_p = float(gen_kwargs.get("top_p", 1) or 1)
        gen_kwargs["do_sample"] = temperature > 0 or top_p < 1

    return gen_kwargs


def default_output_dir(task: str, config_name: str) -> Path:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    safe_config = config_name.replace("/", "-")
    return ROOT / "runs" / f"{task}-{safe_config}-{timestamp}"


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n")


@contextmanager
def stream_logged_samples(output_dir: Path, task_name: str, enabled: bool):
    if enabled:
        # Newer lm-eval forks no longer expose TaskOutput for monkey-patching.
        # Keep compatibility by creating the file eagerly and letting lm-eval
        # handle sample logging internally.
        (output_dir / f"samples_{task_name}.jsonl").touch(exist_ok=True)
    yield


@contextmanager
def stream_generation_responses(output_dir: Path, task_name: str, enabled: bool):
    if not enabled:
        yield
        return

    responses_path = output_dir / f"responses_{task_name}.jsonl"
    responses_path.touch(exist_ok=True)
    original_model_call = TemplateAPI.model_call
    original_amodel_call = TemplateAPI.amodel_call

    def patched_model_call(self, messages, *, generate=True, gen_kwargs=None, **kwargs):
        outputs = original_model_call(
            self,
            messages,
            generate=generate,
            gen_kwargs=gen_kwargs,
            **kwargs,
        )
        if generate:
            append_jsonl(
                responses_path,
                {
                    "logged_at_utc": datetime.now(UTC).isoformat(),
                    "mode": "sync",
                    "messages": messages,
                    "gen_kwargs": gen_kwargs,
                    "outputs": outputs,
                    "parsed": self.parse_generations(outputs=outputs) if outputs else None,
                },
            )
        return outputs

    async def patched_amodel_call(
        self,
        session,
        sem,
        messages,
        *,
        generate=True,
        cache_keys=None,
        ctxlens=None,
        gen_kwargs=None,
        **kwargs,
    ):
        answers = await original_amodel_call(
            self,
            session,
            sem,
            messages,
            generate=generate,
            cache_keys=cache_keys,
            ctxlens=ctxlens,
            gen_kwargs=gen_kwargs,
            **kwargs,
        )
        if generate:
            append_jsonl(
                responses_path,
                {
                    "logged_at_utc": datetime.now(UTC).isoformat(),
                    "mode": "async",
                    "messages": messages,
                    "gen_kwargs": gen_kwargs,
                    "answers": answers,
                },
            )
        return answers

    TemplateAPI.model_call = patched_model_call
    TemplateAPI.amodel_call = patched_amodel_call
    try:
        yield
    finally:
        TemplateAPI.model_call = original_model_call
        TemplateAPI.amodel_call = original_amodel_call


def main() -> None:
    args = parse_args()
    if args.model_configs is None:
        raise ValueError("--model-configs is required")
    configs = load_model_configs(args.model_configs)

    if args.show_configs:
        for config in configs:
            print(config.name)
        return

    if not args.task:
        raise ValueError("--task is required")
    if not args.config:
        raise ValueError("--config is required")
    if args.env_file is None:
        raise ValueError("--env-file is required")

    config = get_config(configs, args.config)
    env_values = load_env_file(args.env_file)
    resolve_api_key(config, env_values, args.env_file)

    output_dir = args.output_dir or default_output_dir(args.task, config.name)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_args = build_model_args(config, args)
    gen_kwargs = build_gen_kwargs(config)
    metadata = {
        "started_at_utc": datetime.now(UTC).isoformat(),
        "task": args.task,
        "config_name": config.name,
        "config_description": config.description,
        "env_file": str(args.env_file),
        "model_configs": str(args.model_configs),
        "api_key_name": config.api_key_name,
        "model_args": model_args,
        "gen_kwargs": gen_kwargs,
        "limit": args.limit,
        "log_samples": args.log_samples,
        "predict_only": args.predict_only,
        "output_dir": str(output_dir),
    }
    write_json(output_dir / "meta.json", metadata)

    print(f"Running task '{args.task}' with config '{config.name}'")
    print(f"Writing outputs to {output_dir}")

    with stream_logged_samples(output_dir, args.task, args.log_samples):
        with stream_generation_responses(output_dir, args.task, args.log_samples):
            results = simple_evaluate(
                model="local-chat-completions",
                model_args=model_args,
                tasks=[args.task],
                apply_chat_template=True,
                gen_kwargs=gen_kwargs,
                limit=args.limit,
                log_samples=args.log_samples,
                predict_only=args.predict_only,
            )

    metadata["finished_at_utc"] = datetime.now(UTC).isoformat()
    write_json(output_dir / "meta.json", metadata)
    write_json(output_dir / "results.json", results)

    summary = results.get("results", {}).get(args.task)
    if summary:
        print(json.dumps(summary, indent=2, ensure_ascii=False))
    else:
        print("Run completed. See results.json for details.")


if __name__ == "__main__":
    main()
