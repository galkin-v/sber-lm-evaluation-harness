from __future__ import annotations

import argparse
import errno
import json
import os
import re
import subprocess
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parent


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generic provider runner for lm-eval.")
    parser.add_argument("--benchmark-name", default=None)
    parser.add_argument("--task", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--candidate-base-url", required=True)
    parser.add_argument("--candidate-model-id", required=True)
    parser.add_argument("--candidate-api-key", default="")
    parser.add_argument("--parallelism", type=int, default=1)
    parser.add_argument("--request-timeout", type=int, default=900)
    parser.add_argument("--limit-samples", type=int, default=-1)
    parser.add_argument("--request-params-json", default="{}")
    parser.add_argument("--resume", default="1")
    parser.add_argument("--show-live-stats", default="1")
    return parser.parse_args()


def _to_bool(raw: str) -> bool:
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _safe_json_obj(raw: str) -> dict[str, Any]:
    payload = json.loads(raw) if raw.strip() else {}
    return dict(payload) if isinstance(payload, Mapping) else {}


def _api_key_env_name(model_id: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9]+", "_", model_id).strip("_").upper()
    return f"LM_HARNESS_PROVIDER_{normalized or 'MODEL'}_API_KEY"


def _first_text_content(value: Any) -> str | None:
    if isinstance(value, str):
        text = value.strip()
        return text or None
    if isinstance(value, Mapping):
        content = value.get("content")
        nested = _first_text_content(content)
        if nested:
            return nested
        text = value.get("text")
        nested = _first_text_content(text)
        if nested:
            return nested
        for nested_value in value.values():
            nested = _first_text_content(nested_value)
            if nested:
                return nested
        return None
    if isinstance(value, list):
        for item in value:
            nested = _first_text_content(item)
            if nested:
                return nested
    return None


def _extract_prompt(value: Any) -> str:
    prompt = _first_text_content(value)
    if prompt is None:
        return ""
    try:
        parsed = json.loads(prompt)
    except json.JSONDecodeError:
        return prompt
    if isinstance(parsed, list):
        for item in parsed:
            if isinstance(item, Mapping) and isinstance(item.get("content"), str):
                return str(item["content"])
    if isinstance(parsed, Mapping) and isinstance(parsed.get("content"), str):
        return str(parsed["content"])
    return prompt


def _extract_response(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()

    if isinstance(value, list):
        for item in value:
            text = _extract_response(item)
            if text:
                return text
        return ""

    if not isinstance(value, Mapping):
        return ""

    # Most lm-harness response logs keep parsed text as a top-level list.
    parsed_text = _extract_response(value.get("parsed"))
    if parsed_text:
        return parsed_text

    # OpenAI-compatible response envelope.
    outputs = value.get("outputs")
    if isinstance(outputs, Mapping):
        choices = outputs.get("choices")
        if isinstance(choices, list) and choices:
            first_choice = choices[0] if isinstance(choices[0], Mapping) else {}
            message = first_choice.get("message") if isinstance(first_choice, Mapping) else {}
            content = _first_text_content(message)
            if content:
                return content

    # Some providers may emit raw choices directly.
    choices = value.get("choices")
    if isinstance(choices, list) and choices:
        first_choice = choices[0] if isinstance(choices[0], Mapping) else {}
        message = first_choice.get("message") if isinstance(first_choice, Mapping) else {}
        content = _first_text_content(message)
        if content:
            return content

    # Backward/alternate fields used by some bridges.
    for key in ("answers", "outputs", "response", "content", "text"):
        text = _extract_response(value.get(key))
        if text:
            return text

    return ""


def _sanitize_metric_name(metric_name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", metric_name).strip("_").lower() or "metric"


def _extract_metrics(results_payload: Mapping[str, Any], task_name: str) -> dict[str, float]:
    task_results = results_payload.get("results", {}).get(task_name)
    if not isinstance(task_results, Mapping):
        return {}
    metrics: dict[str, float] = {}
    for raw_key, raw_value in task_results.items():
        if not isinstance(raw_key, str):
            continue
        key = raw_key.strip()
        if not key or key.endswith("_stderr") or "_stderr," in key:
            continue
        metric_name = key.split(",", 1)[0].strip() or key
        try:
            metrics[_sanitize_metric_name(metric_name)] = float(raw_value)
        except (TypeError, ValueError):
            continue
    return metrics


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")


def _stream_and_capture_output(
    command: list[str],
    *,
    cwd: Path,
) -> tuple[int, str]:
    captured_chunks: list[str] = []

    # On POSIX, run child process in a pseudo-terminal so progress bars (tqdm) render.
    if os.name != "nt":
        import pty

        master_fd, slave_fd = pty.openpty()
        process = subprocess.Popen(
            command,
            cwd=cwd,
            stdout=slave_fd,
            stderr=slave_fd,
            text=False,
            bufsize=0,
        )
        os.close(slave_fd)

        line_buffer = ""
        try:
            while True:
                try:
                    chunk = os.read(master_fd, 4096)
                except OSError as exc:
                    if exc.errno == errno.EIO:
                        break
                    raise
                if not chunk:
                    break

                decoded = chunk.decode("utf-8", errors="replace")
                captured_chunks.append(decoded)
                for ch in decoded:
                    if ch in {"\r", "\n"}:
                        if line_buffer:
                            print(line_buffer, flush=True)
                            line_buffer = ""
                    else:
                        line_buffer += ch
        finally:
            os.close(master_fd)

        if line_buffer:
            print(line_buffer, flush=True)

        return process.wait(), "".join(captured_chunks)

    process = subprocess.Popen(
        command,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    if process.stdout is None:
        return process.wait(), ""

    for raw_line in process.stdout:
        captured_chunks.append(raw_line)
        text_line = raw_line.rstrip("\n")
        if text_line:
            print(text_line, flush=True)

    return process.wait(), "".join(captured_chunks)


def main() -> int:
    args = _parse_args()
    started_at = datetime.now(UTC)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    show_live_stats = _to_bool(args.show_live_stats)
    request_params = _safe_json_obj(args.request_params_json)
    model_request_params = {
        key: value
        for key, value in request_params.items()
        if not str(key).startswith("external_") and not str(key).startswith("lm_eval_")
    }
    log_samples = bool(request_params.get("external_log_samples", True))
    predict_only = bool(request_params.get("external_predict_only", False))

    with tempfile.TemporaryDirectory(prefix="lm-harness-provider-") as tmp_dir:
        tmp_path = Path(tmp_dir)
        model_configs_path = tmp_path / "model_configs.json"
        env_path = tmp_path / ".env"
        config_name = "lm-harness-provider-runtime"
        api_key_name = _api_key_env_name(args.candidate_model_id)
        model_configs_payload = {
            "configs": [
                {
                    "name": config_name,
                    "description": "Generated by provider contract runner",
                    "target": {
                        "api_endpoint": {
                            "url": f"{args.candidate_base_url}/chat/completions",
                            "type": "chat",
                            "model_id": args.candidate_model_id,
                            "api_key_name": api_key_name,
                            "default_request_params": model_request_params,
                        }
                    },
                }
            ]
        }
        model_configs_path.write_text(
            json.dumps(model_configs_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        env_path.write_text(f"{api_key_name}={args.candidate_api_key}\n", encoding="utf-8")

        command = [
            sys.executable,
            str((ROOT / "main.py").resolve()),
            "--task",
            args.task,
            "--config",
            config_name,
            "--model-configs",
            str(model_configs_path),
            "--env-file",
            str(env_path),
            "--output-dir",
            str(output_dir),
            "--concurrency",
            str(max(1, args.parallelism)),
            "--timeout",
            str(max(1, args.request_timeout)),
        ]
        if args.limit_samples >= 0:
            command.extend(["--limit", str(args.limit_samples)])
        if not log_samples:
            command.append("--no-log-samples")
        if predict_only:
            command.append("--predict-only")

        returncode, combined_output = _stream_and_capture_output(command, cwd=ROOT)
        if returncode != 0:
            raise RuntimeError(
                f"lm-eval provider run failed (exit_code={returncode}): "
                f"{combined_output.strip()}"
            )

    results_path = output_dir / "results.json"
    if not results_path.exists():
        raise FileNotFoundError(f"Missing results.json in {output_dir}")
    results_payload = json.loads(results_path.read_text(encoding="utf-8"))
    metrics = _extract_metrics(results_payload, args.task)

    response_candidates = [output_dir / f"responses_{args.task}.jsonl"]
    response_candidates.extend(sorted(output_dir.glob("responses_*.jsonl")))
    responses_path = next((path for path in response_candidates if path.exists()), None)
    predictions: list[dict[str, Any]] = []
    if responses_path is not None:
        with responses_path.open("r", encoding="utf-8") as handle:
            for sample_id, line in enumerate(handle):
                line = line.strip()
                if not line:
                    continue
                payload = json.loads(line)
                prompt = _extract_prompt(payload.get("messages"))
                response = _extract_response(payload.get("answers", payload.get("outputs")))
                error = None if response else "ExternalPredictionError: empty response"
                predictions.append(
                    {
                        "sample_id": sample_id,
                        "prompt": prompt,
                        "response": response,
                        "target": "",
                        "status": "scored" if error is None else "error",
                        "error": error,
                        "scores": {},
                        "metadata": {
                            "external_task": args.task,
                            "external_source": responses_path.name,
                            "model_id": args.candidate_model_id,
                        },
                    }
                )

    _write_jsonl(output_dir / "byob_predictions.jsonl", predictions)

    sample_count = len(predictions)
    scores = {
        metric_name: {
            "stats": {
                "count": sample_count,
                "mean": round(value, 4),
                "stddev": 0.0,
                "stderr": 0.0,
            },
            "value": value,
        }
        for metric_name, value in metrics.items()
    }
    benchmark_name = args.benchmark_name or args.task
    _write_json(
        output_dir / "byob_results.json",
        {
            "tasks": {
                benchmark_name: {
                    "metrics": {
                        "pass@1": {
                            "scores": scores,
                        }
                    }
                }
            }
        },
    )

    finished_at = datetime.now(UTC)
    inference_time = max(0.0, (finished_at - started_at).total_seconds())
    successful_count = sum(1 for row in predictions if row.get("error") in (None, ""))
    _write_json(
        output_dir / "eval_factory_metrics.json",
        {
            "response_stats": {
                "count": sample_count,
                "successful_count": successful_count,
                "avg_latency_ms": 0.0,
                "avg_total_tokens": 0.0,
                "avg_completion_tokens": 0.0,
            },
            "timing": {
                "started_at": started_at.isoformat(),
                "finished_at": finished_at.isoformat(),
                "inference_time_seconds": inference_time,
            },
        },
    )
    _write_json(
        output_dir / "params.json",
        {
            "parallelism": max(1, args.parallelism),
            "request_timeout": max(1, args.request_timeout),
            "limit_samples": args.limit_samples if args.limit_samples >= 0 else None,
            "resume": _to_bool(args.resume),
            "show_live_stats": show_live_stats,
            "request_params": request_params,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
