from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from image_pipeline.config import Settings
from image_pipeline.pipeline import ImagePipeline, TARGET_COST_CNY
from image_pipeline.telemetry import TelemetryStore, parse_utc_timestamp


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="gpt-image-2 generation + local Real-ESRGAN 2K/4K pipeline"
    )
    commands = parser.add_subparsers(dest="command", required=True)

    generate = commands.add_parser("generate", help="Generate and upscale an image")
    generate.add_argument("--prompt", required=True)
    generate.add_argument("--tier", choices=("low", "medium", "high"), default="low")
    generate.add_argument("--target", choices=("2k", "4k"), default="4k")

    upscale = commands.add_parser("upscale", help="Upscale an existing image")
    upscale.add_argument("--input", required=True, type=Path)
    upscale.add_argument("--target", choices=("2k", "4k"), default="4k")
    upscale.add_argument("--output-dir", required=True, type=Path)

    upscale_batch = commands.add_parser(
        "upscale-batch", help="Upscale every image in a directory serially"
    )
    upscale_batch.add_argument("--input-dir", required=True, type=Path)
    upscale_batch.add_argument("--target", choices=("2k", "4k"), default="4k")
    upscale_batch.add_argument("--output-dir", required=True, type=Path)

    batch = commands.add_parser(
        "batch", help="Generate JSONL prompt items serially and write JSONL results"
    )
    batch.add_argument("--input", required=True, type=Path)
    batch.add_argument("--output", required=True, type=Path)
    batch.add_argument("--default-tier", choices=("low", "medium", "high"), default="low")
    batch.add_argument("--default-target", choices=("2k", "4k"), default="4k")

    commands.add_parser("cost-table", help="Show configured per-tier cost status")

    telemetry_task = commands.add_parser(
        "telemetry-task", help="Show a secret-safe task telemetry timeline"
    )
    telemetry_task.add_argument("--database", required=True, type=Path)
    telemetry_task.add_argument("--task-id", required=True)

    telemetry_stats = commands.add_parser(
        "telemetry-stats", help="Show SQLite-derived telemetry window statistics"
    )
    telemetry_stats.add_argument("--database", required=True, type=Path)
    telemetry_stats.add_argument("--since", required=True)
    telemetry_stats.add_argument("--until")

    serve = commands.add_parser("serve", help="Run the local FastAPI service")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", default=8012, type=int)
    commands.add_parser(
        "upscale-worker",
        help="Poll the cloud service and process claimed jobs with local Real-ESRGAN",
    )
    return parser


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = _parser().parse_args()
    if args.command == "generate":
        result = ImagePipeline().generate_and_upscale(args.prompt, args.tier, args.target)
        print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
        return 0
    if args.command == "upscale":
        settings = Settings.from_env(require_key=False)
        result = ImagePipeline.__new__(ImagePipeline)
        result.settings = settings
        from image_pipeline.upscaler import RealEsrganUpscaler

        upscale_result = RealEsrganUpscaler(settings).upscale(
            args.input, args.output_dir, {"2k": (2048, 2048), "4k": (3840, 2160)}[args.target]
        )
        print(json.dumps(asdict(upscale_result), ensure_ascii=False, indent=2))
        return 0
    if args.command == "upscale-batch":
        from image_pipeline.config import TARGET_SIZES
        from image_pipeline.upscaler import RealEsrganUpscaler

        extensions = {".png", ".jpg", ".jpeg", ".webp"}
        sources = sorted(
            path for path in args.input_dir.iterdir()
            if path.is_file() and path.suffix.lower() in extensions
        )
        if not sources:
            raise SystemExit("No PNG/JPEG/WebP images found in input directory")
        results = RealEsrganUpscaler(Settings.from_env(require_key=False)).upscale_batch(
            sources, args.output_dir, TARGET_SIZES[args.target]
        )
        print(json.dumps([asdict(item) for item in results], ensure_ascii=False, indent=2))
        return 0
    if args.command == "batch":
        pipeline = ImagePipeline()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        completed = 0
        with args.input.open("r", encoding="utf-8") as source, args.output.open(
            "w", encoding="utf-8"
        ) as destination:
            for line_number, line in enumerate(source, start=1):
                if not line.strip():
                    continue
                item = json.loads(line)
                result = pipeline.generate_and_upscale(
                    item["prompt"],
                    item.get("tier", args.default_tier),
                    item.get("target", args.default_target),
                )
                destination.write(
                    json.dumps(result.to_dict(), ensure_ascii=False) + "\n"
                )
                destination.flush()
                completed += 1
                print(f"completed {completed} item(s); last input line={line_number}", file=sys.stderr)
        return 0
    if args.command == "telemetry-task":
        telemetry = TelemetryStore(args.database, initialize=False)
        try:
            payload = telemetry.task_timeline(args.task_id)
        except KeyError as exc:
            raise SystemExit(str(exc)) from None
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    if args.command == "telemetry-stats":
        from datetime import datetime, timezone

        since = parse_utc_timestamp(args.since)
        until = (
            parse_utc_timestamp(args.until)
            if args.until
            else datetime.now(tz=timezone.utc).timestamp()
        )
        payload = TelemetryStore(args.database, initialize=False).window_stats(since, until)
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    if args.command == "cost-table":
        settings = Settings.from_env(require_key=False)
        rows = []
        for tier in ("low", "medium", "high"):
            price = settings.api_cost_cny[tier]
            rows.append(
                {
                    "tier": tier,
                    "api_cost_cny": price,
                    "target_cny": TARGET_COST_CNY[tier],
                    "status": "需实测确认" if price is None else ("达标" if price <= TARGET_COST_CNY[tier] else "超标"),
                }
            )
        print(json.dumps(rows, ensure_ascii=False, indent=2))
        return 0
    if args.command == "serve":
        import uvicorn

        uvicorn.run(
            "image_pipeline.service:app",
            host=args.host,
            port=args.port,
            proxy_headers=True,
            forwarded_allow_ips="*",
        )
        return 0
    if args.command == "upscale-worker":
        from image_pipeline.remote_worker import run_remote_worker

        run_remote_worker()
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
