from __future__ import annotations

import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from .config import TARGET_SIZES, TIERS, Settings
from .generator import GptImageGenerator
from .models import CostResult, PipelineResult
from .upscaler import RealEsrganUpscaler


TARGET_COST_CNY = {"low": 0.08, "medium": 0.15, "high": 0.22}


class ImagePipeline:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or Settings.from_env(require_key=True)
        self.generator = GptImageGenerator(self.settings)
        self.upscaler = RealEsrganUpscaler(self.settings)

    def _make_run_dir(self, tier: str, target: str) -> tuple[str, Path]:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        run_id = f"{timestamp}-{tier}-{target}-{uuid.uuid4().hex[:8]}"
        run_dir = (self.settings.output_root / run_id).resolve()
        run_dir.mkdir(parents=True, exist_ok=False)
        return run_id, run_dir

    def _cost(self, tier: str, local_seconds: float) -> CostResult:
        local_cost = (
            self.settings.gpu_power_watts_estimate
            * local_seconds
            / 3_600_000
            * self.settings.electricity_cny_per_kwh
        )
        api_cost = self.settings.api_cost_cny[tier]
        target = TARGET_COST_CNY[tier]
        total = None if api_cost is None else api_cost + local_cost
        status = "需实测确认" if total is None else ("达标" if total <= target else "超标")
        return CostResult(
            api_cost_cny=api_cost,
            api_cost_status=(
                "确定（来自环境变量配置的渠道单价）" if api_cost is not None else "需实测确认（渠道未提供可核验单价/账单差额）"
            ),
            local_upscale_cost_cny_estimate=round(local_cost, 6),
            local_cost_basis=(
                f"估算：{self.settings.gpu_power_watts_estimate:g}W × {local_seconds:.3f}s × "
                f"{self.settings.electricity_cny_per_kwh:g}元/kWh"
            ),
            total_cost_cny=None if total is None else round(total, 6),
            target_cny=target,
            target_status=status,
        )

    def generate_and_upscale(
        self, prompt: str, tier: str, target: str
    ) -> PipelineResult:
        tier = tier.lower()
        target = target.lower()
        if tier not in TIERS:
            raise ValueError(f"tier must be one of {', '.join(TIERS)}")
        if target not in TARGET_SIZES:
            raise ValueError(f"target must be one of {', '.join(TARGET_SIZES)}")

        run_id, run_dir = self._make_run_dir(tier, target)
        started = time.perf_counter()
        generation = self.generator.generate(prompt, tier, run_dir)
        upscale = self.upscaler.upscale(
            Path(generation.image.path), run_dir, TARGET_SIZES[target]
        )
        total_seconds = time.perf_counter() - started
        manifest_path = run_dir / "manifest.json"
        result = PipelineResult(
            run_id=run_id,
            run_dir=str(run_dir),
            tier=tier,
            target=target,
            target_pixels=TARGET_SIZES[target],
            generation=generation,
            upscale=upscale,
            cost=self._cost(tier, upscale.total_seconds),
            total_seconds=round(total_seconds, 3),
            manifest_path=str(manifest_path),
        )
        manifest_path.write_text(
            json.dumps(result.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return result

    def upscale_existing(self, source: Path, target: str, output_dir: Path):
        target = target.lower()
        if target not in TARGET_SIZES:
            raise ValueError(f"target must be one of {', '.join(TARGET_SIZES)}")
        return self.upscaler.upscale(source, output_dir, TARGET_SIZES[target])
