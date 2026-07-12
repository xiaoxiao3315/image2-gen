from __future__ import annotations

import csv
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from capacity_benchmark import nearest_rank


ROOT = Path(__file__).resolve().parent
API_DIRS = [
    ROOT / "capacity-benchmarks" / "20260711T101109Z-api-97b81276",
    ROOT / "capacity-benchmarks" / "20260711T104052Z-api-72bbd983",
]
SOAK_DIR = ROOT / "capacity-benchmarks" / "20260711T101807Z-api-soak-01f49986"
GPU_DIR = ROOT / "capacity-benchmarks" / "20260711T095446Z-gpu-8274eeb4"
TOKEN_DIR = ROOT / "capacity-benchmarks" / "20260711T102427Z-tokens-fad1069f"
REPORT = ROOT / "capacity-benchmark-report-2026-07-11.md"


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def fmt(value: float | None, digits: int = 3) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def failure_text(records: list[dict[str, Any]]) -> str:
    counts = Counter(record.get("error_category") or "unknown" for record in records if record["outcome"] != "success")
    return "无" if not counts else ", ".join(f"{name}×{count}" for name, count in sorted(counts.items()))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    api_records: list[dict[str, Any]] = []
    api_steps: dict[int, dict[str, Any]] = {}
    for directory in API_DIRS:
        api_records.extend(load_jsonl(directory / "api-requests.jsonl"))
        for step in load_json(directory / "api-staircase-summary.json")["steps"]:
            api_steps[int(step["concurrency"])] = step
    grouped_api: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for record in api_records:
        grouped_api[int(record["concurrency"])].append(record)

    requested_steps = [1, 3, 5, 8, 9, 10, 11, 12, 16, 24, 32]
    api_rows: list[dict[str, Any]] = []
    for concurrency in requested_steps:
        records = grouped_api.get(concurrency, [])
        if not records:
            api_rows.append(
                {
                    "concurrency": concurrency,
                    "success_total": "未执行（安全停止）",
                    "success_rate": "—",
                    "failure_types": "—",
                    "p50_seconds": "—",
                    "p95_seconds": "—",
                    "combined_rpm": "—",
                    "round_rpm_range": "—",
                    "judgement": "C=10/C=12已崩溃，未继续施压",
                }
            )
            continue
        successes = [record for record in records if record["outcome"] == "success"]
        latencies = [float(record["usable_image_elapsed_s"]) for record in successes]
        step = api_steps[concurrency]
        judgement = "稳定" if step["stable"] else ("崩溃" if step["collapse"] else "不稳定")
        api_rows.append(
            {
                "concurrency": concurrency,
                "success_total": f"{len(successes)}/{len(records)}",
                "success_rate": f"{len(successes) / len(records) * 100:.1f}%",
                "failure_types": failure_text(records),
                "p50_seconds": fmt(nearest_rank(latencies, 50)),
                "p95_seconds": fmt(nearest_rank(latencies, 95)),
                "combined_rpm": fmt(step["combined_throughput_images_per_minute"]),
                "round_rpm_range": (
                    f"{min(step['round_throughputs']):.3f}–{max(step['round_throughputs']):.3f}"
                ),
                "judgement": judgement,
            }
        )
    write_csv(ROOT / "api-capacity-benchmark-2026-07-11.csv", api_rows)

    soak = load_json(SOAK_DIR / "api-soak-summary.json")
    full_minutes = [
        row["successes"]
        for row in soak["completion_minute_buckets"]
        if row["minute_index"] < int(soak["requested_duration_seconds"] // 60)
    ]
    api_conservative_rpm = min(full_minutes)

    gpu_summary = load_json(GPU_DIR / "gpu-summary.json")
    gpu_records = load_jsonl(GPU_DIR / "gpu-requests.jsonl")
    grouped_gpu: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for record in gpu_records:
        grouped_gpu[(record["target"], int(record["concurrency"]))].append(record)
    gpu_rows: list[dict[str, Any]] = []
    chosen_gpu_rpm: dict[str, float] = {}
    for target in ("2k", "4k"):
        levels = gpu_summary["targets"][target]["levels"]
        for level in levels:
            concurrency = int(level["concurrency"])
            rounds = level["rounds"]
            records = grouped_gpu[(target, concurrency)]
            successes = [record for record in records if record["outcome"] == "success"]
            durations = [float(record["total_seconds"]) for record in successes]
            walls = [float(row["total_wall_seconds"]) for row in rounds]
            gpu_rows.append(
                {
                    "target": target,
                    "concurrency": concurrency,
                    "success_total": f"{len(successes)}/{len(records)}",
                    "wall_median_seconds": round(statistics.median(walls), 3),
                    "wall_range_seconds": f"{min(walls):.3f}–{max(walls):.3f}",
                    "item_mean_seconds": round(statistics.mean(durations), 3),
                    "item_p95_seconds": nearest_rank(durations, 95),
                    "median_rpm": level["median_throughput_images_per_minute"],
                    "conservative_rpm": level["conservative_throughput_images_per_minute"],
                    "peak_vram_mib": level["peak_vram_mib"],
                    "errors": failure_text(records),
                    "stable": level["stable"],
                }
            )
        best = max(
            (level for level in levels if level["stable"]),
            key=lambda level: level["conservative_throughput_images_per_minute"],
        )
        chosen_gpu_rpm[target] = float(best["conservative_throughput_images_per_minute"])
    write_csv(ROOT / "gpu-capacity-benchmark-2026-07-11.csv", gpu_rows)

    token_records = load_jsonl(TOKEN_DIR / "token-samples.jsonl")
    token_rows = [
        {
            "quality": record["quality"],
            "http_status": record["http_status"],
            "actual_pixels": f"{record['actual_width']}×{record['actual_height']}",
            "elapsed_seconds": record["usable_image_elapsed_s"],
            "input_tokens": record["usage"]["input_tokens"],
            "output_tokens": record["usage"]["output_tokens"],
            "total_tokens": record["usage"]["total_tokens"],
        }
        for record in token_records
    ]
    write_csv(ROOT / "image-token-samples-2026-07-11.csv", token_rows)

    capacity_rows = []
    for target in ("2k", "4k"):
        rpm = min(float(api_conservative_rpm), chosen_gpu_rpm[target])
        capacity_rows.append(
            {
                "target": target.upper(),
                "api_conservative_rpm": api_conservative_rpm,
                "gpu_conservative_rpm": chosen_gpu_rpm[target],
                "system_rpm": rpm,
                "system_per_hour": rpm * 60,
                "system_per_day_theoretical": rpm * 1440,
                "bottleneck": "本地Real-ESRGAN超分",
            }
        )

    api_table = "\n".join(
        f"| {row['concurrency']} | {row['success_total']} | {row['success_rate']} | {row['failure_types']} | "
        f"{row['p50_seconds']} / {row['p95_seconds']} | {row['combined_rpm']} | {row['round_rpm_range']} | {row['judgement']} |"
        for row in api_rows
    )
    gpu_table = "\n".join(
        f"| {row['target'].upper()} | {row['concurrency']} | {row['success_total']} | {row['wall_median_seconds']:.3f} "
        f"({row['wall_range_seconds']}) | {row['item_mean_seconds']:.3f} / {row['item_p95_seconds']:.3f} | "
        f"{row['median_rpm']:.3f} / {row['conservative_rpm']:.3f} | {row['peak_vram_mib']:.3f} | "
        f"{row['errors']} | {'是' if row['stable'] else '否'} |"
        for row in gpu_rows
    )
    token_table = "\n".join(
        f"| {index} | {row['quality']} | {row['actual_pixels']} | {row['elapsed_seconds']:.3f} | "
        f"{row['input_tokens']} | {row['output_tokens']} | {row['total_tokens']} |"
        for index, row in enumerate(token_rows, start=1)
    )
    capacity_table = "\n".join(
        f"| {row['target']} | {row['api_conservative_rpm']:.3f} | {row['gpu_conservative_rpm']:.3f} | "
        f"{row['system_rpm']:.3f} | {row['system_per_hour']:.1f} | {row['system_per_day_theoretical']:.1f} | {row['bottleneck']} |"
        for row in capacity_rows
    )

    report = f"""# 图像生成 + 本地超分真实产能压测报告

实测日期：2026-07-11（America/Los_Angeles）

## 最终结论

- API 最大稳定并发：**8**。C=8 阶梯两轮 16/16 成功；C=9 两轮各出现1个503，已不稳定；C=10 合计14/20成功并触发崩溃规则；C=12两轮均仅8/12成功、各4个503。
- C=8 持续闭环：**105/105成功，329.220秒，平均19.136张/分**；完整测量分钟最少完成 **{api_conservative_rpm}张**，因此产能外推采用保守 **{api_conservative_rpm:.3f}张/分**，不取单分钟21张峰值。
- 本地超分吞吐最优并发均为 **M=6**。M=8没有OOM，但2K/4K吞吐都回落；4K M=8三轮吞吐CV超过10%，不满足稳定规则。
- 端到端瓶颈明确为：**本地 Real-ESRGAN 超分段**。

## API 阶梯并发

成功定义为 HTTP 2xx + 响应含图 + Base64/URL 可取回 + Pillow 可解码真实像素。P95采用 nearest-rank。吞吐为成功图片数除以该轮完整墙钟时间；没有再乘并发数。

| 并发 | 成功/总数 | 成功率 | 失败类型 | P50/P95 秒 | 合并吞吐 张/分 | 两轮吞吐范围 | 判定 |
|---:|---:|---:|---|---:|---:|---:|---|
{api_table}

安全说明：原阶梯在C=12出现66.7%成功率后停止16/24/32；随后只在崩溃点以下补测9/10/11，C=10再次崩溃，因此C=11也停止执行。失败均如实计入，未把503、连接错误或未执行项算成功。

### C=8 持续闭环

| 请求 | 成功 | 失败 | P50 | P95 | 实测平均吞吐 | 完整分钟成功数 |
|---:|---:|---:|---:|---:|---:|---|
| {soak['requests']} | {soak['successes']} | {soak['failures']} | {soak['p50_seconds']:.3f}s | {soak['p95_seconds_nearest_rank']:.3f}s | {soak['throughput_images_per_minute']:.3f}张/分 | {', '.join(map(str, full_minutes))} |

## 本地超分并发

每个目标先暖机一次，再每档实测3轮。成功必须满足 ncnn exit code=0、中间图可读、最终文件经Pillow验证为精确目标像素。表中“墙钟”是三轮中位数（括号为范围）；“单张”是真实请求延迟均值/P95，不是墙钟除以M。

| 目标 | M | 成功/总数 | 墙钟中位秒（范围） | 单张均值/P95秒 | 吞吐中位/保守 张分 | 聚合峰值显存MiB | 报错 | 稳定 |
|---|---:|---:|---:|---:|---:|---:|---|---|
{gpu_table}

显存由一个中央 Windows GPU 性能计数器采样器测得：同一时刻只汇总本轮精确 ncnn PID，再取时间序列峰值；没有把各进程不同时刻峰值相加。M=6聚合峰值11,393.602 MiB；M=8最高13,497.523 MiB。采样实际间隔约1.1–1.3秒，所有轮次无无效显存样本。`nvidia-smi`/NVML当前不可用，因此未使用NVML数据。

## 端到端稳定产能

系统吞吐取 API 保守持续吞吐与GPU三轮最小稳定吞吐的较小值。

| 交付 | API保守 张分 | GPU保守 张分 | 系统 张分 | 每小时 | 理论24小时 | 瓶颈 |
|---|---:|---:|---:|---:|---:|---|
{capacity_table}

“理论24小时”是实测稳定速率的数学外推，本次没有运行24小时耐久测试，不能等同于已验证日产量。实际生产还应扣除任务调度、磁盘归档、重试和维护窗口。

### 当前服务实现限制

上述端到端数字严格按本任务指定口径计算：API段与GPU段解耦成队列，系统产能取两段实测吞吐较小值。当前 `image_pipeline/service.py` 仍用全局 `_pipeline_lock` 包住整段 `generate_and_upscale`，会把API等待和GPU处理全部串行化，因此**现有单进程HTTP服务尚不能兑现11.82/11.452张每分**。要实现压测产能，需要把服务改为API并发上限8、GPU worker池并发6的两级队列；本次是压测任务，未擅自修改生产调度语义。

## 三档 provider usage

以下数值是渠道响应原样返回，每档2个成功样本；每张图片都经过Pillow验证。

| 样本 | quality | 真实像素 | 耗时秒 | input_tokens | output_tokens | total_tokens |
|---:|---|---:|---:|---:|---:|---:|
{token_table}

**口径存疑：** output_tokens仅59–64，主压测low成功样本也只在53–69之间，明显不符合官方image-output-token的数百到上万量级；这些值不能直接套用官方 `$30/1M` 或任何官方图像token单价计算成本，只能作为第三方渠道返回字段留证。

## 原始证据与安全

- API阶梯：`capacity-benchmarks/20260711T101109Z-api-97b81276`、`capacity-benchmarks/20260711T104052Z-api-72bbd983`
- API持续：`capacity-benchmarks/20260711T101807Z-api-soak-01f49986`
- GPU并发：`capacity-benchmarks/20260711T095446Z-gpu-8274eeb4`
- token样本：`capacity-benchmarks/20260711T102427Z-tokens-fad1069f`
- CSV：`api-capacity-benchmark-2026-07-11.csv`、`gpu-capacity-benchmark-2026-07-11.csv`、`image-token-samples-2026-07-11.csv`

结果只保存请求状态、时延、真实像素、文件字节、SHA-256、request id和usage白名单字段；未保存Authorization、Key、完整Base64、签名图片URL或完整响应体。

本次共实际发送207个API请求，190个返回了经Pillow验证的可用图片，17个失败（15个HTTP 503、2个connect_error）；失败全部进入统计。190张成功图片的SHA-256没有发现重复组。最终对19个文本结果文件执行环境Key精确匹配和`sk-*`模式扫描，命中均为0。
"""
    REPORT.write_text(report, encoding="utf-8")
    print(REPORT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
