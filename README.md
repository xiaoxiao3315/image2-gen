# gpt-image-2 + 本地 Real-ESRGAN 流水线

这套服务固定使用 `gpt-image-2` 出原图，再用 RTX 4090 Laptop GPU 上的
Real-ESRGAN NCNN/Vulkan x4 超分，最后以 `cover + center crop + Lanczos`
交付精确的 2K（2048×2048）或 4K（3840×2160）PNG。

所有尺寸都在文件落盘后由 Pillow 解码复验；manifest 保存真实像素、文件字节数、
SHA-256、API/下载/超分/后处理耗时、GPU 设备和峰值显存采样。

> ⚠️ 分辨率说明：`gpt-image-2` 原生输出约 1536×1024，2K/4K 由 Real-ESRGAN
> 超分得到，新增细节为放大模型推断，不属于原生生成。

## 依赖

Real-ESRGAN 二进制未包含在仓库中，请自行下载并放到 `tools/realesrgan-ncnn-vulkan/`：
- 下载：https://github.com/xinntao/Real-ESRGAN/releases （`realesrgan-ncnn-vulkan` 对应平台包）
- 解压后确保 `tools/realesrgan-ncnn-vulkan/realesrgan-ncnn-vulkan(.exe)` 可执行。

## 环境变量

不要把 Key 写进命令、代码、`.env` 或配置文件：

```powershell
$env:IMAGE_API_KEY = Read-Host "IMAGE_API_KEY" -MaskInput
$env:IMAGE_API_BASE_URL = "https://your-image-api-endpoint/v1"
$env:IMAGE_API_PROXY = ""   # 可选：需要代理时填 http://host:port
```

也支持从进程环境读取 `OPENAI_API_KEY` 作为备用名称。

## CLI

```powershell
python cli.py generate --prompt "雨后未来城市的自然光街景，无文字，无水印" --tier low --target 4k
python cli.py generate --prompt "产品棚拍，无文字，无水印" --tier medium --target 2k
python cli.py upscale --input .\source.png --target 4k --output-dir .\offline-output
python cli.py upscale-batch --input-dir .\sources --target 4k --output-dir .\batch-upscaled
python cli.py batch --input .\prompts.jsonl --output .\batch-results.jsonl --default-tier low --default-target 4k
python cli.py cost-table
```

最终文件位于每次运行的 `runs/<run-id>/final-WIDTHxHEIGHT.png`，同目录的
`manifest.json` 是验收证据。

## 本地 HTTP 服务

```powershell
python cli.py serve --host 127.0.0.1 --port 8012
```

```http
POST /v1/generate
Content-Type: application/json

{"prompt":"雨后未来城市街景，无文字，无水印","tier":"low","target":"4k"}
```

批量接口 `POST /v1/generate/batch` 接收 `{"items":[...最多32项...]}`。CLI 和 HTTP
批量任务都串行占用 GPU，避免 16GB 显存被并发任务挤爆；每一项仍有独立 run 目录和 manifest。

健康检查：`GET /health`。MVP 接口为同步调用，返回完整 manifest；调用方超时应大于
API 生成与本地超分总耗时。

## 档位与成本

档位严格映射到 API 的同名 quality：low→low、medium→medium、high→high；三档
使用同一 `realesrgan-x4plus` 超分模型。

渠道 API 单价只有在账单或采购报价核验后才能视为确定。可用以下环境变量注入已核验
人民币单价，manifest 才会计算总成本和达标状态：

- `IMAGE_API_COST_CNY_LOW`
- `IMAGE_API_COST_CNY_MEDIUM`
- `IMAGE_API_COST_CNY_HIGH`

本地超分电费是估算值，默认按 175W 和 0.60 元/kWh 计算，通常远小于 0.01 元；
这不能替代渠道 API 账单。

## 超分引擎

项目包含 Real-ESRGAN 官方 Windows NCNN/Vulkan 发行包。来源、版本、哈希与许可证
见 `tools/realesrgan-ncnn-vulkan/SOURCE.md`。默认 `-g 0` 强制选择 RTX 4090，tile=256。
本机 NVML 当前不可用，因此峰值显存来自 Windows
`GPU Process Memory/Dedicated Usage` 性能计数器，而不是 `nvidia-smi`。
