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

## 部署架构：云 A + 本机 B

正式体验不再让领导访问本机端口或临时隧道：

1. **云 A（公网常驻）**：提供中文网页/API、SQLite 队列、调用 `gpt-image-2` 生成原图，并保存最终 PNG。当前演示由 Nginx 在公网 80 端口接入，应用只监听服务器回环地址的 8012 端口。
2. **本机 B（RTX 4090）**：只运行 `python cli.py upscale-worker`，主动从云 A 领取原图，Real-ESRGAN 超分并将最终 PNG 回传。它不开放入站端口，也不需要渠道 API Key。
3. **成品交付**：云 A 对 worker 上传的 PNG 再用 Pillow 验证真实像素，只有精确 2048×2048 或 3840×2160 才标记完成并返回图片 URL。

本机 B 离线时，云端网页仍可打开、任务仍可排队，但会停在等待超分状态；本机恢复联网后 worker 会自动继续领取。领导体验期间，本机必须保持开机、Windows 用户已登录、网络正常且关闭自动睡眠。

超分端现在是拉取式 GPU worker 池，不再绑定某一台固定电脑。每个 GPU worker 只需持有同一个内部 worker 令牌和自己唯一的 `IMAGE_UPSCALE_WORKER_ID`，即可从任意 NAT 后网络主动连接云端领活。SQLite 原子领取保证同一任务只交给一个 worker；领取记录包含 worker 身份、随机 claim token、租约心跳和尝试次数。worker 正常运行会续租，处理失败会立即释放供其他 worker 重试，进程或机器失联则在租约到期后自动恢复。增加一台 GPU 机器并启动 worker 就会增加池容量，不需要让云服务器反向连接 GPU。

对外开发者流量使用 LiteLLM Proxy 作为独立网关：Nginx 将 OpenAI 兼容的生图请求转到只监听回环地址 `127.0.0.1:4000` 的 LiteLLM，再由 LiteLLM 调用 `127.0.0.1:8012` 上的私有流水线适配端点。LiteLLM 负责虚拟 Key、每 Key RPM/预算、SpendLogs 和模型 ACL；SQLite 图片任务账本继续独立保存，不写入 LiteLLM 核心表。客户只会看到公共别名 `image-gen`，看不到上游模型、渠道地址或 GPU worker。

## 对外开发者 API（LiteLLM）

正式客户接口是 OpenAI Images 兼容接口：

```http
POST /v1/images/generations
Authorization: Bearer <customer-virtual-key>
Idempotency-Key: <stable-client-request-id>
Content-Type: application/json

{"model":"image-gen","prompt":"雨后未来城市街景，无文字，无水印","n":2,"size":"2048x2048","response_format":"url"}
```

- `model` 只能是 `image-gen`；虚拟 Key 查询 `GET /v1/models` 也只看到该别名，直连 `image-pipeline-private` 返回 403。
- `n` 为 `1`–`5`。每张图独立进入云端任务账本、独立超分，全部完成后以 OpenAI 图片响应格式返回 URL 数组。
- `size` 支持 `2048x2048`（2K）或 `3840x2160`（4K）；最终文件仍由服务器 Pillow 复验真实像素。
- `Idempotency-Key` 建议每个客户逻辑请求固定使用；同一虚拟 Key 重放相同请求复用原 batch，不同请求复用同一 Key 返回 409。
- `GET /v1/stats` 使用同一个客户虚拟 Key。Nginx 通过内部 `auth_request` 调 LiteLLM `/v1/models` 验证 Key，验证通过后才访问私有 stats 端点；没有另写一套客户鉴权。
- 该接口后端最多等待 900 秒，LiteLLM 上游等待 910 秒，Nginx 读取/发送超时为 930 秒；外层必须比内层多留响应构造余量，客户端自身超时应大于 930 秒。领导网页仍使用下文的异步 `/v1/generate`、`/v1/result/{id}`，互不影响。

LiteLLM 生产部署固定为数据库镜像 `v1.91.2` 的不可漂移 digest，PostgreSQL 同样固定到 16.14/Alpine 3.24。配置见 [deploy/litellm](deploy/litellm)：

```bash
cd /opt/image2-gen/deploy/litellm
sudo install -o root -g root -m 0600 litellm.env.example /etc/image2-gen/litellm.env
# 编辑 /etc/image2-gen/litellm.env 并填写所有占位符；真实文件不得复制回仓库。
sudo systemctl enable --now image2-gen-gateway.service
sudo docker compose --env-file /etc/image2-gen/litellm.env -f docker-compose.yml ps
curl --fail http://127.0.0.1:4000/health/liveliness
```

安装 Nginx 时，还要从与 `IMAGE_LITELLM_BACKEND_TOKEN` 相同的值生成 root 所有、`0640` 权限的 `/etc/nginx/snippets/image2-gen-backend-auth.conf`；仓库只提供占位模板 [image2-gen-backend-auth.conf.example](deploy/nginx/image2-gen-backend-auth.conf.example)。不要把填好令牌的 snippet 放进 Git。

管理员只在服务器回环地址用 master key 创建客户虚拟 Key，例如：

```bash
curl --fail http://127.0.0.1:4000/key/generate \
  -H "Authorization: Bearer $LITELLM_MASTER_KEY" \
  -H 'Content-Type: application/json' \
  -d '{"models":["image-gen"],"rpm_limit":5,"max_budget":20,"duration":"30d","key_alias":"customer-placeholder"}'
```

每个客户单独生成 Key，分别设置 RPM 和预算；master key 不提供给客户。LiteLLM `input_cost_per_image` 使用美元口径，来自 `IMAGE_COST_USD_PER_IMAGE`，应填写对客户计量的每张价格而不是渠道人民币成本。真实 LiteLLM 1.91.2 集成已验证 `n=1` 记 0.01、`n=5` 记 0.05，金额随 `n` 线性增长。

小内存服务器上，LiteLLM 在本地稳定运行时实测约 700–750 MiB，但目标云主机首次 Prisma 迁移时，900 MiB 与 1200 MiB 限制均会触发 cgroup OOM；因此生产容器限制为 1600 MiB、内存预留为 768 MiB，PostgreSQL 限制为 256 MiB。云端 Python 服务仍独立保持 systemd `MemoryHigh=450M`、`MemoryMax=500M`。部署前必须确认主机至少有约 3.5 GiB 内存和可用 swap，并监控三个进程，不能只看 Python 服务。

2026-07-12 目标云真实验收已完成：两个临时虚拟 Key 并发提交 `n=1` 与 `n=2`，三张成品均从公网 URL 下载并由 Pillow 解码为 2048×2048 PNG；同一个幂等键在两个客户作用域内形成两条相互隔离的账本记录，SpendLogs 增量为 `0.01 + 0.02` USD。测试期间 Python 后端峰值 143.324 MiB、LiteLLM 峰值 1197.477 MiB、主机可用内存最低 1075.164 MiB，服务无重启、无 OOM。完整证据见 [弹性服务验收记录](docs/elastic-service-validation.md)。

客户虚拟 Key、提示词和图片 URL 不能走明文 HTTP。当前 IP+HTTP 配置仅保留给单人演示；签发正式客户 Key 前必须配置域名、可信 TLS 证书和 HTTP→HTTPS 强制跳转。8012、4000、55432 均只允许回环访问，防火墙不得对公网放行。

```bash
# 先把 Nginx server_name 改成已解析到该服务器的真实域名，再签证书。
sudo apt-get install -y certbot python3-certbot-nginx
sudo certbot --nginx --redirect -d api.example.com
sudo nginx -t
sudo systemctl reload nginx
curl --fail https://api.example.com/health
```

`api.example.com` 只是文档占位符。没有可验证域名和可信证书时，只能继续做内部验收，不能发放正式客户 Key。

## 环境变量

真实 Key/令牌只注入进程环境，绝不写进代码、日志、返回体或 Git。仓库中的 [.env.example](.env.example) 和 [云端 systemd 环境模板](deploy/systemd/image2-gen.env.example) 只有占位符。

云 A 使用：

- `IMAGE_API_KEY`（或备用名 `OPENAI_API_KEY`）与 `IMAGE_API_BASE_URL`：渠道凭据与占位 URL。
- `IMAGE_SERVICE_DATA`：SQLite、原图和成品目录；systemd 固定为 `/var/lib/image2-gen`。
- `IMAGE_GENERATION_MIN_WORKERS` / `IMAGE_GENERATION_MAX_WORKERS`：弹性出图 worker 的空闲下限与积压上限，默认 `1` / `3`；全局硬上限为 `8`。旧变量 `IMAGE_GENERATION_WORKERS` 只作为 max 的兼容回退。
- `IMAGE_GENERATION_IDLE_RETIRE_SECONDS`：扩容 worker 的空闲回落时间，默认 `30` 秒。
- `IMAGE_GENERATION_SHUTDOWN_TIMEOUT_SECONDS`：优雅停机 drain 时间，默认 `990` 秒；它覆盖最坏情况下 3 次“20 秒连接 + 300 秒读取”及退避，配合 systemd 1020 秒停止窗口，避免进行中的计费请求被强杀。
- `IMAGE_FIXED_QUALITY`：当前服务统一生成质量，默认 `low`。
- `IMAGE_API_CONNECT_TIMEOUT_SECONDS`：渠道连接超时，默认 `20` 秒。
- `IMAGE_API_TIMEOUT_SECONDS`：渠道读取超时，默认 `300` 秒。
- `IMAGE_API_MAX_ATTEMPTS`：503、连接错误或超时的最大总尝试次数，默认 `3`，允许 `1`–`3`；上限与停机 drain 的最坏时间预算绑定。
- `IMAGE_API_COST_CNY_FIXED`：异步服务使用的已核验渠道打包单价；只从环境变量读取。
- `PUBLIC_BASE_URL`：领导打开的云端公网基址，不带末尾 `/`；当前格式为 `http://<云服务器公网IP>`。
- `IMAGE_SERVICE_TOKEN`：领导网页/API 的可选 Bearer 令牌。
- `IMAGE_REQUIRE_SERVICE_AUTH`：生产模板固定为 `true`；未填 `IMAGE_SERVICE_TOKEN` 时领导提交/查询接口返回 503，而不是无鉴权放行。本机开发若明确需要无令牌模式才设为 `false`。
- `IMAGE_UPSCALE_WORKER_TOKEN`：云端与本机共享的 worker 专用令牌，至少 32 位，且必须与 `IMAGE_SERVICE_TOKEN` 不同。
- `IMAGE_LITELLM_BACKEND_TOKEN`：LiteLLM 到私有流水线适配端点的第三个独立令牌，至少 32 位，必须与领导令牌、worker 令牌均不同。
- `IMAGE_LITELLM_PRIVATE_MODEL` / `IMAGE_PUBLIC_MODEL_ALIAS`：当前部署契约固定为 `image-pipeline-private` / `image-gen`，前者永不暴露给客户。不要只改其中一个；如确需改名，必须同时修改 LiteLLM 配置、hook 和后端适配器并重新跑隔离测试。
- `IMAGE_LITELLM_SYNC_TIMEOUT_SECONDS`：开发者同步图片接口等待终态的秒数，部署默认 `900`。
- `IMAGE_UPSCALE_LEASE_SECONDS`：worker 任务租约，默认 `600` 秒。
- `IMAGE_UPSCALE_MAX_ATTEMPTS`：超分失败后的最大领取次数，默认 `3`，允许 `1`–`10`；超过后任务进入终态失败，避免确定性坏图无限重试。
- `IMAGE_UPSCALE_MAX_UPLOAD_BYTES`：worker 最终 PNG 上传上限；代码默认 25 MiB，部署模板为 50 MiB。

本机 B 使用：

- `IMAGE_CLOUD_BASE_URL`：本机 worker 可访问的云 A HTTP(S) 基址。
- `IMAGE_UPSCALE_WORKER_TOKEN`：与云 A 相同的 worker 专用令牌。
- `IMAGE_UPSCALE_WORKER_ID`：可选 worker 名，仅允许字母、数字、点、下划线和连字符；默认电脑名。
- `IMAGE_UPSCALE_WORKER_CONCURRENCY`：本机并行超分槽数，默认 `3`，允许 `1`–`5`；每个槽独立领取、下载、超分和回传。
- `IMAGE_UPSCALE_POLL_SECONDS`：无任务时轮询间隔，默认 `4` 秒。
- `IMAGE_UPSCALE_WORK_ROOT`：本机临时工作目录，默认 `remote-worker-data`。
- `REALESRGAN_TILE_SIZE`：Real-ESRGAN 切块尺寸，默认 `0`（整张一次超分、不切块）；显存较小的设备可改为正整数。RTX 4090 的实测推荐见“超分引擎”。

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

## 异步网页/API

云 A 的 systemd 服务监听 `127.0.0.1:8012`，由同机 Nginx 暴露公网 80 端口。开发机临时调试可运行：

```powershell
python cli.py serve --host 127.0.0.1 --port 8012
```

网页可选择一次生成 `1`–`5` 张同提示词变体，以及 `1`–`5` 的出图并发档位；前后端都会拒绝超出上限的值。提交后立即得到一组任务 ID，并每隔数秒分别查询状态。云端生成原图后，每张图独立等待本机 B 超分；完成一张就显示一张大图和对应下载按钮，单张失败也不会遮住同批次的其他结果。

```http
POST /v1/generate
Content-Type: application/json

{"prompt":"雨后未来城市街景，无文字，无水印","size":"4k","count":5,"concurrency":5}
```

`count` 和 `concurrency` 均为可选字段：`count` 默认 `1`，表示同一提示词生成多少个不同变体；`concurrency` 默认 `3`，表示该批次希望同时发起多少个单图请求。后端先尝试以渠道原生 `n=count` 一次生成多张；渠道明确拒绝 `n>1`，或 HTTP 200 实际返回张数少于 `count` 时，才回退为最多 `concurrency` 路并发单图请求。若原生调用已经返回部分图片，会保留这些已付费结果，只为缺少的张数发起单图请求；进程内还会记住该渠道不支持原生 batch，后续 batch 直接走单图并发。网络超时不会被误判成“不支持 n”并盲目切换请求形态。单批张数与网页档位仍硬限制为 `5`；多个 batch 的出图 worker 会按积压从 `IMAGE_GENERATION_MIN_WORKERS` 自动扩到 `IMAGE_GENERATION_MAX_WORKERS`，全局代码硬上限为 `8`。

`POST /v1/generate` 返回 HTTP 202、`batch_id`、`task_ids`、`result_urls` 和 `batch_result_url`。逐张查询 `GET /v1/result/{task_id}`，或用 `GET /v1/batch/{batch_id}` 一次查看整批摘要与结果；每张完成时都有独立的 `image_url`。当 `count=1` 时，响应仍额外保留原有的 `task_id` 和 `result_url`，旧客户端无需修改。若设置了 `IMAGE_SERVICE_TOKEN`，API 请求须带 `Authorization: Bearer <token>`，网页也会带上用户填写的令牌。

健康检查：`GET /health`。

渠道调用只会对 HTTP 503、连接错误和超时做有限指数退避重试，默认最多三次；同一逻辑请求在全部尝试中复用同一个 `Idempotency-Key`。上游幂等键在任务创建时即持久化到 SQLite，服务进程重启后仍复用原值，而不是生成新 Key 重发。这能在渠道支持幂等语义时避免超时或重启后的重复生成/计费；若渠道不保证幂等，可将 `IMAGE_API_MAX_ATTEMPTS=1` 关闭自动重发。图片 URL 下载失败不会重新调用出图 API。

### 领导如何访问

领导只需在任意设备打开 `http://<云服务器公网IP>/`；不安装 Python、Real-ESRGAN、VPN 或内网穿透。若启用了 `IMAGE_SERVICE_TOKEN`，在网页令牌框填写领导令牌。不要通过聊天、截图或 URL 查询参数传递令牌。

演示前按真实用户路径验证一次：网页提交 → 等待完成 → 浏览器显示图片 → 下载 PNG → Pillow 解码真实像素。2K 必须为 2048×2048，4K 必须为 3840×2160；不能用 HTTP 状态或 JSON `size` 字段代替验图。

## 云 A：systemd 常驻部署

以下约定适用于 Ubuntu/Debian：代码在 `/opt/image2-gen`，虚拟环境在 `/opt/image2-gen/.venv`，运行用户为 `image2gen`，数据在 `/var/lib/image2-gen`，真实环境文件只存在服务器 `/etc/image2-gen/image2-gen.env`。

```bash
sudo useradd --system --home /var/lib/image2-gen --shell /usr/sbin/nologin image2gen
sudo install -d -o image2gen -g image2gen -m 0750 /var/lib/image2-gen
sudo install -d -o root -g image2gen -m 0750 /etc/image2-gen
cd /opt/image2-gen
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
sudo install -o root -g image2gen -m 0640 deploy/systemd/image2-gen.env.example /etc/image2-gen/image2-gen.env
```

编辑服务器上的环境文件，把所有秘密占位符替换为真实值；不要把填好的文件复制回仓库。`IMAGE_SERVICE_TOKEN` 与 `IMAGE_UPSCALE_WORKER_TOKEN` 必须不同。然后安装应用、Nginx 和 7 天清理 timer：

```bash
sudo install -o root -g root -m 0644 deploy/systemd/image2-gen.service /etc/systemd/system/image2-gen.service
sudo install -o root -g root -m 0644 deploy/systemd/image2-gen-gateway.service /etc/systemd/system/image2-gen-gateway.service
sudo install -o root -g root -m 0644 deploy/systemd/image2-gen-cleanup.service /etc/systemd/system/image2-gen-cleanup.service
sudo install -o root -g root -m 0644 deploy/systemd/image2-gen-cleanup.timer /etc/systemd/system/image2-gen-cleanup.timer
sudo install -o root -g root -m 0600 deploy/litellm/litellm.env.example /etc/image2-gen/litellm.env
# 编辑 /etc/image2-gen/litellm.env；其中 IMAGE_LITELLM_BACKEND_TOKEN 必须与
# /etc/image2-gen/image2-gen.env 中的同名值一致，其余 secret 必须各不相同。
sudo apt-get install -y nginx
docker --version
docker compose version
sudo install -o root -g root -m 0640 deploy/nginx/image2-gen-backend-auth.conf.example /etc/nginx/snippets/image2-gen-backend-auth.conf
# 编辑上述 snippet，把占位符替换成同一个 IMAGE_LITELLM_BACKEND_TOKEN。
sudo install -o root -g root -m 0644 deploy/nginx/image2-gen.conf /etc/nginx/sites-available/image2-gen
# 将配置中的 your-cloud-public-ip 替换为实际公网 IP 或域名。
sudo rm -f /etc/nginx/sites-enabled/default
sudo ln -sfn /etc/nginx/sites-available/image2-gen /etc/nginx/sites-enabled/image2-gen
sudo nginx -t
sudo systemctl daemon-reload
sudo systemctl enable --now image2-gen.service image2-gen-gateway.service image2-gen-cleanup.timer nginx.service
sudo systemctl status image2-gen.service image2-gen-gateway.service --no-pager
curl --fail http://127.0.0.1:8012/health
curl --fail http://127.0.0.1:4000/health/liveliness
sudo ss -lnt | awk '$4 ~ /:(8012|4000|55432)$/ {print $4}'
sudo ufw allow 80/tcp
sudo ufw status
```

检查 `/health` 返回的 `api_key_configured`、`service_auth_enabled`、`service_auth_required`、`upscale_worker_auth_configured` 都为 `true`；否则不要开放公网端口，先修正服务器环境变量。仓库模板中的秘密值故意留空，复制后未填写会失败关闭；形如 `<set-with-...>` 的占位令牌也会被代码识别为未配置。健康检查响应不会返回任何令牌内容。上面的三个私有端口必须都只显示 `127.0.0.1` 或 `[::1]`；不要运行会展开私有 Bearer snippet 的 `nginx -T`，配置校验只使用 `nginx -t`。

systemd 服务退出或崩溃后 5 秒自动重启，并设置 `MemoryHigh=450M`（软限速）、`MemoryMax=500M`（硬上限），避免并发出图在小内存云主机上挤占系统。图片下载、落盘和 worker 上传都按块流式处理，不把一批大图同时保存在内存。清理 timer 每天约 03:30 运行，删除超过 7 天且状态为 `done`/`failed` 的数据库记录、原图和成品图；正在排队、生成或超分的任务不会删除。可用以下命令检查：

清理按整个 batch 的最终完成时间执行：只有同批所有任务都已终态且最新一张也超过保留期时才整批删除，避免破坏幂等重放的张数完整性。`systemctl reload image2-gen-gateway` 会强制重建 LiteLLM 容器，确保 bind mount 的 hook/config 真正重新加载；停止网关容器时保留 930 秒宽限，后端服务保留 1020 秒，不能使用 Docker 默认约 10 秒的停止窗口截断同步请求。

```bash
systemctl list-timers image2-gen-cleanup.timer
sudo systemctl start image2-gen-cleanup.service
sudo journalctl -u image2-gen.service -n 100 --no-pager
sudo journalctl -u image2-gen-cleanup.service -n 50 --no-pager
systemctl show image2-gen -p MemoryCurrent -p MemoryHigh -p MemoryMax -p NRestarts
```

还必须在云厂商安全组中放行入站 TCP 80；能固定领导出口 IP 时优先只允许该 IP，否则仅在演示期临时开放并在结束后收回。应用的 8012 端口不对公网开放。配置完成后，从另一台电脑访问 `http://<云服务器公网IP>/health` 和网页，不能只在服务器本机 curl。

当前公网直连是为了尽快完成单人体验，HTTP 不加密，Bearer 令牌会以明文经过网络；演示后应轮换领导令牌和 worker 令牌。生产升级时应为 Nginx 配置正式域名与 HTTPS。当前反向代理请求体上限为 64 MiB，worker 上传超时为 300 秒，公网不开放应用的 8012 端口。

### 2026-07-11 公网端到端验收

验收使用真实 Chrome，从公网首页填写领导令牌和提示词，依次选择 2K、4K，等待网页自动轮询到完成，看到大图后点击“下载 PNG”。以下像素均来自对浏览器实际下载文件的 Pillow 解码，不采信 HTTP 状态或 JSON `size`：

| 尺寸 | 浏览器下载文件 | Pillow 实际像素 | 全链路 | 原图云→本机 | 本机超分 | 后处理 | 远端阶段合计 | 浏览器下载 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2K | 4,071,572 bytes | 2048×2048 PNG | 164.426s | 6.945s | 12.112s | 1.775s | 24.082s | 4.897s |
| 4K | 8,672,287 bytes | 3840×2160 PNG | 72.267s | 4.886s | 10.291s | 2.950s | 21.030s | 3.468s |

两张下载文件的字节数与 SHA-256 均与云端复验记录一致。2K 这次渠道生成耗时 137.775 秒，说明“约 90 秒”只是常见值而不是承诺；前端会持续显示等待时间，不会白屏或因超过 90 秒自动失败。4K 经公网回传与下载均未出现 502。

### 2026-07-11：一次 5 张 4K 并发验收

从公网网页一次提交 `count=5`、`concurrency=5`。本次渠道明确不支持原生 `n=5`，服务自动切换为 5 路单图 fallback；5 次渠道调用各耗时约 39.4–44.9 秒，时间区间高度重叠，确认不是串行排队。最慢一张从提交到完成为 92.836 秒；作为量级参考，上表单张 4K 验收为 72.267 秒，但渠道时延会波动，二者不是严格同条件基准。若这 5 次渠道调用串行，仅 API 阶段就至少需要约 197 秒。

5 张成品全部成功，服务器复验和客户端下载均为 `3840×2160 PNG`。客户端下载 5 张图片的并行墙钟时间为 4.545 秒；每张下载文件的 SHA-256 均与云端记录一致。浏览器最终显示 5 张独立大图卡片和 5 个下载按钮。

同一轮测试对云端连续采样 740 秒、共 668 个内存样本：应用 `MemoryCurrent` 峰值为 315.195 MiB，系统可用内存最低为 1084.84 MiB，`NRestarts` 保持 `0 → 0`，未触发 `MemoryHigh=450M`、`MemoryMax=500M` 或 OOM。由此看，当前首先受限的是渠道生成耗时和本机超分/上行，而不是云端内存；仍需保留 500M 硬上限，不能据一次测试继续抬高并发上限。

## 本机 B：Windows 超分 worker 常驻

本机只需要云端地址和 worker 令牌，不设置 `IMAGE_API_KEY`。先把非秘密配置写入当前 Windows 用户环境；令牌用遮蔽输入写入，避免出现在 PowerShell 历史：

```powershell
[Environment]::SetEnvironmentVariable("IMAGE_CLOUD_BASE_URL", "http://your-cloud-public-ip", "User")
[Environment]::SetEnvironmentVariable("IMAGE_UPSCALE_POLL_SECONDS", "4", "User")
[Environment]::SetEnvironmentVariable("IMAGE_UPSCALE_WORKER_CONCURRENCY", "3", "User")
[Environment]::SetEnvironmentVariable("REALESRGAN_TILE_SIZE", "0", "User")

$workerToken = [PSCredential]::new("unused", (Read-Host "IMAGE_UPSCALE_WORKER_TOKEN" -AsSecureString)).GetNetworkCredential().Password
[Environment]::SetEnvironmentVariable("IMAGE_UPSCALE_WORKER_TOKEN", $workerToken, "User")
Remove-Variable workerToken
```

注销并重新登录一次，让任务计划程序获得新的用户环境。然后在仓库根目录安装隐藏任务：

```powershell
# 若以前装过本机 API 常驻任务，先移除；云 A 已接管网页/API。
powershell -ExecutionPolicy Bypass -File .\scripts\Uninstall-ImageServiceTask.ps1
powershell -ExecutionPolicy Bypass -File .\scripts\Install-UpscaleWorkerTask.ps1
```

脚本优先使用 `.venv\Scripts\python.exe`，否则使用 `PATH` 中的 `python.exe`。也可显式指定：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\Install-UpscaleWorkerTask.ps1 -PythonPath "C:\path\to\python.exe"
```

`Start-UpscaleWorker.ps1` 在 worker 异常退出后按 5、10、20、40、60 秒退避重启，Task Scheduler 再做每分钟兜底。任务在当前用户登录时自动启动并隐藏运行，关闭终端不会停止。常用运维命令：

```powershell
Get-ScheduledTask -TaskName Image2GenUpscaleWorker
Get-ScheduledTaskInfo -TaskName Image2GenUpscaleWorker
Stop-ScheduledTask -TaskName Image2GenUpscaleWorker
Start-ScheduledTask -TaskName Image2GenUpscaleWorker
Get-Content .\remote-worker-data\logs\launcher.log -Tail 50
Get-Content .\remote-worker-data\logs\worker.log -Tail 100
powershell -ExecutionPolicy Bypass -File .\scripts\Uninstall-UpscaleWorkerTask.ps1
```

领导体验期间必须保证本机开机、该 Windows 用户已登录、网络畅通、GPU 驱动正常且系统不自动睡眠。worker 只发起出站 HTTP(S)，不需要路由器端口转发或 Windows 入站防火墙规则。

扩容时，在新 GPU 机器上安装同一版本代码和 Real-ESRGAN，配置相同的云端地址与 worker 令牌，并给每台机器设置唯一 worker ID。单机优先用 `IMAGE_UPSCALE_WORKER_CONCURRENCY` 控制槽数；若确实要在同一机器启动多个独立进程，每个进程也必须使用不同 worker ID。云端可通过受 worker 令牌保护的 `GET /internal/upscale/workers` 查看最近活跃的 worker、状态、当前任务和完成计数。阶段一的双进程真实 GPU 验收记录与复现命令见 [弹性服务验证记录](docs/elastic-service-validation.md)。

## 档位与成本

档位严格映射到 API 的同名 quality：low→low、medium→medium、high→high；三档
使用同一 `realesrgan-x4plus` 超分模型。

当前渠道为按张打包价，不按尺寸、quality 或 token 区分；已核验 API 费约为 ¥0.0375/张，
完整交付实测约 ¥0.0566/张。异步服务用 `IMAGE_API_COST_CNY_FIXED` 注入已核验打包单价，
不会根据渠道返回的可疑 token 字段推算价格。

旧版 CLI manifest 仍支持以下分档环境变量；使用当前打包价渠道时可将三项设为同一值：

- `IMAGE_API_COST_CNY_LOW`
- `IMAGE_API_COST_CNY_MEDIUM`
- `IMAGE_API_COST_CNY_HIGH`

本地超分电费是估算值，默认按 175W 和 0.60 元/kWh 计算，通常远小于 0.01 元；
这不能替代渠道 API 账单。

## 超分引擎

项目包含 Real-ESRGAN 官方 Windows NCNN/Vulkan 发行包。来源、版本、哈希与许可证
见 `tools/realesrgan-ncnn-vulkan/SOURCE.md`。默认 `-g 0` 强制选择 RTX 4090，`REALESRGAN_TILE_SIZE=0`，即整张图片一次超分。旧默认 `tile=256` 会把原图拆成很多小块分别处理，在天空、墙面等平滑区域形成可见网格接缝；显存足够时不应继续使用小 tile。

同一张真实 1536×1024 原图输出 4K 的对照如下。接缝比值为“预期切块边界上的平均梯度 ÷ 相邻控制线平均梯度”；越接近 `1`，越没有系统性的网格边界：

| 配置 | Real-ESRGAN 超分 | 含后处理总耗时 | GPU 专用显存增量 | 同位置接缝比值 |
| --- | ---: | ---: | ---: | ---: |
| `tile=256`，并发 1 | 12.380s | 13.785s | +1838 MiB | 1.210850 |
| `tile=0`，并发 1 | 11.818s | 13.183s | +1246 MiB | 1.025973 |

`tile=0` 的网格特征基本消失，而且本次单张超分更快、显存增量更低；因此无需为了性能或显存退回 `tile=512`。如果换成显存较小的 GPU，再通过环境变量按实测设置正整数 tile。

RTX 4090 16GB 上继续以同一张图做 `tile=0` 并发阶梯测试：

| 超分并发 | 每张 Real-ESRGAN 耗时（约） | 整批墙钟 | GPU Adapter Memory 峰值 |
| ---: | ---: | ---: | ---: |
| 1 | 11.818s | 13.183s | 单任务增量 +1246 MiB |
| 2 | 20.23s | 24.056s | 6352 MiB |
| 3 | 28.1s | 31.277s | 7672 MiB |
| 4 | 36.6s | 39.655s | 8992 MiB |
| 5 | 45.2s | 49.291s | 10313 MiB |

16GB 显存在 `tile=0 + 并发 5` 下仍未耗尽，但 C4/C5 的总吞吐提升已经很小，单张延迟却明显增加。真实 5×4K 链路中的 3 路 worker 采样为：GPU Adapter Memory 基线 3647.5 MiB、峰值 7631.5 MiB、增量 3984.0 MiB，并同时观察到 3 个 Real-ESRGAN 进程。

默认推荐 **`tile=0 + IMAGE_UPSCALE_WORKER_CONCURRENCY=3`**：优先消除接缝，同时给 16GB 显存、本机桌面程序和上传留余量。吞吐优先可试并发 `4`，但不建议常态设为 `5`。当前全链路最大的绝对耗时仍是渠道出图；本机侧首先出现的是 GPU 计算争用导致的吞吐趋平，而不是显存耗尽。5 张约 10MB 的 4K 成品同时回传还可能使本机上行成为下一瓶颈，因此提高 worker 数前应同时观察单张延迟和上传耗时，而不能只看空余显存。

本机 NVML 当前不可用，因此峰值显存来自 Windows
`GPU Process Memory/Dedicated Usage` 性能计数器，而不是 `nvidia-smi`。
