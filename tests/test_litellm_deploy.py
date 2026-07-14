from __future__ import annotations

from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def test_litellm_config_exposes_alias_and_keeps_private_backend_internal() -> None:
    config = yaml.safe_load(
        (ROOT / "deploy" / "litellm" / "config.yaml").read_text(encoding="utf-8")
    )
    model = config["model_list"][0]

    assert model["model_name"] == "image-pipeline-private"
    assert model["litellm_params"]["model"] == "openai/image-pipeline-private"
    assert model["litellm_params"]["api_base"] == "http://127.0.0.1:8012/v1"
    assert model["litellm_params"]["api_key"] == (
        "os.environ/IMAGE_LITELLM_BACKEND_TOKEN"
    )
    assert model["litellm_params"]["timeout"] == 910
    assert model["model_info"]["mode"] == "image_generation"
    assert model["model_info"]["input_cost_per_image"] == (
        "os.environ/IMAGE_COST_USD_PER_IMAGE"
    )
    assert config["router_settings"]["model_group_alias"] == {
        "image-gen": "image-pipeline-private"
    }
    assert config["litellm_settings"]["callbacks"] == [
        "gateway.hook.image_gateway_hook"
    ]
    assert config["general_settings"]["store_model_in_db"] is False
    assert config["general_settings"]["disable_spend_logs"] is False
    assert "pass_through_endpoints" not in config["general_settings"]


def test_compose_pins_images_and_keeps_all_ports_on_loopback() -> None:
    compose = yaml.safe_load(
        (ROOT / "deploy" / "litellm" / "docker-compose.yml").read_text(
            encoding="utf-8"
        )
    )
    services = compose["services"]
    postgres = services["postgres"]
    proxy = services["litellm"]

    assert postgres["image"].startswith("postgres:16.14-alpine3.24@sha256:")
    assert postgres["platform"] == "linux/amd64"
    assert postgres["ports"] == ["127.0.0.1:55432:5432"]
    assert postgres["mem_limit"] == "256m"
    assert proxy["image"].startswith(
        "ghcr.io/berriai/litellm-database:v1.91.2@sha256:"
    )
    assert proxy["platform"] == "linux/amd64"
    assert proxy["network_mode"] == "host"
    assert "127.0.0.1" in proxy["command"]
    assert proxy["mem_limit"] == "1600m"
    assert proxy["stop_grace_period"] == "930s"
    assert "../../gateway:/app/gateway:ro" in proxy["volumes"]

    integration = yaml.safe_load(
        (ROOT / "deploy" / "litellm" / "docker-compose.integration.yml").read_text(
            encoding="utf-8"
        )
    )
    assert "host.docker.internal:host-gateway" in integration["services"]["litellm"][
        "extra_hosts"
    ]


def test_nginx_routes_only_developer_endpoints_to_litellm() -> None:
    nginx = (ROOT / "deploy" / "nginx" / "image2-gen.conf").read_text(
        encoding="utf-8"
    )

    for path in ("/v1/images/generations", "/v1/models", "/v1/stats"):
        assert f"location = {path}" in nginx
    assert "proxy_pass http://127.0.0.1:4000" in nginx
    assert "auth_request /_litellm_virtual_key_auth" in nginx
    assert "proxy_pass http://127.0.0.1:8012/internal/gateway/stats" in nginx
    assert "include /etc/nginx/snippets/image2-gen-backend-auth.conf" in nginx
    assert "location ^~ /internal/gateway/" in nginx
    assert "return 404" in nginx
    assert "proxy_pass http://127.0.0.1:8012" in nginx


def test_cloud_service_memory_and_shutdown_limits_remain_bounded() -> None:
    unit = (ROOT / "deploy" / "systemd" / "image2-gen.service").read_text(
        encoding="utf-8"
    )

    assert "MemoryHigh=450M" in unit
    assert "MemoryMax=500M" in unit
    assert "TimeoutStopSec=1020s" in unit

    gateway_unit = (
        ROOT / "deploy" / "systemd" / "image2-gen-gateway.service"
    ).read_text(encoding="utf-8")
    assert "Requires=docker.service" in gateway_unit
    assert "--env-file /etc/image2-gen/litellm.env" in gateway_unit
    assert "--force-recreate" in gateway_unit
    assert "stop -t 930" in gateway_unit
    assert "TimeoutStartSec=600s" in gateway_unit
    assert "TimeoutStopSec=950s" in gateway_unit


def test_deployment_templates_fail_closed_until_secrets_are_filled() -> None:
    backend_env = (ROOT / "deploy" / "systemd" / "image2-gen.env.example").read_text(
        encoding="utf-8"
    )
    gateway_env = (ROOT / "deploy" / "litellm" / "litellm.env.example").read_text(
        encoding="utf-8"
    )

    for name in (
        "IMAGE_API_KEY",
        "IMAGE_SERVICE_TOKEN",
        "IMAGE_UPSCALE_WORKER_TOKEN",
        "IMAGE_LITELLM_BACKEND_TOKEN",
    ):
        assert f"{name}=\n" in backend_env
    assert "IMAGE_REQUIRE_SERVICE_AUTH=true" in backend_env
    for name in (
        "POSTGRES_PASSWORD",
        "DATABASE_URL",
        "LITELLM_MASTER_KEY",
        "LITELLM_SALT_KEY",
        "IMAGE_LITELLM_BACKEND_TOKEN",
    ):
        assert f"{name}=\n" in gateway_env
