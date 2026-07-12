from __future__ import annotations

import hashlib
import os
import re
from typing import Any

from fastapi import HTTPException
from litellm.integrations.custom_logger import CustomLogger


PUBLIC_ALIAS = os.getenv("IMAGE_PUBLIC_MODEL_ALIAS", "image-gen").strip()
PRIVATE_GROUP = os.getenv(
    "IMAGE_LITELLM_PRIVATE_GROUP", "image-pipeline-private"
).strip()


def _auth_field(auth: Any, name: str) -> Any:
    if isinstance(auth, dict):
        return auth.get(name)
    return getattr(auth, name, None)


def _tenant_scope(user_api_key_dict: Any) -> str:
    """Create a stable opaque scope without forwarding a virtual key."""
    # Idempotency is isolated per virtual key, even when several keys belong to
    # the same LiteLLM team. Only the hash is forwarded to the image backend.
    for name in ("api_key", "api_key_alias", "user_id", "team_id"):
        value = _auth_field(user_api_key_dict, name)
        if value:
            return hashlib.sha256(str(value).encode("utf-8")).hexdigest()[:32]
    raise HTTPException(status_code=401, detail="virtual key tenant identity is missing")


def _idempotency_key(data: dict[str, Any]) -> str | None:
    request = data.get("proxy_server_request")
    headers = request.get("headers") if isinstance(request, dict) else None
    if not isinstance(headers, dict):
        return None
    value = headers.get("idempotency-key") or headers.get("Idempotency-Key")
    if value is None:
        return None
    value = str(value)
    if not 8 <= len(value) <= 200 or not re.fullmatch(r"[\x21-\x7e]+", value):
        raise HTTPException(status_code=422, detail="invalid Idempotency-Key")
    return value


class ImageGatewayHook(CustomLogger):
    """Expose one alias, reject the private group, and inject trusted tenant scope."""

    async def async_pre_call_hook(
        self,
        user_api_key_dict: Any,
        cache: Any,
        data: dict[str, Any],
        call_type: str,
    ) -> dict[str, Any]:
        requested_model = str(data.get("model") or "")
        if requested_model in {PUBLIC_ALIAS, PRIVATE_GROUP} and call_type != "image_generation":
            raise HTTPException(
                status_code=403,
                detail="image model alias is only available on the image generation endpoint",
            )
        if requested_model == PRIVATE_GROUP:
            raise HTTPException(
                status_code=403,
                detail="private image model is not directly accessible",
            )
        if requested_model != PUBLIC_ALIAS:
            return data

        data["model"] = PRIVATE_GROUP
        data["user"] = _tenant_scope(user_api_key_dict)
        idempotency_key = _idempotency_key(data)
        if idempotency_key:
            extra_headers = dict(data.get("extra_headers") or {})
            extra_headers["Idempotency-Key"] = idempotency_key
            data["extra_headers"] = extra_headers
        return data


image_gateway_hook = ImageGatewayHook()
