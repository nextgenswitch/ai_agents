"""Utilities for creating support tickets in NextGenSwitch."""

from __future__ import annotations

import asyncio
import os
from datetime import datetime, timezone
from typing import Optional

import requests
from loguru import logger


def _env(key: str, default: Optional[str] = None) -> Optional[str]:
    value = os.getenv(key)
    if value is None or value == "":
        return default
    return value


async def create_ticket(
    call_sid: str,
    subject: str,
    description: str,
    name: Optional[str] = None,
    email: Optional[str] = None,
    phone: Optional[str] = None,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    api_secret: Optional[str] = None,
    timeout: Optional[float] = None,
) -> bool:
    """Create a support ticket for a call in NextGenSwitch."""

    base_url = base_url or _env("NEXTGENSWITCH_API_URL") or _env("NEXTGENSWITCH_URL")
    api_key = api_key or _env("NEXTGENSWITCH_KEY") or _env("NEXTGENSWITCH_API_KEY")
    api_secret = api_secret or _env("NEXTGENSWITCH_SECRET") or _env("NEXTGENSWITCH_API_SECRET")
    timeout_value = timeout if timeout is not None else float(_env("NEXTGENSWITCH_TIMEOUT", "10"))

    if not base_url:
        logger.error(
            "create_ticket: missing NEXTGENSWITCH_API_URL; cannot create ticket (call_sid={call_sid})",
            call_sid=call_sid,
        )
        return False

    if not api_key or not api_secret:
        logger.warning(
            "create_ticket: missing auth header(s); request may fail "
            "(call_sid={call_sid}, has_key={has_key}, has_secret={has_secret})",
            call_sid=call_sid,
            has_key=bool(api_key),
            has_secret=bool(api_secret),
        )

    url = f"{base_url.rstrip('/')}/support_tickets"

    headers = {}
    if api_key:
        headers["X-Authorization"] = api_key
    if api_secret:
        headers["X-Authorization-Secret"] = api_secret

    payload = {"call_id": call_sid, "subject": subject, "description": description}
    if name:
        payload["name"] = name
    if email:
        payload["email"] = email
    if phone:
        payload["phone"] = phone

    logger.info(
        "create_ticket: sending POST {url} (call_sid={call_sid}, timeout={timeout})",
        url=url,
        call_sid=call_sid,
        timeout=timeout_value,
    )

    try:
        response = await asyncio.to_thread(
            requests.post,
            url,
            headers=headers,
            json=payload,
            timeout=timeout_value,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception(
            "create_ticket: request failed (call_sid={call_sid}, error={error})",
            call_sid=call_sid,
            error=str(exc),
        )
        return False

    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    logger.debug(
        "create_ticket: received response (call_sid={call_sid}, status_code={status}, at={ts}, body_preview={preview})",
        call_sid=call_sid,
        status=response.status_code,
        ts=timestamp,
        preview=response.text[:500],
    )

    if response.ok:
        logger.info(
            "create_ticket: success (call_sid={call_sid}, status_code={status}, at={ts})",
            call_sid=call_sid,
            status=response.status_code,
            ts=timestamp,
        )
        return True

    logger.warning(
        "create_ticket: non-2xx response (call_sid={call_sid}, status_code={status}, at={ts})",
        call_sid=call_sid,
        status=response.status_code,
        ts=timestamp,
    )
    return False
