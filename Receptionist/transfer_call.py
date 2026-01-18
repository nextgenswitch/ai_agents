"""Utilities for transferring an active call to a live agent."""

import asyncio
from datetime import datetime
from typing import Optional

import requests
from loguru import logger


XML_TEMPLATE = """<?xml version="1.0"?>\n<response>\n    <dial>{number}</dial>\n</response>"""


async def transfer_call(call_sid: str, dial_number: str, base_url: str, api_key: str, api_secret: str, transfer_delay: float = 5.0, timeout: int = 10) -> None:
    """Transfer an active call to the supplied number."""

    if not base_url:
        logger.error("NEXTGENSWITCH_URL is not configured; unable to transfer call {}", call_sid)
        return

    if transfer_delay:
        logger.debug("Waiting {} seconds before transferring call {}", transfer_delay, call_sid)
        await asyncio.sleep(transfer_delay)

    url = f"{base_url.rstrip('/')}/{call_sid}"
    headers = {}
    if api_key:
        headers["X-Authorization"] = api_key
    if api_secret:
        headers["X-Authorization-Secret"] = api_secret

    payload = {"responseXml": XML_TEMPLATE.format(number=dial_number)}
    logger.info("Transferring call {} to {} via {}", call_sid, dial_number, url)

    try:
        response = await asyncio.to_thread(
            requests.put,
            url,
            headers=headers,
            data=payload,
            timeout=timeout,
        )
    except Exception as exc:
        logger.exception("Failed to transfer call {}: {}", call_sid, exc)
        return

    timestamp = datetime.now().strftime("%H:%M:%S")
    if response.ok:
        logger.info(
            "Call {} transferred successfully (status {} at {})",
            call_sid,
            response.status_code,
            timestamp,
        )
    else:
        logger.error(
            "Call {} transfer failed (status {} at {}): {}",
            call_sid,
            response.status_code,
            timestamp,
            response.text,
        )
