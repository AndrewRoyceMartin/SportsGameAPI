from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Optional

import requests


@dataclass
class ConnStatus:
    name: str
    ok: bool
    latency_ms: Optional[int] = None
    detail: str = ""


def check_sofascore(timeout: float = 8.0) -> ConnStatus:
    url = "https://api.sofascore.com/api/v1/sport/football/scheduled-events/2025-01-01"
    headers = {"User-Agent": "Mozilla/5.0 SportsGameAPI/1.0"}
    t0 = time.time()
    try:
        r = requests.get(url, headers=headers, timeout=timeout)
        ms = int((time.time() - t0) * 1000)
        if r.status_code == 200:
            return ConnStatus("SofaScore", True, ms, "OK")
        return ConnStatus("SofaScore", False, ms, f"HTTP {r.status_code}")
    except requests.Timeout:
        return ConnStatus("SofaScore", False, detail="Timeout")
    except requests.ConnectionError:
        return ConnStatus("SofaScore", False, detail="Connection failed")
    except Exception as e:
        return ConnStatus("SofaScore", False, detail=str(e)[:80])


def check_apify(timeout: float = 8.0) -> ConnStatus:
    token = os.getenv("APIFY_TOKEN")
    if not token:
        return ConnStatus("Apify", False, detail="APIFY_TOKEN not set")
    url = "https://api.apify.com/v2/users/me"
    headers = {"Authorization": f"Bearer {token}"}
    t0 = time.time()
    try:
        r = requests.get(url, headers=headers, timeout=timeout)
        ms = int((time.time() - t0) * 1000)
        if r.status_code == 200:
            data = r.json().get("data", {})
            username = data.get("username", "")
            return ConnStatus("Apify", True, ms, f"Authenticated as {username}" if username else "OK")
        if r.status_code in (401, 403):
            return ConnStatus("Apify", False, ms, "Invalid token")
        return ConnStatus("Apify", False, ms, f"HTTP {r.status_code}")
    except requests.Timeout:
        return ConnStatus("Apify", False, detail="Timeout")
    except requests.ConnectionError:
        return ConnStatus("Apify", False, detail="Connection failed")
    except Exception as e:
        return ConnStatus("Apify", False, detail=str(e)[:80])


def check_all() -> list[ConnStatus]:
    return [check_sofascore(), check_apify()]
