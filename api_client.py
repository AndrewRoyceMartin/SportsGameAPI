from __future__ import annotations

import requests
from typing import Any


class SportsAPIClient:
    def __init__(self, base_url: str, api_key: str) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.session = requests.Session()
        self.session.headers.update({
            "x-api-key": self.api_key,
            "Accept": "application/json",
        })

    def _get(self, endpoint: str, params: dict[str, Any] | None = None) -> Any:
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        resp = self.session.get(url, params=params, timeout=30)
        resp.raise_for_status()
        return resp.json()

    def list_completed_games(
        self, sport: str, league: str, season: str
    ) -> list[dict[str, Any]]:
        data = self._get(
            f"/{sport}/{league}/completed",
            params={"season": season},
        )
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            for key in ("games", "results", "data", "matches"):
                if key in data and isinstance(data[key], list):
                    return data[key]
        return []

    def list_upcoming_games(
        self, sport: str, league: str, season: str
    ) -> list[dict[str, Any]]:
        data = self._get(
            f"/{sport}/{league}/upcoming",
            params={"season": season},
        )
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            for key in ("games", "fixtures", "data", "matches"):
                if key in data and isinstance(data[key], list):
                    return data[key]
        return []
