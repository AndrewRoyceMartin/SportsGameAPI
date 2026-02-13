from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from datetime import date, timedelta
import requests


@dataclass
class SportsAPIClient:
    base_url: str
    api_key: str
    timeout_s: int = 20
    _session: requests.Session = field(default_factory=requests.Session, repr=False)

    def _get(self, params: Dict[str, Any]) -> Any:
        params["APIkey"] = self.api_key
        url = self.base_url.rstrip("/")
        r = self._session.get(url, params=params, timeout=self.timeout_s)
        r.raise_for_status()
        data = r.json()
        if data.get("success") != 1:
            raise RuntimeError(f"API error: {data}")
        return data.get("result", [])

    def list_countries(self) -> List[Dict[str, Any]]:
        return self._get({"met": "Countries"})

    def list_leagues(self, country_id: Optional[str] = None) -> List[Dict[str, Any]]:
        params: Dict[str, Any] = {"met": "Leagues"}
        if country_id:
            params["countryId"] = country_id
        return self._get(params)

    def list_fixtures(
        self,
        from_date: str,
        to_date: str,
        league_id: Optional[str] = None,
        timezone: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        params: Dict[str, Any] = {"met": "Fixtures", "from": from_date, "to": to_date}
        if league_id:
            params["leagueId"] = league_id
        if timezone:
            params["timezone"] = timezone
        return self._get(params)

    def get_h2h(
        self, first_team_id: str, second_team_id: str
    ) -> Dict[str, Any]:
        return self._get({
            "met": "H2H",
            "firstTeamId": first_team_id,
            "secondTeamId": second_team_id,
        })

    def _parse_score(self, result_str: str) -> tuple[int, int] | None:
        if not result_str or result_str.strip() in ("", "-"):
            return None
        parts = result_str.split("-")
        if len(parts) != 2:
            return None
        try:
            return int(parts[0].strip()), int(parts[1].strip())
        except ValueError:
            return None

    def list_completed_games(
        self,
        league_id: str,
        from_date: str,
        to_date: str,
        timezone: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        raw = self.list_fixtures(from_date, to_date, league_id=league_id, timezone=timezone)

        games: List[Dict[str, Any]] = []
        for ev in raw:
            if ev.get("event_status") != "Finished":
                continue

            score = self._parse_score(ev.get("event_final_result", ""))
            if score is None:
                continue

            games.append({
                "game_id": str(ev.get("event_key", "")),
                "date_utc": ev.get("event_date", ""),
                "time": ev.get("event_time", ""),
                "home_team": ev.get("event_home_team", ""),
                "away_team": ev.get("event_away_team", ""),
                "home_team_key": ev.get("home_team_key", ""),
                "away_team_key": ev.get("away_team_key", ""),
                "home_score": score[0],
                "away_score": score[1],
                "league_name": ev.get("league_name", ""),
                "league_round": ev.get("league_round", ""),
                "stadium": ev.get("event_stadium", ""),
                "home_logo": ev.get("home_team_logo", ""),
                "away_logo": ev.get("away_team_logo", ""),
            })
        return games

    def list_upcoming_games(
        self,
        league_id: str,
        from_date: str,
        to_date: str,
        timezone: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        raw = self.list_fixtures(from_date, to_date, league_id=league_id, timezone=timezone)

        fixtures: List[Dict[str, Any]] = []
        for ev in raw:
            if ev.get("event_status") == "Finished":
                continue

            fixtures.append({
                "game_id": str(ev.get("event_key", "")),
                "date_utc": ev.get("event_date", ""),
                "time": ev.get("event_time", ""),
                "home_team": ev.get("event_home_team", ""),
                "away_team": ev.get("event_away_team", ""),
                "home_team_key": ev.get("home_team_key", ""),
                "away_team_key": ev.get("away_team_key", ""),
                "league_name": ev.get("league_name", ""),
                "country_name": ev.get("country_name", ""),
                "league_round": ev.get("league_round", ""),
                "stadium": ev.get("event_stadium", ""),
                "home_logo": ev.get("home_team_logo", ""),
                "away_logo": ev.get("away_team_logo", ""),
                "status": ev.get("event_status", ""),
            })
        return fixtures
