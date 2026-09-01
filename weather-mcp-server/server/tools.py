"""
Tools module for the MCP server.

This module defines all the tools (functions) that the MCP server exposes to clients.
Tools are the core functionality of an MCP server - they are callable functions that
AI assistants and other clients can invoke to perform specific actions.

Each tool should:
- Have a clear, descriptive name
- Include comprehensive docstrings (used by AI to understand when to call the tool)
- Return structured data (typically dict or list)
- Handle errors gracefully
"""

import hashlib
import json
import os
import time
from typing import Any

import psycopg2
import psycopg2.extras
import requests

from server import utils

# ---------------------------------------------------------------------------
# Weather constants and helpers (ported from flask-Weather-Rag-App)
# ---------------------------------------------------------------------------

_NWS_BASE_URL = os.environ.get("NWS_API_BASE_URL", "https://api.weather.gov")
_NWS_TIMEOUT = 30
_NWS_USER_AGENT = os.environ.get(
    "NWS_USER_AGENT", "weather-mcp-server (contact: set NWS_USER_AGENT env var)"
)
_SECRET_SCOPE = os.environ.get("WEATHER_SECRET_SCOPE", "weather-app-secrets")
_SECRET_KEY = os.environ.get("WEATHER_SECRET_KEY", "database-url")
EMBEDDING_DIM = 384
DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
BATCH_SIZE = 64

CITY_COORDS = {
    "New York, NY": (40.7128, -74.0060),
    "Los Angeles, CA": (34.0522, -118.2437),
    "Chicago, IL": (41.8781, -87.6298),
    "Houston, TX": (29.7604, -95.3698),
    "Phoenix, AZ": (33.4484, -112.0740),
    "Philadelphia, PA": (39.9526, -75.1652),
    "San Antonio, TX": (29.4241, -98.4936),
    "San Diego, CA": (32.7157, -117.1611),
    "Dallas, TX": (32.7767, -96.7970),
    "San Jose, CA": (37.3382, -121.8863),
    "Austin, TX": (30.2672, -97.7431),
    "Seattle, WA": (47.6062, -122.3321),
    "Denver, CO": (39.7392, -104.9903),
    "Washington, DC": (38.9072, -77.0369),
    "Boston, MA": (42.3601, -71.0589),
    "Miami, FL": (25.7617, -80.1918),
    "Atlanta, GA": (33.7490, -84.3880),
    "San Francisco, CA": (37.7749, -122.4194),
}


class WeatherClientError(Exception):
    """Raised when the NWS API returns an unexpected response."""


class WeatherClient:
    """Thin wrapper around the NWS API with retry-friendly session."""

    def __init__(self, base_url=_NWS_BASE_URL, timeout=_NWS_TIMEOUT, coords_lookup=None):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.coords_lookup = coords_lookup or {}
        self.session = requests.Session()
        self.session.headers.update(
            {"User-Agent": _NWS_USER_AGENT, "Accept": "application/geo+json"}
        )

    def _get(self, path, params=None, retries=3):
        url = path if path.startswith("http") else f"{self.base_url}{path}"
        last_exc = None
        for attempt in range(retries):
            try:
                resp = self.session.get(url, params=params, timeout=self.timeout)
                if resp.status_code == 429:
                    time.sleep(2**attempt)
                    continue
                resp.raise_for_status()
                return resp.json()
            except requests.RequestException as exc:
                last_exc = exc
                time.sleep(min(2**attempt, 8))
        raise WeatherClientError(f"GET {url} failed after {retries} retries: {last_exc}")

    def geocode(self, location):
        if location in self.coords_lookup:
            return self.coords_lookup[location]
        headers = {"User-Agent": "weather-mcp-server/1.0"}
        resp = self.session.get(
            "https://nominatim.openstreetmap.org/search",
            params={"q": location, "format": "json", "limit": 1, "countrycodes": "us"},
            headers=headers,
            timeout=self.timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        if not data:
            raise WeatherClientError(f"Could not geocode location: {location}")
        return float(data[0]["lat"]), float(data[0]["lon"])

    def resolve_gridpoint(self, lat, lon):
        data = self._get(f"/points/{lat:.4f},{lon:.4f}")
        props = data.get("properties", {})
        return {
            "grid_id": props.get("gridId"),
            "grid_x": props.get("gridX"),
            "grid_y": props.get("gridY"),
            "forecast_url": props.get("forecast"),
            "timezone": props.get("timeZone"),
        }

    def get_active_alerts(self, lat, lon):
        data = self._get("/alerts/active", params={"point": f"{lat:.4f},{lon:.4f}"})
        return data.get("features", [])

    def get_forecast(self, forecast_url):
        data = self._get(forecast_url)
        return data.get("properties", {}).get("periods", [])

    @staticmethod
    def _stable_hash(*parts):
        raw = "|".join(p or "" for p in parts)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]

    def normalize_alert(self, location, feature):
        props = feature.get("properties", {})
        doc_id = props.get("id") or self._stable_hash(location, "alert", props.get("headline", ""))
        narrative_parts = [props.get("headline") or "", props.get("description") or "", props.get("instruction") or ""]
        return {
            "doc_id": doc_id, "location": location, "source_type": "alert",
            "headline": props.get("headline") or props.get("event"),
            "event": props.get("event"),
            "narrative_text": "\n\n".join(p for p in narrative_parts if p),
            "issued_at": props.get("sent") or props.get("effective"),
            "raw_payload": feature,
        }

    def normalize_forecast_period(self, location, grid_id, period):
        name = period.get("name", "")
        start_time = period.get("startTime", "")
        return {
            "doc_id": self._stable_hash(location, grid_id, name, start_time),
            "location": location, "source_type": "forecast",
            "headline": name, "event": period.get("shortForecast"),
            "narrative_text": f"{name}: {period.get('detailedForecast', '')}".strip(),
            "issued_at": start_time, "raw_payload": period,
        }

    def fetch_documents_for_location(self, location, limit=None):
        lat, lon = self.geocode(location)
        grid = self.resolve_gridpoint(lat, lon)
        documents = []
        for feature in self.get_active_alerts(lat, lon):
            documents.append(self.normalize_alert(location, feature))
        if grid.get("forecast_url"):
            periods = self.get_forecast(grid["forecast_url"])
            if limit is not None:
                periods = periods[:limit]
            for period in periods:
                documents.append(self.normalize_forecast_period(location, grid.get("grid_id", ""), period))
        return documents


# --- Lakebase helpers ---

_DDL_STATEMENTS = [
    "CREATE EXTENSION IF NOT EXISTS vector;",
    """CREATE TABLE IF NOT EXISTS weather_documents (
        doc_id TEXT PRIMARY KEY, location TEXT NOT NULL,
        source_type TEXT NOT NULL CHECK (source_type IN ('alert', 'forecast')),
        headline TEXT, event TEXT, narrative_text TEXT, issued_at TIMESTAMPTZ,
        raw_payload JSONB, synced_at TIMESTAMPTZ NOT NULL DEFAULT now()
    );""",
    "CREATE INDEX IF NOT EXISTS idx_weather_documents_location ON weather_documents (location);",
    """CREATE TABLE IF NOT EXISTS weather_embeddings (
        embedding_id BIGSERIAL PRIMARY KEY,
        document_id TEXT NOT NULL REFERENCES weather_documents (doc_id) ON DELETE CASCADE,
        chunk_index INT NOT NULL, chunk_text TEXT NOT NULL,
        embedding VECTOR(384) NOT NULL, model_name TEXT NOT NULL,
        created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
        UNIQUE (document_id, chunk_index, model_name)
    );""",
    """CREATE INDEX IF NOT EXISTS idx_weather_embeddings_hnsw
    ON weather_embeddings USING hnsw (embedding vector_cosine_ops);""",
]


def _get_db_connection():
    database_url = os.environ.get("DATABASE_URL")
    if not database_url or database_url.startswith("{{"):
        from databricks.sdk import WorkspaceClient
        w = WorkspaceClient()
        secret_response = w.secrets.get_secret(scope=_SECRET_SCOPE, key=_SECRET_KEY)
        secret_value = secret_response.value if hasattr(secret_response, "value") else str(secret_response)
        try:
            database_url = __import__("base64").b64decode(secret_value).decode("utf-8")
        except Exception:
            database_url = secret_value
    return psycopg2.connect(database_url)


def _upsert_weather_documents(documents):
    if not documents:
        return 0
    conn = _get_db_connection()
    try:
        cur = conn.cursor()
        rows = [(d["doc_id"], d["location"], d["source_type"], d.get("headline"), d.get("event"), d.get("narrative_text"), d.get("issued_at"), json.dumps(d.get("raw_payload") or {})) for d in documents]
        psycopg2.extras.execute_values(cur, """INSERT INTO weather_documents (doc_id, location, source_type, headline, event, narrative_text, issued_at, raw_payload) VALUES %s ON CONFLICT (doc_id) DO UPDATE SET location = EXCLUDED.location, source_type = EXCLUDED.source_type, headline = EXCLUDED.headline, event = EXCLUDED.event, narrative_text = EXCLUDED.narrative_text, issued_at = EXCLUDED.issued_at, raw_payload = EXCLUDED.raw_payload, synced_at = now()""", rows, template="(%s, %s, %s, %s, %s, %s, %s, %s::jsonb)")
        conn.commit()
        return len(rows)
    finally:
        conn.close()


def _search_weather_embeddings(query_embedding, top_k=5, model_name=DEFAULT_MODEL_NAME):
    vector_literal = "[" + ",".join(str(x) for x in query_embedding) + "]"
    conn = _get_db_connection()
    try:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute("SELECT COUNT(*) AS cnt FROM weather_embeddings")
        if cur.fetchone()["cnt"] == 0:
            return []
        cur.execute("""SELECT d.doc_id, d.location, d.source_type, d.headline, d.event, d.issued_at, e.chunk_index, e.chunk_text, 1 - (e.embedding <=> %s::vector) AS similarity FROM weather_embeddings e JOIN weather_documents d ON d.doc_id = e.document_id WHERE e.model_name = %s ORDER BY e.embedding <=> %s::vector LIMIT %s""", (vector_literal, model_name, vector_literal, top_k))
        return [dict(row) for row in cur.fetchall()]
    finally:
        conn.close()


def _chunk_text(text):
    text = (text or "").strip()
    if not text:
        return []
    if len(text) <= CHUNK_SIZE:
        return [text]
    chunks = []
    start, step = 0, max(CHUNK_SIZE - CHUNK_OVERLAP, 1)
    while start < len(text):
        end = min(start + CHUNK_SIZE, len(text))
        chunks.append(text[start:end])
        if end == len(text):
            break
        start += step
    return chunks


_embedding_model = None
_weather_client = None


def _get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        from sentence_transformers import SentenceTransformer
        _embedding_model = SentenceTransformer(DEFAULT_MODEL_NAME)
    return _embedding_model


def _get_weather_client():
    global _weather_client
    if _weather_client is None:
        _weather_client = WeatherClient(coords_lookup=CITY_COORDS)
    return _weather_client


def load_tools(mcp_server):
    """
    Register all MCP tools with the server.

    This function is called during server initialization to register all available
    tools with the MCP server instance. Tools are registered using the @mcp_server.tool
    decorator, which makes them available to clients via the MCP protocol.

    Args:
        mcp_server: The FastMCP server instance to register tools with. This is the
                   main server object that handles tool registration and routing.

    Example:
        To add a new tool, define it within this function using the decorator:

        @mcp_server.tool
        def my_new_tool(param: str) -> dict:
            '''Description of what the tool does.'''
            return {"result": f"Processed {param}"}
    """

    @mcp_server.tool
    def health() -> dict:
        """
        Check the health of the MCP server and Databricks connection.

        This is a simple diagnostic tool that confirms the server is running properly.
        It's useful for:
        - Monitoring and health checks
        - Testing the MCP connection
        - Verifying the server is responsive

        Returns:
            dict: A dictionary containing:
                - status (str): The health status ("healthy" if operational)
                - message (str): A human-readable status message

        Example response:
            {
                "status": "healthy",
                "message": "Custom MCP Server is healthy and connected to Databricks Apps."
            }
        """
        return {
            "status": "healthy",
            "message": "Custom MCP Server is healthy and connected to Databricks Apps.",
        }

    @mcp_server.tool
    def get_current_user() -> dict:
        """
        Get information about the current authenticated user.

        This tool retrieves details about the user who is currently authenticated
        with the MCP server. When deployed as a Databricks App, this returns
        information about the end user making the request. When running locally,
        it returns information about the developer's Databricks identity.

        Useful for:
        - Personalizing responses based on the user
        - Authorization checks
        - Audit logging
        - User-specific operations

        Returns:
            dict: A dictionary containing:
                - display_name (str): The user's display name
                - user_name (str): The user's username/email
                - active (bool): Whether the user account is active

        Example response:
            {
                "display_name": "John Doe",
                "user_name": "john.doe@example.com",
                "active": true
            }

        Raises:
            Returns error dict if authentication fails or user info cannot be retrieved.
        """
        try:
            w = utils.get_user_authenticated_workspace_client()
            user = w.current_user.me()
            return {
                "display_name": user.display_name,
                "user_name": user.user_name,
                "active": user.active,
            }
        except Exception as e:
            return {"error": str(e), "message": "Failed to retrieve user information"}

    # ------------------------------------------------------------------
    # Weather Tools (ported from flask-Weather-Rag-App)
    # ------------------------------------------------------------------

    @mcp_server.tool
    def get_weather_alerts(location: str) -> dict:
        """
        Fetch active weather alerts (watches, warnings, advisories) for a US location.

        Uses the National Weather Service API. The location is geocoded to
        lat/lon (major US cities are resolved instantly; other locations fall
        back to geocoding via Nominatim). Only US locations are supported.

        Args:
            location: A free-text US location string, e.g. "Chicago, IL" or "Seattle, WA".

        Returns:
            dict: A dictionary containing:
                - location (str): The resolved location string
                - alert_count (int): Number of active alerts found
                - alerts (list[dict]): Each alert has headline, event, severity,
                  description, instruction, and issued_at fields.
                - If no alerts, returns an empty alerts list.

        Example response:
            {
                "location": "Chicago, IL",
                "alert_count": 1,
                "alerts": [{"headline": "Winter Storm Warning", ...}]
            }
        """
        try:
            client = _get_weather_client()
            lat, lon = client.geocode(location)
            features = client.get_active_alerts(lat, lon)
            alerts = []
            for f in features:
                props = f.get("properties", {})
                alerts.append({
                    "headline": props.get("headline"),
                    "event": props.get("event"),
                    "severity": props.get("severity"),
                    "description": props.get("description"),
                    "instruction": props.get("instruction"),
                    "issued_at": props.get("sent") or props.get("effective"),
                })
            return {"location": location, "alert_count": len(alerts), "alerts": alerts}
        except WeatherClientError as e:
            return {"error": str(e), "message": f"Failed to fetch alerts for {location}"}
        except Exception as e:
            return {"error": str(e), "message": f"Failed to fetch alerts for {location}"}

    @mcp_server.tool
    def get_weather_forecast(location: str, limit: int = 14) -> dict:
        """
        Fetch the narrative weather forecast for a US location.

        Retrieves multi-day forecast periods (day and night) from the National
        Weather Service. Each period includes a name, temperature, wind info,
        and a detailed narrative forecast.

        Args:
            location: A free-text US location string, e.g. "Austin, TX".
            limit: Maximum number of forecast periods to return (default 14).
                   Each period covers ~12 hours, so 14 gives roughly a week.

        Returns:
            dict: A dictionary containing:
                - location (str): The resolved location string
                - period_count (int): Number of forecast periods returned
                - periods (list[dict]): Each period has name, startTime, endTime,
                  temperature, temperatureUnit, windSpeed, windDirection,
                  shortForecast, and detailedForecast fields.

        Example response:
            {
                "location": "Austin, TX",
                "period_count": 14,
                "periods": [{"name": "This Afternoon", "temperature": 75, ...}]
            }
        """
        try:
            client = _get_weather_client()
            lat, lon = client.geocode(location)
            grid = client.resolve_gridpoint(lat, lon)
            if not grid.get("forecast_url"):
                return {"error": "Could not resolve forecast grid for location", "location": location}
            periods_raw = client.get_forecast(grid["forecast_url"])
            periods = periods_raw[:limit]
            simplified = []
            for p in periods:
                simplified.append({
                    "name": p.get("name"),
                    "startTime": p.get("startTime"),
                    "endTime": p.get("endTime"),
                    "temperature": p.get("temperature"),
                    "temperatureUnit": p.get("temperatureUnit"),
                    "windSpeed": p.get("windSpeed"),
                    "windDirection": p.get("windDirection"),
                    "shortForecast": p.get("shortForecast"),
                    "detailedForecast": p.get("detailedForecast"),
                })
            return {"location": location, "period_count": len(simplified), "periods": simplified}
        except WeatherClientError as e:
            return {"error": str(e), "message": f"Failed to fetch forecast for {location}"}
        except Exception as e:
            return {"error": str(e), "message": f"Failed to fetch forecast for {location}"}

    @mcp_server.tool
    def sync_weather_data(locations: list[str], limit: int = 50) -> dict:
        """
        Sync weather data (alerts + forecasts) from the NWS API into the Lakebase
        Postgres database for later semantic search.

        This tool fetches active alerts and forecast periods for each location,
        normalizes them into documents, and upserts them into the weather_documents
        table. Re-running sync for the same location updates existing rows rather
        than creating duplicates. After syncing, run search_weather to query the
        data semantically.

        Args:
            locations: List of US location strings, e.g. ["Chicago, IL", "Austin, TX"].
            limit: Maximum number of forecast periods to sync per location (default 50).

        Returns:
            dict: A dictionary containing:
                - total_documents (int): Total documents upserted across all locations
                - per_location (list[dict]): Per-location results with location,
                  documents_synced, and any error message.

        Example response:
            {
                "total_documents": 52,
                "per_location": [
                    {"location": "Chicago, IL", "documents_synced": 28, "status": "success"},
                    {"location": "Austin, TX", "documents_synced": 24, "status": "success"}
                ]
            }
        """
        client = _get_weather_client()
        per_location = []
        total = 0
        for loc in locations:
            try:
                docs = client.fetch_documents_for_location(loc, limit=limit)
                count = _upsert_weather_documents(docs)
                total += count
                per_location.append({"location": loc, "documents_synced": count, "status": "success"})
            except Exception as e:
                per_location.append({"location": loc, "documents_synced": 0, "status": "error", "error": str(e)})
        return {"total_documents": total, "per_location": per_location}

    @mcp_server.tool
    def search_weather(query: str, top_k: int = 5) -> dict:
        """
        Search weather documents semantically using natural language.

        Runs a pgvector cosine-distance search over weather alerts and forecasts
        previously synced via sync_weather_data. The query is embedded using the
        MiniLM sentence-transformer model and matched against document chunks in
        the Lakebase database. Results are ranked by semantic similarity.

        Prerequisite: Call sync_weather_data at least once before searching, so
        that documents and embeddings exist in the database.

        Args:
            query: A natural-language weather question, e.g. "risk of flooding near rivers"
                   or "winter storm warnings in the Midwest".
            top_k: Number of top results to return (default 5, clamped to 1-20).

        Returns:
            dict: A dictionary containing:
                - query (str): The original query string
                - result_count (int): Number of matching documents returned
                - results (list[dict]): Each result has location, source_type,
                  headline, event, chunk_text, and similarity score.
                - If no data has been synced yet, returns an empty results list.

        Example response:
            {
                "query": "risk of flooding near rivers",
                "result_count": 3,
                "results": [{"location": "Houston, TX", "similarity": 0.82, ...}]
            }
        """
        top_k = max(1, min(top_k, 20))
        try:
            model = _get_embedding_model()
            query_embedding = model.encode([query], normalize_embeddings=True)[0].tolist()
            results = _search_weather_embeddings(query_embedding, top_k=top_k)
            simplified = []
            for r in results:
                simplified.append({
                    "location": r.get("location"),
                    "source_type": r.get("source_type"),
                    "headline": r.get("headline"),
                    "event": r.get("event"),
                    "issued_at": str(r.get("issued_at")) if r.get("issued_at") else None,
                    "chunk_text": r.get("chunk_text"),
                    "similarity": round(r.get("similarity", 0), 4),
                })
            return {"query": query, "result_count": len(simplified), "results": simplified}
        except Exception as e:
            return {"query": query, "result_count": 0, "results": [], "error": str(e),
                    "message": "Search failed. Ensure sync_weather_data has been run first."}
    @mcp_server.tool
    def assess_weather_risk(location: str) -> dict:
        """
        Assess severe weather risk for a US location by combining NWS alerts
        with forecast temperature analysis.

        This tool does more than return raw API data. It applies the following
        threshold logic to produce a risk assessment:

        Alert-based scoring (NWS alert severity field):
          - 'Extreme' severity alert present      -> risk = EXTREME
          - 'Severe' severity alert present      -> risk = HIGH
          - 'Moderate' severity alert present    -> risk = MODERATE
          - 'Minor' severity or Advisory only     -> risk = LOW

        Forecast-based overrides (if no higher alert risk already set):
          - Any forecast period temperature >= 100F  -> bumps risk to HIGH (heat)
          - Any forecast period temperature <= 20F   -> bumps risk to HIGH (cold)
          - Forecast text contains 'thunderstorm' or 'severe' -> bumps to MODERATE

        Keyword flags (from alert event names and forecast short descriptions):
          - tornado, flash flood, blizzard -> flagged as 'severe_hazard'
          - heat, excessive heat           -> flagged as 'heat_hazard'
          - winter storm, ice, freeze      -> flagged as 'cold_hazard'
          - thunderstorm, hail, damaging wind -> flagged as 'storm_hazard'
          - flood, flooding                -> flagged as 'flood_hazard'

        The final risk level is the maximum of the alert-based and forecast-based
        scores. A summary explains the reasoning so the calling agent can relay
        it to the user without hallucinating additional details.

        Args:
            location: A free-text US location string, e.g. "Chicago, IL".

        Returns:
            dict: A dictionary containing:
                - location (str): The resolved location string
                - risk_level (str): One of 'EXTREME', 'HIGH', 'MODERATE', 'LOW'
                - risk_score (int): Numeric score 0-3 (0=low, 3=extreme)
                - hazards (list[str]): Detected hazard keywords
                - active_alerts (list[dict]): Simplified alerts with event and severity
                - temperature_extremes (dict): Max and min temperatures from forecast
                - summary (str): Human-readable explanation of the risk assessment

        Example response:
            {
                "location": "Phoenix, AZ",
                "risk_level": "HIGH",
                "risk_score": 2,
                "hazards": ["heat_hazard"],
                "active_alerts": [{"event": "Excessive Heat Warning", "severity": "Severe"}],
                "temperature_extremes": {"max": 112, "min": 78, "unit": "F"},
                "summary": "HIGH risk: Excessive Heat Warning active. Forecast high of 112F exceeds the 100F extreme-heat threshold."
            }
        """
        try:
            client = _get_weather_client()
            lat, lon = client.geocode(location)

            # Fetch alerts
            alert_features = client.get_active_alerts(lat, lon)
            alerts_simplified = []
            for f in alert_features:
                props = f.get("properties", {})
                alerts_simplified.append({
                    "event": props.get("event"),
                    "severity": props.get("severity"),
                    "headline": props.get("headline"),
                })

            # Fetch forecast for temperature analysis
            grid = client.resolve_gridpoint(lat, lon)
            periods = []
            if grid.get("forecast_url"):
                periods = client.get_forecast(grid["forecast_url"])

            # --- Apply threshold logic ---
            risk_score = 0  # 0=low, 1=moderate, 2=high, 3=extreme
            risk_labels = {0: "LOW", 1: "MODERATE", 2: "HIGH", 3: "EXTREME"}
            reasons = []
            hazards = []

            # Hazard keyword maps
            all_keyword_maps = [
                {"tornado": "severe_hazard", "flash flood": "flood_hazard", "blizzard": "cold_hazard"},
                {"heat": "heat_hazard", "excessive heat": "heat_hazard"},
                {"winter storm": "cold_hazard", "ice": "cold_hazard", "freeze": "cold_hazard", "frost": "cold_hazard"},
                {"thunderstorm": "storm_hazard", "hail": "storm_hazard", "damaging wind": "storm_hazard", "severe": "storm_hazard"},
                {"flood": "flood_hazard", "flooding": "flood_hazard"},
            ]

            # Alert-based scoring
            for alert in alerts_simplified:
                sev = (alert.get("severity") or "").lower()
                event_text = (alert.get("event") or "").lower()

                if sev == "extreme":
                    risk_score = max(risk_score, 3)
                    reasons.append(f"Extreme severity alert: {alert.get('event')}")
                elif sev == "severe":
                    risk_score = max(risk_score, 2)
                    reasons.append(f"Severe alert: {alert.get('event')}")
                elif sev == "moderate":
                    risk_score = max(risk_score, 1)
                    reasons.append(f"Moderate alert: {alert.get('event')}")
                else:
                    reasons.append(f"Minor alert: {alert.get('event')}")

                for kw_map in all_keyword_maps:
                    for kw, hazard in kw_map.items():
                        if kw in event_text and hazard not in hazards:
                            hazards.append(hazard)

            # Forecast-based temperature analysis
            max_temp = None
            min_temp = None
            for p in periods:
                temp = p.get("temperature")
                if temp is not None:
                    if max_temp is None or temp > max_temp:
                        max_temp = temp
                    if min_temp is None or temp < min_temp:
                        min_temp = temp

                combined = f"{(p.get('shortForecast') or '').lower()} {(p.get('detailedForecast') or '').lower()}"
                for kw_map in all_keyword_maps:
                    for kw, hazard in kw_map.items():
                        if kw in combined and hazard not in hazards:
                            hazards.append(hazard)

                if ("thunderstorm" in combined or "severe" in combined) and risk_score < 1:
                    risk_score = 1
                    reasons.append("Forecast mentions thunderstorms or severe conditions")

            # Temperature threshold overrides
            if max_temp is not None and max_temp >= 100:
                risk_score = max(risk_score, 2)
                reasons.append(f"Forecast high of {max_temp}F exceeds the 100F extreme-heat threshold")
                if "heat_hazard" not in hazards:
                    hazards.append("heat_hazard")

            if min_temp is not None and min_temp <= 20:
                risk_score = max(risk_score, 2)
                reasons.append(f"Forecast low of {min_temp}F is at or below the 20F extreme-cold threshold")
                if "cold_hazard" not in hazards:
                    hazards.append("cold_hazard")

            # Build summary
            risk_level = risk_labels[risk_score]
            if not reasons:
                reasons.append("No active alerts and temperatures within normal range")
            summary = f"{risk_level} risk: " + ". ".join(reasons) + "."

            return {
                "location": location,
                "risk_level": risk_level,
                "risk_score": risk_score,
                "hazards": hazards,
                "active_alerts": alerts_simplified,
                "temperature_extremes": {"max": max_temp, "min": min_temp, "unit": "F"},
                "summary": summary,
            }
        except WeatherClientError as e:
            return {"error": str(e), "message": f"Failed to assess weather risk for {location}. The location may not be valid or the NWS API may be unavailable. Ask the user to verify the location."}
        except Exception as e:
            return {"error": str(e), "message": f"Failed to assess weather risk for {location}. Ask the user to try a different location."}

