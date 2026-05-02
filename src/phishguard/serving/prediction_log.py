"""Prediction log writer.

Logs every prediction to a Postgres table when DATABASE_URL is set. Falls
back to a JSONL append in /tmp when the database is unreachable, so the
API never fails purely because the drift sink is down.

Schema:

    create table if not exists predictions (
        id bigserial primary key,
        ts timestamptz not null default now(),
        url text not null,
        p_url double precision,
        p_html double precision,
        p_img double precision,
        p_phish double precision not null,
        threshold double precision not null,
        is_phish boolean not null,
        latency_ms_total double precision not null,
        had_html boolean not null,
        had_img boolean not null,
        url_redacted text
    );
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
from contextlib import suppress
from pathlib import Path
from typing import Any

import psycopg
from psycopg import sql

logger = logging.getLogger("phishguard.predlog")

# Redact common credential params from URLs before persistence. Don't log
# secrets that show up in shareable URL links (ICS calendar tokens, magic
# login links, signed URLs).
_REDACT_RE = re.compile(
    r"([?&](?:token|access_token|auth|session|sig|signature|password|key|api_key)=)[^&#]*",
    re.IGNORECASE,
)
_FALLBACK_PATH = Path("/tmp/phishguard_predictions.jsonl")
_SCHEMA = """
create table if not exists predictions (
    id bigserial primary key,
    ts timestamptz not null default now(),
    url text not null,
    p_url double precision,
    p_html double precision,
    p_img double precision,
    p_phish double precision not null,
    threshold double precision not null,
    is_phish boolean not null,
    latency_ms_total double precision not null,
    had_html boolean not null,
    had_img boolean not null,
    url_redacted text
);
"""


def redact(url: str) -> str:
    return _REDACT_RE.sub(r"\1<redacted>", url)


class PredictionLogger:
    def __init__(self, dsn: str | None = None) -> None:
        self.dsn = dsn or os.getenv("DATABASE_URL")
        self._lock = threading.Lock()
        self._conn: psycopg.Connection[Any] | None = None
        self._enabled = bool(self.dsn)
        if self._enabled:
            try:
                self._conn = psycopg.connect(self.dsn, autocommit=True)
                self._conn.execute(sql.SQL(_SCHEMA))
                logger.info("predictions table ready")
            except Exception:
                logger.exception("postgres unavailable; falling back to %s", _FALLBACK_PATH)
                self._conn = None

    def log(self, row: dict[str, Any]) -> None:
        row = {**row, "url_redacted": redact(row.get("url", ""))}
        if self._conn is not None:
            try:
                with self._lock:
                    self._conn.execute(
                        """
                        insert into predictions
                          (url, p_url, p_html, p_img, p_phish, threshold, is_phish,
                           latency_ms_total, had_html, had_img, url_redacted)
                        values (%(url)s, %(p_url)s, %(p_html)s, %(p_img)s, %(p_phish)s,
                                %(threshold)s, %(is_phish)s, %(latency_ms_total)s,
                                %(had_html)s, %(had_img)s, %(url_redacted)s)
                        """,
                        row,
                    )
                return
            except Exception:
                with self._lock:
                    conn = self._conn
                    self._conn = None
                    if conn is not None:
                        with suppress(Exception):
                            conn.close()
                logger.exception("postgres insert failed; falling back to file")
        with self._lock, _FALLBACK_PATH.open("a") as f:
            f.write(json.dumps(row, default=str) + "\n")

    def close(self) -> None:
        if self._conn is not None:
            with suppress(Exception):
                self._conn.close()
