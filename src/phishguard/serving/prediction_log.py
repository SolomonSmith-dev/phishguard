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
# login links, OAuth redirects, signed URLs). Fragments count too: implicit
# OAuth flows put access_token after '#'.
_REDACT_RE = re.compile(
    r"([?&#;](?:token|access_token|refresh_token|id_token|auth|code|state|session|sid|sig"
    r"|signature|x-amz-signature|x-amz-credential|x-amz-security-token|password|passwd|pwd"
    r"|key|api_key|apikey|secret|client_secret|otp|jwt|reset)=)[^&#;]*",
    re.IGNORECASE,
)
# user:password@ in the authority part.
_USERINFO_RE = re.compile(r"^([a-zA-Z][a-zA-Z0-9+.-]*://)[^/?#@]*@")
_FALLBACK_PATH = Path("/tmp/phishguard_predictions.jsonl")
# The file fallback has no rotation; stop appending past this size instead of
# filling the disk (the public Space has no Postgres, so this is its only sink).
_FALLBACK_MAX_BYTES = 50 * 1024 * 1024
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
    return _USERINFO_RE.sub(r"\1<redacted>@", _REDACT_RE.sub(r"\1<redacted>", url))


class PredictionLogger:
    def __init__(self, dsn: str | None = None, store_raw_url: bool | None = None) -> None:
        self.dsn = dsn or os.getenv("DATABASE_URL")
        # Raw URLs are kept by default for local ops. A public deploy sets
        # PHISHGUARD_STORE_RAW_URL=0 so only the redacted form is persisted.
        if store_raw_url is None:
            store_raw_url = os.getenv("PHISHGUARD_STORE_RAW_URL", "1") != "0"
        self.store_raw_url = store_raw_url
        self._fallback_full_warned = False
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
        redacted = redact(row.get("url", ""))
        row = {**row, "url_redacted": redacted}
        if not self.store_raw_url:
            row["url"] = redacted
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
        # Logging must never fail a prediction: a full disk or a capped file
        # drops the row with one warning instead of raising into /predict.
        try:
            with self._lock:
                if _FALLBACK_PATH.exists() and _FALLBACK_PATH.stat().st_size >= _FALLBACK_MAX_BYTES:
                    if not self._fallback_full_warned:
                        logger.warning("%s reached its size cap; dropping rows", _FALLBACK_PATH)
                        self._fallback_full_warned = True
                    return
                with _FALLBACK_PATH.open("a") as f:
                    f.write(json.dumps(row, default=str) + "\n")
        except OSError:
            logger.exception("prediction log write failed; row dropped")

    def close(self) -> None:
        if self._conn is not None:
            with suppress(Exception):
                self._conn.close()
