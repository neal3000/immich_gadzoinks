"""Automatic `ai/` tagging, run from inside the gadzoinks sidecar.

Replaces the standalone SQLite worker. State lives in Immich's own Postgres
under the `gz_tag_*` prefix, alongside `gz_asset_metadata`, so backups cover
it and `ON DELETE CASCADE` cleans up after deleted assets — something the
SQLite version never did.

The one design decision worth preserving: inference output and tag output are
stored separately. `gz_tag_prediction` holds what the model said, at a low
confidence floor, stamped with the model version. `gz_tag_applied` holds what
reached Immich, stamped with the vocabulary version. Neither derives from the
other at write time, so editing the vocabulary or a threshold is a re-map
over rows already in the database — seconds — rather than hours of
re-inference.

Called by main.py on a timer. Two entry points:

    run_pass(...)      one tick: infer what is new, map what changed
    status(...)        counts, for an endpoint or the log

Everything here is synchronous and CPU-bound, so main.py must run it in a
worker thread. In the event loop it would stall every /gz/ request for the
length of the pass, which during a first backfill is hours.
"""

from __future__ import annotations

import hashlib
import os
import threading
import time
import traceback
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from io import BytesIO
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import psycopg2
import psycopg2.extras
import requests as req

from gz_prompts import PromptIndex
from gz_taggers import Prediction, above_threshold
from gz_vocabulary import Vocabulary

# ---------------------------------------------------------------------------
# Configuration — environment only, so the container needs no arguments.
# ---------------------------------------------------------------------------


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


@dataclass
class Config:
    enabled: bool = field(
        default_factory=lambda: os.environ.get("GZ_TAG_ENABLED", "0") == "1"
    )
    interval: int = field(default_factory=lambda: _env_int("GZ_TAG_INTERVAL", 300))
    vocab_path: Path = field(
        default_factory=lambda: Path(
            os.environ.get("GZ_TAG_VOCAB", "/app/gz_vocabulary.yaml")
        )
    )
    model: str = field(
        default_factory=lambda: os.environ.get(
            "GZ_TAG_MODEL", "SmilingWolf/wd-vit-tagger-v3"
        )
    )
    general: float | None = field(
        default_factory=lambda: (
            _env_float("GZ_TAG_GENERAL", -1.0)
            if os.environ.get("GZ_TAG_GENERAL")
            else None
        )
    )
    prompt_boost: float = field(
        default_factory=lambda: _env_float("GZ_TAG_PROMPT_BOOST", 0.0)
    )
    prompt_floor: float = field(
        default_factory=lambda: _env_float("GZ_TAG_PROMPT_FLOOR", 0.15)
    )
    batch: int = field(default_factory=lambda: _env_int("GZ_TAG_BATCH", 8))
    # A cap per tick keeps one pass bounded even on a cold library, so the
    # first run cannot hold the model and the CPU for eight hours straight.
    per_pass: int = field(default_factory=lambda: _env_int("GZ_TAG_PER_PASS", 500))
    image_size: str = field(
        default_factory=lambda: os.environ.get("GZ_TAG_IMAGE_SIZE", "preview")
    )
    immich_url: str = field(
        default_factory=lambda: os.environ.get("IMMICH_URL", "http://localhost:2281")
    )
    immich_key: str = field(
        default_factory=lambda: os.environ.get("IMMICH_API_KEY", "")
    )
    dry_run: bool = field(
        default_factory=lambda: os.environ.get("GZ_TAG_DRY_RUN", "0") == "1"
    )

    @property
    def thresholds(self) -> dict[str, float]:
        base = {"general": 0.371, "character": 0.75}
        if self.general is not None and self.general >= 0:
            base["general"] = self.general
        return base


# Predictions are stored well below any shippable threshold, so changing a
# threshold is a re-map rather than a re-inference.
FLOOR = 0.05

# Backoff for an asset Immich has not finished ingesting. A preview that does
# not exist yet is a wait, not a failure: marking those failed would skip
# precisely the newest uploads, and do it invisibly.
BACKOFF_MINUTES = (2, 5, 15, 60, 240)
MAX_ATTEMPTS = 8


# ---------------------------------------------------------------------------
# Schema — appended to main.py's SCHEMA_SQL
# ---------------------------------------------------------------------------

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS gz_tag_asset (
    asset_id      UUID PRIMARY KEY REFERENCES asset(id) ON DELETE CASCADE,
    tagger        TEXT        NOT NULL,
    model_version TEXT        NOT NULL,
    inferred_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    -- Which vocabulary this asset was last mapped against. Held on the asset
    -- rather than only on its tags, because an asset that produced no tags
    -- still needs reconsidering when the vocabulary changes.
    mapped_vocab  TEXT
);

CREATE TABLE IF NOT EXISTS gz_tag_prediction (
    asset_id   UUID NOT NULL REFERENCES asset(id) ON DELETE CASCADE,
    label      TEXT NOT NULL,
    confidence REAL,
    category   TEXT,
    PRIMARY KEY (asset_id, label)
);

CREATE TABLE IF NOT EXISTS gz_tag_applied (
    asset_id           UUID NOT NULL REFERENCES asset(id) ON DELETE CASCADE,
    path               TEXT NOT NULL,
    vocabulary_version TEXT NOT NULL,
    applied_at         TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (asset_id, path)
);

CREATE TABLE IF NOT EXISTS gz_tag_deferred (
    asset_id UUID PRIMARY KEY REFERENCES asset(id) ON DELETE CASCADE,
    reason   TEXT,
    attempts INTEGER     NOT NULL DEFAULT 0,
    next_try TIMESTAMPTZ NOT NULL
);

CREATE TABLE IF NOT EXISTS gz_tag_state (
    key   TEXT PRIMARY KEY,
    value TEXT
);

CREATE INDEX IF NOT EXISTS gz_tag_prediction_asset ON gz_tag_prediction(asset_id);
CREATE INDEX IF NOT EXISTS gz_tag_applied_asset    ON gz_tag_applied(asset_id);
CREATE INDEX IF NOT EXISTS gz_tag_asset_mapped     ON gz_tag_asset(mapped_vocab);
CREATE INDEX IF NOT EXISTS gz_tag_deferred_next    ON gz_tag_deferred(next_try);
"""


def vocabulary_version(path: Path) -> str:
    """Content hash, so an edit is noticed without anyone bumping a number."""
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12]


# ---------------------------------------------------------------------------
# Queries
# ---------------------------------------------------------------------------


def needs_inference(cur, model_version: str, limit: int) -> list[str]:
    """Image assets with no predictions from this model, newest first.

    Done in SQL rather than by paging the search API, which is the single
    biggest change from the standalone worker: `LEFT JOIN ... IS NULL` over
    an indexed column replaces walking the whole library over HTTP every
    five minutes.

    Deleted and trashed assets are excluded here rather than discovered by a
    failing fetch later. So is `visibility = 'locked'`: those are Immich's
    locked folder, and writing tags for them would surface their content in
    the tag tree, which is the one place a private asset must not appear.
    `hidden` is the video half of a live photo — nothing to tag.
    """
    cur.execute(
        """
        SELECT a.id
          FROM asset a
          LEFT JOIN gz_tag_asset t
                 ON t.asset_id = a.id AND t.model_version = %s
          LEFT JOIN gz_tag_deferred d
                 ON d.asset_id = a.id
                AND (d.next_try > now() OR d.attempts >= %s)
         WHERE a.type = 'IMAGE'
           AND a."deletedAt" IS NULL
           AND a.visibility IN ('timeline', 'archive')
           AND t.asset_id IS NULL
           AND d.asset_id IS NULL
         ORDER BY a."createdAt" DESC
         LIMIT %s
        """,
        (model_version, MAX_ATTEMPTS, limit),
    )
    return [r["id"] for r in cur.fetchall()]


def due_deferrals(cur, limit: int) -> list[str]:
    cur.execute(
        """
        SELECT d.asset_id
          FROM gz_tag_deferred d
          JOIN asset a ON a.id = d.asset_id
                      AND a."deletedAt" IS NULL
                      AND a.visibility IN ('timeline', 'archive')
         WHERE d.next_try <= now() AND d.attempts < %s
         ORDER BY d.next_try
         LIMIT %s
        """,
        (MAX_ATTEMPTS, limit),
    )
    return [r["asset_id"] for r in cur.fetchall()]


def needs_mapping(cur, vocab_version: str, limit: int) -> list[str]:
    """Assets never mapped, or mapped against a different vocabulary.

    This is what keeps a tick proportional to what changed. Mapping every
    asset with predictions on every tick is invisible at 500 assets and
    about 100,000 pointless queries every five minutes at 50,000.
    """
    cur.execute(
        """
        SELECT asset_id FROM gz_tag_asset
         WHERE mapped_vocab IS NULL OR mapped_vocab <> %s
         ORDER BY inferred_at DESC
         LIMIT %s
        """,
        (vocab_version, limit),
    )
    return [r["asset_id"] for r in cur.fetchall()]


def record_predictions(
    cur, asset_id: str, tagger: str, model_version: str, rows: Sequence[tuple]
) -> None:
    cur.execute(
        """
        INSERT INTO gz_tag_asset(asset_id, tagger, model_version, inferred_at)
        VALUES (%s, %s, %s, now())
        ON CONFLICT (asset_id) DO UPDATE
           SET tagger = EXCLUDED.tagger,
               model_version = EXCLUDED.model_version,
               inferred_at = now(),
               -- A new model invalidates the mapping too.
               mapped_vocab = NULL
        """,
        (asset_id, tagger, model_version),
    )
    # Replace rather than merge, so a model swap cannot leave a mixture of
    # two models' labels on one asset.
    cur.execute("DELETE FROM gz_tag_prediction WHERE asset_id = %s", (asset_id,))
    if rows:
        psycopg2.extras.execute_values(
            cur,
            "INSERT INTO gz_tag_prediction(asset_id, label, confidence, category) "
            "VALUES %s",
            [(asset_id, label, conf, cat) for label, conf, cat in rows],
        )
    cur.execute("DELETE FROM gz_tag_deferred WHERE asset_id = %s", (asset_id,))


def defer(cur, asset_id: str, reason: str) -> None:
    cur.execute(
        "SELECT attempts FROM gz_tag_deferred WHERE asset_id = %s", (asset_id,)
    )
    row = cur.fetchone()
    attempts = (row["attempts"] if row else 0) + 1
    minutes = BACKOFF_MINUTES[min(attempts - 1, len(BACKOFF_MINUTES) - 1)]
    cur.execute(
        """
        INSERT INTO gz_tag_deferred(asset_id, reason, attempts, next_try)
        VALUES (%s, %s, %s, now() + (%s || ' minutes')::interval)
        ON CONFLICT (asset_id) DO UPDATE
           SET reason = EXCLUDED.reason,
               attempts = EXCLUDED.attempts,
               next_try = EXCLUDED.next_try
        """,
        (asset_id, reason[:300], attempts, str(minutes)),
    )


def predictions_for(cur, asset_id: str) -> list[tuple]:
    cur.execute(
        "SELECT label, confidence, category FROM gz_tag_prediction "
        "WHERE asset_id = %s",
        (asset_id,),
    )
    return [(r["label"], r["confidence"], r["category"]) for r in cur.fetchall()]


def prompt_for(cur, asset_id: str) -> str | None:
    """The positive prompt from the metadata the sidecar already stores.

    Deliberately selects one column. `negative_prompt` sits in the same row
    and names what was *excluded* — mining it would tag an image with
    precisely what is absent — so it is never read.
    """
    cur.execute(
        "SELECT prompt FROM gz_asset_metadata WHERE asset_id = %s", (asset_id,)
    )
    row = cur.fetchone()
    return row["prompt"] if row else None


def applied_paths(cur, asset_id: str) -> set[str]:
    cur.execute("SELECT path FROM gz_tag_applied WHERE asset_id = %s", (asset_id,))
    return {r["path"] for r in cur.fetchall()}


def get_state(cur, key: str, default: str | None = None) -> str | None:
    cur.execute("SELECT value FROM gz_tag_state WHERE key = %s", (key,))
    row = cur.fetchone()
    return row["value"] if row else default


def set_state(cur, key: str, value: str) -> None:
    cur.execute(
        "INSERT INTO gz_tag_state(key, value) VALUES (%s, %s) "
        "ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value",
        (key, value),
    )


# ---------------------------------------------------------------------------
# Immich API — only for things SQL cannot do: fetch pixels, write tags
# ---------------------------------------------------------------------------


class ImmichAPI:
    def __init__(self, url: str, key: str, timeout: float = 30.0):
        self.url = url.rstrip("/")
        self.session = req.Session()
        self.session.headers.update({"x-api-key": key})
        self.timeout = timeout

    def image(self, asset_id: str, size: str) -> bytes | None:
        """None when the rendition does not exist yet."""
        r = self.session.get(
            f"{self.url}/api/assets/{asset_id}/thumbnail",
            params={"size": size},
            headers={"Accept": "image/*"},
            timeout=self.timeout,
        )
        if r.status_code in (400, 404):
            return None
        r.raise_for_status()
        return r.content

    def upsert_tags(self, paths: list[str]) -> dict[str, str]:
        r = self.session.put(
            f"{self.url}/api/tags", json={"tags": paths}, timeout=self.timeout
        )
        r.raise_for_status()
        return {t["value"]: t["id"] for t in r.json()}

    def all_tags(self) -> dict[str, str]:
        r = self.session.get(f"{self.url}/api/tags", timeout=self.timeout)
        r.raise_for_status()
        return {t["value"]: t["id"] for t in r.json()}

    def link(self, tag_id: str, asset_ids: list[str]) -> int:
        """One tag to many assets.

        This direction, rather than many tags to one asset, is what makes a
        pass cheap: 500 assets across 60 paths is 60 calls, not 500.
        """
        if not asset_ids:
            return 0
        r = self.session.put(
            f"{self.url}/api/tags/{tag_id}/assets",
            json={"ids": asset_ids},
            timeout=self.timeout,
        )
        r.raise_for_status()
        return sum(1 for row in r.json() if row.get("success"))


# ---------------------------------------------------------------------------
# Model, loaded lazily and released
# ---------------------------------------------------------------------------

_model_lock = threading.Lock()
_model: Any = None
_model_id: str | None = None


def get_model(repo_id: str, log: Callable[[str], None]):
    """Loads on demand and keeps it only while passes keep finding work.

    The sidecar also serves HTTP, so a permanently resident gigabyte is rent
    paid on every idle tick. Checking for work before loading means the
    common case — nothing new — never touches the model at all.
    """
    global _model, _model_id
    with _model_lock:
        if _model is not None and _model_id == repo_id:
            return _model
        from gz_taggers import WDTagger

        log(f"tagger: loading {repo_id}")
        _model = WDTagger(repo_id=repo_id)
        _model_id = repo_id
        log(f"tagger: {_model.name} ready on {_model.provider}")
        return _model


def release_model(log: Callable[[str], None]) -> None:
    global _model, _model_id
    with _model_lock:
        if _model is None:
            return
        _model = None
        _model_id = None
        log("tagger: model released")


# ---------------------------------------------------------------------------
# A pass
# ---------------------------------------------------------------------------


def _infer(
    api: ImmichAPI,
    cur,
    model,
    cfg: Config,
    asset_ids: Sequence[str],
    log: Callable[[str], None],
) -> tuple[int, int]:
    from PIL import Image

    done = skipped = 0
    for i in range(0, len(asset_ids), cfg.batch):
        chunk = asset_ids[i : i + cfg.batch]
        images, kept = [], []
        for asset_id in chunk:
            try:
                blob = api.image(asset_id, cfg.image_size)
            except Exception as exc:
                defer(cur, asset_id, f"fetch: {exc}")
                skipped += 1
                continue
            if blob is None:
                defer(cur, asset_id, "no preview rendition yet")
                skipped += 1
                continue
            try:
                image = Image.open(BytesIO(blob))
                image.load()
            except Exception as exc:
                defer(cur, asset_id, f"decode: {exc}")
                skipped += 1
                continue
            images.append(image)
            kept.append(asset_id)

        if not images:
            continue
        try:
            results = model.predict(images)
        except Exception as exc:
            for asset_id in kept:
                defer(cur, asset_id, f"inference: {exc}")
            skipped += len(kept)
            continue

        for asset_id, predictions in zip(kept, results):
            rows = [
                (p.label, p.confidence, p.category)
                for p in predictions
                if p.confidence is None or p.confidence >= FLOOR
            ]
            record_predictions(cur, asset_id, model.name, cfg.model, rows)
        for image in images:
            image.close()
        done += len(kept)

    return done, skipped


def _map(
    api: ImmichAPI,
    cur,
    cfg: Config,
    vocab: Vocabulary,
    vocab_version: str,
    asset_ids: Sequence[str],
    log: Callable[[str], None],
) -> dict[str, int]:
    thresholds = cfg.thresholds
    pending: dict[str, list[str]] = defaultdict(list)
    counts = {"considered": 0, "with_tags": 0, "links": 0, "promoted": 0}

    for asset_id in asset_ids:
        counts["considered"] += 1
        raw = predictions_for(cur, asset_id)
        objs = [Prediction(label, conf, cat) for label, conf, cat in raw]

        if cfg.prompt_boost > 0:
            index = PromptIndex.build(prompt_for(cur, asset_id))
        else:
            index = None

        if index is None:
            kept = above_threshold(objs, thresholds)
        else:
            plain = {p.label for p in above_threshold(objs, thresholds)}
            kept = []
            for p in objs:
                limit = thresholds.get(p.category, thresholds["general"])
                # Only general tags get the boost; a prompt should not talk
                # up a character or a content rating.
                if p.category == "general" and index.mentions(p.label):
                    limit = max(cfg.prompt_floor, limit - cfg.prompt_boost)
                if p.confidence is None or p.confidence >= limit:
                    kept.append(p)
            counts["promoted"] += sum(1 for p in kept if p.label not in plain)

        paths, _, _ = vocab.apply([(p.label, p.category) for p in kept])
        if paths:
            counts["with_tags"] += 1
        have = applied_paths(cur, asset_id)
        for path in paths:
            if path not in have:
                pending[path].append(asset_id)

    if pending and not cfg.dry_run:
        paths = sorted(pending)
        ids = api.upsert_tags(paths)
        missing = [p for p in paths if p not in ids]
        if missing:
            ids.update(
                {k: v for k, v in api.all_tags().items() if k in missing}
            )
        for path in paths:
            tag_id = ids.get(path)
            if not tag_id:
                log(f"tagger: no tag id for {path}, skipped")
                continue
            counts["links"] += api.link(tag_id, pending[path])
            psycopg2.extras.execute_values(
                cur,
                "INSERT INTO gz_tag_applied(asset_id, path, vocabulary_version) "
                "VALUES %s ON CONFLICT (asset_id, path) DO UPDATE "
                "SET vocabulary_version = EXCLUDED.vocabulary_version, "
                "applied_at = now()",
                [(a, path, vocab_version) for a in pending[path]],
            )
    elif pending:
        counts["links"] = sum(len(v) for v in pending.values())
        for path in sorted(pending)[:20]:
            log(f"tagger: would tag {len(pending[path])} assets {path}")

    if not cfg.dry_run and asset_ids:
        # Everything considered is current now, including assets that
        # produced nothing — otherwise they are reconsidered forever.
        psycopg2.extras.execute_values(
            cur,
            "UPDATE gz_tag_asset SET mapped_vocab = data.v "
            "FROM (VALUES %s) AS data(a, v) WHERE asset_id = data.a::uuid",
            [(a, vocab_version) for a in asset_ids],
        )

    return counts


_pass_lock = threading.Lock()


def run_pass(
    get_connection: Callable[[], Any],
    cfg: Config | None = None,
    log: Callable[[str], None] = print,
) -> dict[str, int] | None:
    """One tick. Returns counts, or None if a pass was already running.

    The guard is not optional: a first backfill outruns a five-minute timer
    by hours, and stacking passes would have several model copies and
    several writers going at once.
    """
    cfg = cfg or Config()
    if not cfg.enabled:
        return None
    if not _pass_lock.acquire(blocking=False):
        log("tagger: previous pass still running, skipping this tick")
        return None

    try:
        return _run_pass_locked(get_connection, cfg, log)
    except Exception:
        log("tagger: pass failed\n" + traceback.format_exc())
        return None
    finally:
        _pass_lock.release()


def _run_pass_locked(get_connection, cfg: Config, log) -> dict[str, int]:
    if not cfg.vocab_path.exists():
        log(f"tagger: no vocabulary at {cfg.vocab_path}; nothing to do")
        return {}
    vocab = Vocabulary.load(cfg.vocab_path)
    vocab_version = vocabulary_version(cfg.vocab_path)

    api = ImmichAPI(cfg.immich_url, cfg.immich_key)
    started = time.monotonic()
    totals = {"inferred": 0, "skipped": 0, "mapped": 0, "links": 0, "promoted": 0}

    conn = get_connection()
    try:
        conn.autocommit = False
        cur = conn.cursor()

        # Find the work before loading the model: most ticks have none, and
        # the model costs seconds and a gigabyte.
        candidates = needs_inference(cur, cfg.model, cfg.per_pass)
        room = cfg.per_pass - len(candidates)
        if room > 0:
            have = set(candidates)
            candidates += [a for a in due_deferrals(cur, room) if a not in have]
        conn.commit()

        if candidates:
            log(f"tagger: {len(candidates)} assets need inference")
            model = get_model(cfg.model, log)
            done, skipped = _infer(api, cur, model, cfg, candidates, log)
            conn.commit()
            totals["inferred"], totals["skipped"] = done, skipped
            log(f"tagger: inferred {done}, deferred {skipped}")

        targets = needs_mapping(cur, vocab_version, cfg.per_pass * 4)
        if targets:
            counts = _map(api, cur, cfg, vocab, vocab_version, targets, log)
            set_state(cur, "vocabulary_version", vocab_version)
            set_state(cur, "model_version", cfg.model)
            conn.commit()
            totals["mapped"] = counts["considered"]
            totals["links"] = counts["links"]
            totals["promoted"] = counts["promoted"]
            verb = "would write" if cfg.dry_run else "wrote"
            log(
                f"tagger: mapped {counts['considered']}, {verb} "
                f"{counts['links']} links"
                + (
                    f", {counts['promoted']} promoted by prompt"
                    if counts["promoted"]
                    else ""
                )
            )

        if not candidates and not targets:
            # Idle: let the model go rather than paying for it all day.
            release_model(log)
    finally:
        conn.close()

    totals["seconds"] = round(time.monotonic() - started, 1)
    return totals


def status(get_connection: Callable[[], Any]) -> dict[str, Any]:
    conn = get_connection()
    try:
        cur = conn.cursor()
        out: dict[str, Any] = {}
        for key, sql in (
            ("assets_inferred", "SELECT count(*) c FROM gz_tag_asset"),
            ("predictions", "SELECT count(*) c FROM gz_tag_prediction"),
            ("tags_applied", "SELECT count(*) c FROM gz_tag_applied"),
            (
                "assets_tagged",
                "SELECT count(DISTINCT asset_id) c FROM gz_tag_applied",
            ),
            ("deferred", "SELECT count(*) c FROM gz_tag_deferred"),
            (
                "gave_up",
                f"SELECT count(*) c FROM gz_tag_deferred "
                f"WHERE attempts >= {MAX_ATTEMPTS}",
            ),
            (
                "pending_map",
                "SELECT count(*) c FROM gz_tag_asset WHERE mapped_vocab IS NULL",
            ),
        ):
            cur.execute(sql)
            out[key] = cur.fetchone()["c"]
        out["model_version"] = get_state(cur, "model_version")
        out["vocabulary_version"] = get_state(cur, "vocabulary_version")
        out["model_loaded"] = _model is not None
        out["running"] = _pass_lock.locked()
        return out
    finally:
        conn.close()
