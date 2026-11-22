"""Vocabulary: which labels to keep, and what tag path each becomes.

The taggers emit thousands of labels. This turns that into the handful you
actually want, as hierarchical paths under a root (`ai/` by default).

Four operations, in the order they're applied:

  1. category blocklist  drop whole Danbooru categories — `rating` is
                         general/sensitive/questionable/explicit, which are
                         content ratings masquerading as content tags
  2. explicit rule       per-label: a path, or null to drop
  2b. fallback rule      a coarse label kept only when nothing more specific
                         landed under it — `jewelry` when no `earrings`
  3. whitelist           optional: keep only labels in an external list
  4. unlisted policy     everything with no rule: drop (default) or keep

Paths rather than names, because Danbooru implies parents: `earrings` and
`jewelry` both fire on the same image. `earrings: jewelry/earrings` gives one
tag path whose parent node still finds everything, instead of two flat tags
meaning nearly the same thing.

Several labels mapping to one path is normal and deliberate — that's how you
fold `blue_sky` into `sky`. Duplicates collapse.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

DEFAULT_ROOT = "ai"

# Why a label didn't make it, for the report.
REASONS = ("category", "explicit", "whitelist", "stopped", "blocked",
           "unlisted", "fallback")


@dataclass
class Vocabulary:
    root: str = DEFAULT_ROOT
    drop_categories: set[str] = field(default_factory=lambda: {"rating"})
    labels: dict[str, str | None] = field(default_factory=dict)
    fallbacks: dict[str, str] = field(default_factory=dict)
    unlisted: str = "drop"  # "drop" | "keep"
    use_rules: bool = True  # fall back to taxonomy.py morphology + stop list
    whitelist: set[str] | None = None

    # -- loading ----------------------------------------------------------

    @classmethod
    def load(cls, path: str | Path) -> "Vocabulary":
        import yaml

        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}

        whitelist = None
        wl_path = raw.get("whitelist")
        if wl_path:
            terms = Path(wl_path).read_text(encoding="utf-8").splitlines()
            whitelist = {normalise(t) for t in terms if t.strip()}

        unlisted = str(raw.get("unlisted", "drop")).lower()
        if unlisted not in ("drop", "keep"):
            raise ValueError(f"unlisted must be 'drop' or 'keep', got {unlisted!r}")

        return cls(
            root=raw.get("root", DEFAULT_ROOT),
            drop_categories=set(raw.get("drop_categories") or []),
            labels={str(k): v for k, v in (raw.get("labels") or {}).items()},
            fallbacks={
                str(k): str(v) for k, v in (raw.get("fallback_labels") or {}).items()
            },
            unlisted=unlisted,
            use_rules=bool(raw.get("use_rules", True)),
            whitelist=whitelist,
        )

    @classmethod
    def permissive(cls) -> "Vocabulary":
        """No curation at all — for before/after comparison in the report."""
        return cls(drop_categories=set(), labels={}, unlisted="keep",
                   use_rules=False)

    # -- applying ---------------------------------------------------------

    def _lookup(self, table: dict, label: str):
        """Matches a label against the file regardless of word separator.

        The published label spaces use spaces (`wine glass`, `traffic
        light`); Danbooru taggers use underscores (`wine_glass`). Without
        this, every multi-word entry in the vocabulary file silently never
        matched, and the term either fell through to the morphology rules
        or was dropped as unlisted. Hundreds of terms, no error.
        """
        if label in table:
            return True, table[label]
        spaced = label.replace("_", " ")
        if spaced in table:
            return True, table[spaced]
        under = label.replace(" ", "_")
        if under in table:
            return True, table[under]
        bare = spaced.split("(")[0].strip()
        if bare and bare in table:
            return True, table[bare]
        return False, None

    def resolve(self, label: str, category: str) -> tuple[str | None, str | None]:
        """Returns (tag path, None) when kept, or (None, reason) when dropped.

        Order matters. The file wins over the rules, so a hand-written entry
        is always an override; the rules then catch the regular cases the
        file doesn't enumerate. Without that fallback the file would have to
        list every `*_hair` colour and every sci-fi term by hand, and a
        vocabulary built from photography datasets would silently have no
        home for a library full of renders.
        """
        if category in self.drop_categories:
            return None, "category"

        hit, target = self._lookup(self.fallbacks, label)
        if hit:
            return self.join(str(target)), None

        hit, target = self._lookup(self.labels, label)
        if hit:
            if target is None:
                return None, "explicit"
            return self.join(str(target)), None

        if self.whitelist is not None and normalise(label) not in self.whitelist:
            return None, "whitelist"

        if self.use_rules:
            from gz_taxonomy import is_blocked, is_stop, structure

            # Non-subjects are dropped as such rather than reported as
            # unmapped, so the "needs a rule" list stays a work queue
            # instead of filling up with `1girl` and `looking_at_viewer`.
            if is_stop(label):
                return None, "stopped"
            blocked = is_blocked(label)
            if blocked:
                return None, "blocked"
            path = structure(label)
            if path:
                return self.join(path), None

        if self.unlisted == "drop":
            return None, "unlisted"

        return self.join(readable(label)), None

    def join(self, relative: str) -> str:
        relative = relative.strip("/")
        return f"{self.root}/{relative}" if self.root else relative

    def apply(
        self, predictions: Sequence[tuple[str, str]]
    ) -> tuple[list[str], Counter[str], list[str]]:
        """Maps (label, category) pairs to deduplicated tag paths.

        Returns the paths, a count of drops by reason, and the labels that
        fell through with no rule — the last of which is the useful bit: it's
        the ranked list of what to add to the file next.
        """
        paths: list[str] = []
        fallback_paths: list[str] = []
        dropped: Counter[str] = Counter()
        unmapped: list[str] = []

        for label, category in predictions:
            path, reason = self.resolve(label, category)
            if path is None:
                dropped[reason or "unknown"] += 1
                if reason in ("unlisted", "whitelist"):
                    unmapped.append(label)
                continue
            in_fallback, _ = self._lookup(self.fallbacks, label)
            target = fallback_paths if in_fallback else paths
            if path not in target:
                target.append(path)

        # A fallback only survives if nothing more specific landed under it.
        # `jewelry` is the case: Danbooru fires the parent alongside every
        # child, so without this an image tagged `earrings` also gets the
        # bare `jewelry` — the duplication the paths were meant to remove.
        for path in fallback_paths:
            prefix = path + "/"
            if any(p == path or p.startswith(prefix) for p in paths):
                dropped["fallback"] += 1
                continue
            paths.append(path)

        return paths, dropped, unmapped

    # -- writing ----------------------------------------------------------

    def ancestors(self, paths: Iterable[str]) -> list[str]:
        """Every path plus its parents.

        Immich's `upsert_tags` creates intermediate levels on its own, so this
        isn't needed to create the tree. It's here for the worker: if you want
        `ai/jewelry` to be directly searchable rather than only a container,
        the asset has to carry the parent too.
        """
        out: set[str] = set()
        for path in paths:
            parts = path.split("/")
            for i in range(1, len(parts) + 1):
                out.add("/".join(parts[:i]))
        # Drop the bare root: tagging everything `ai` is useless.
        out.discard(self.root)
        return sorted(out)


def normalise(term: str) -> str:
    """For matching across vocabularies: WD uses underscores, RAM spaces."""
    return term.replace("_", " ").split("(")[0].strip().lower()


def readable(label: str) -> str:
    """`blonde_hair` -> `blonde hair`, for labels kept without a rule."""
    return label.replace("_", " ").strip()
