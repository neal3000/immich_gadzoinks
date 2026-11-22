"""Using the generation prompt to lower thresholds, not to assert tags.

The prompt is what was asked for; the render is what came out. Diffusion
drops things — "buddha statues in background" can produce a blurry lump,
something that is not a buddha, or nothing. Tagging straight from the prompt
asserts those things with total confidence on an image that may not contain
them, and nothing later can detect the error because no evidence disagrees.

So the prompt is treated as a prior and the model as evidence, and a tag
needs both. A term the prompt mentions clears a lower bar; a term the model
saw nothing of is still not tagged however emphatically the prompt asked for
it. In practice that promotes exactly the cases worth promoting: `beach` and
`buddha` sitting at 0.2 in a render whose prompt names them.

A floor stops this going too far. Without one, a 0.06 prediction — noise —
could be promoted by a prompt mention.

Two things deliberately not done:

  * The negative prompt is never read. It names what was excluded, so
    mining it would tag an image with precisely what is absent. The
    gadzoinks sidecar keeps it in its own field, so this is simply a matter
    of not touching it.
  * Boilerplate is not stripped. There is no quality-token blocklist here
    because there does not need to be: prompt text is only ever matched
    *against vocabulary terms*, so `masterpiece, best quality, 8k` matches
    nothing and costs nothing. A stoplist would be one more thing to get
    wrong, and it would eventually strip a term that is in the vocabulary.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# `<lora:dragonStyle_v3:0.8>` and `<lyco:...>`: the filename is arbitrary and
# could coincidentally contain a vocabulary term, so these come out first.
LORA = re.compile(r"<[^>]*>")

# `(buddha:1.3)` -> `buddha`, `[beach:0.5]` -> `beach`, and bare `(term)`.
WEIGHTED = re.compile(r"[(\[]\s*([^():\[\]]+?)\s*(?::\s*[\d.]+\s*)?[)\]]")

# BREAK, AND and similar separators used by some front ends.
SEPARATORS = re.compile(r"\b(BREAK|AND)\b")

WORD = re.compile(r"[a-z0-9]+")


def singular(word: str) -> str:
    """Crudest useful stemming: enough to match plurals, and no more.

    Prompts are written in prose ("buddha statues in background") while the
    vocabulary is singular (`statue`), so without this the two never meet.
    A real stemmer would be worse: aggressive stemming collapses terms the
    vocabulary deliberately keeps apart.

    The `es` ending needs care. It drops whole after a sibilant (glasses ->
    glass, boxes -> box) but only the `s` elsewhere (statues -> statue).
    Taking `es` unconditionally gives `statu`, which matches nothing.
    """
    if len(word) < 4:
        return word
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"
    if word.endswith("es") and len(word) > 4:
        stem = word[:-2]
        if stem.endswith(("s", "x", "z", "ch", "sh")):
            return stem
        return word[:-1]
    if word.endswith("s") and not word.endswith("ss") and len(word) > 3:
        return word[:-1]
    return word


def normalise(text: str) -> str:
    """Prompt text reduced to lowercase words separated by single spaces."""
    text = LORA.sub(" ", text)
    # Repeatedly, because weights nest: `((buddha:1.2):0.8)`.
    for _ in range(4):
        new = WEIGHTED.sub(r"\1", text)
        if new == text:
            break
        text = new
    text = SEPARATORS.sub(" ", text)
    return " ".join(WORD.findall(text.lower()))


def term_words(term: str) -> list[str]:
    """`blonde_hair` and `blonde hair` both become ['blonde', 'hair']."""
    return WORD.findall(term.replace("_", " ").lower())


@dataclass
class PromptIndex:
    """One prompt, prepared for repeated term lookups."""

    text: str
    words: set[str]
    stems: set[str]

    @classmethod
    def build(cls, prompt: str | None) -> "PromptIndex | None":
        if not prompt:
            return None
        text = normalise(prompt)
        if not text:
            return None
        tokens = text.split()
        return cls(
            text=f" {text} ",
            words=set(tokens),
            stems={singular(w) for w in tokens},
        )

    def mentions(self, term: str) -> bool:
        """Does the prompt name this vocabulary term?

        Single words match on a word boundary, so `art` does not match
        `artstation`. Multi-word terms must appear as a contiguous phrase:
        `coral reef` matches "a coral reef at dawn" but not "coral earrings
        near a reef", where the two words are unrelated.
        """
        words = term_words(term)
        if not words:
            return False
        if len(words) == 1:
            word = words[0]
            return word in self.words or singular(word) in self.stems
        phrase = " ".join(words)
        if f" {phrase} " in self.text:
            return True
        # Try the phrase with its last word stemmed, so `coral reef` matches
        # "coral reefs". Only the last word: `coral reefs` is a plural reef,
        # not a plural coral.
        stemmed = " ".join(words[:-1] + [singular(words[-1])])
        if stemmed != phrase and f" {stemmed} " in self.text:
            return True
        # And the reverse: a singular prompt against a plural term.
        return any(
            f" {' '.join(words[:-1])} {w} " in self.text
            for w in (words[-1] + "s", words[-1] + "es")
        )


def boosted_thresholds(
    base: dict[str, float],
    index: PromptIndex | None,
    label: str,
    boost: float,
    floor: float,
) -> float:
    """The threshold for one label, given the prompt."""
    limit = base.get("general", 0.371)
    if index is None or not index.mentions(label):
        return limit
    return max(floor, limit - boost)
