"""Tagger backends.

Two vocabularies because there are genuinely two kinds of library:

  WDTagger   Danbooru vocabulary. Built for illustration and AI renders.
             Official ONNX exports, published decision thresholds, honest
             per-label sigmoid confidences.

  RamPlus    General/photographic vocabulary (~6400 common categories).
             For anything that came out of a camera.

Both expose the same interface, so the survey and the worker don't care
which is which:

    tagger.name            -> "wd-vit-tagger-v3"
    tagger.default_thresholds -> {"general": 0.37, ...}
    tagger.predict(images) -> [[Prediction, ...], ...]

A Prediction's confidence may be None for backends that don't expose one.
The report handles that; don't fabricate a number to fill the hole.
"""

from __future__ import annotations

import csv
import sys
import io
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class Prediction:
    label: str
    confidence: float | None
    category: str  # "general" | "character" | "rating" | "style" | ...


class Tagger(Protocol):
    name: str
    default_thresholds: dict[str, float]

    def predict(self, images: Sequence[Image.Image]) -> list[list[Prediction]]: ...


# ---------------------------------------------------------------------------
# WD / Danbooru family
# ---------------------------------------------------------------------------


class WDTagger:
    """SmilingWolf's WaifuDiffusion taggers via ONNX Runtime.

    Preprocessing is fiddly and easy to get subtly wrong, which shows up as
    plausible-but-mediocre output rather than an error. These models want:

      - pad to square with WHITE, not black, and not a stretched resize
      - resize to the model's own input size (read from the graph)
      - BGR channel order, not RGB
      - raw 0-255 float32, with NO mean/std normalisation
      - NHWC layout

    Get any of those wrong and it still runs.
    """

    # Published decision thresholds (the equal-precision-and-recall point).
    # Overridable from config — these are starting points, not truth.
    KNOWN_THRESHOLDS = {
        "SmilingWolf/wd-vit-tagger-v3": {"general": 0.3710, "character": 0.75},
        "SmilingWolf/wd-vit-large-tagger-v3": {"general": 0.2606, "character": 0.75},
        "SmilingWolf/wd-eva02-large-tagger-v3": {"general": 0.3000, "character": 0.75},
    }

    CATEGORY_BY_ID = {0: "general", 4: "character", 9: "rating"}

    def __init__(
        self,
        repo_id: str = "SmilingWolf/wd-vit-tagger-v3",
        cache_dir: Path | None = None,
        providers: Sequence[str] | None = None,
    ) -> None:
        import onnxruntime as ort
        from huggingface_hub import hf_hub_download

        self.repo_id = repo_id
        self.name = repo_id.split("/")[-1]
        self.default_thresholds = dict(
            self.KNOWN_THRESHOLDS.get(repo_id, {"general": 0.35, "character": 0.75})
        )

        kw = {"repo_id": repo_id}
        if cache_dir:
            kw["cache_dir"] = str(cache_dir)

        def fetch(filename: str) -> str:
            """Cache first, network only if it isn't there.

            The default behaviour calls Hugging Face on every load to check
            for a newer revision, which makes each worker run depend on HF
            being reachable and emits the unauthenticated-requests warning.
            For a frozen model that check buys nothing: if the file is in
            the cache, use it.
            """
            try:
                return hf_hub_download(
                    filename=filename, local_files_only=True, **kw
                )
            except Exception:
                pass
            print(
                f"Fetching {filename} for {repo_id} (first run only)...",
                file=sys.stderr,
            )
            return hf_hub_download(filename=filename, **kw)

        model_path = fetch("model.onnx")
        tags_path = fetch("selected_tags.csv")

        if providers is None:
            available = ort.get_available_providers()
            providers = [
                p
                for p in ("CUDAExecutionProvider", "CPUExecutionProvider")
                if p in available
            ] or ["CPUExecutionProvider"]

        self.session = ort.InferenceSession(model_path, providers=list(providers))
        self.provider = self.session.get_providers()[0]

        # Input size comes from the graph: NHWC, square.
        shape = self.session.get_inputs()[0].shape
        self.input_size = int(shape[1]) if isinstance(shape[1], int) else 448

        with open(tags_path, newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        self.labels = [r["name"] for r in rows]
        self.categories = [
            self.CATEGORY_BY_ID.get(int(r["category"]), "general") for r in rows
        ]

    def _prepare(self, image: Image.Image) -> np.ndarray:
        image = image.convert("RGB")

        # Pad to square on white, preserving aspect.
        side = max(image.size)
        canvas = Image.new("RGB", (side, side), (255, 255, 255))
        canvas.paste(image, ((side - image.width) // 2, (side - image.height) // 2))

        if side != self.input_size:
            canvas = canvas.resize(
                (self.input_size, self.input_size), Image.Resampling.BICUBIC
            )

        array = np.asarray(canvas, dtype=np.float32)  # HWC, RGB, 0-255
        return array[:, :, ::-1]  # -> BGR, still 0-255, no normalisation

    def predict(self, images: Sequence[Image.Image]) -> list[list[Prediction]]:
        if not images:
            return []

        batch = np.stack([self._prepare(im) for im in images])  # NHWC
        input_name = self.session.get_inputs()[0].name
        logits = self.session.run(None, {input_name: batch})[0]

        # These graphs already end in a sigmoid; if a variant doesn't, the
        # values fall outside [0,1] and we apply one.
        if logits.min() < 0.0 or logits.max() > 1.0:
            logits = 1.0 / (1.0 + np.exp(-logits))

        out: list[list[Prediction]] = []
        for row in logits:
            out.append(
                [
                    Prediction(
                        label=self.labels[i],
                        confidence=float(row[i]),
                        category=self.categories[i],
                    )
                    for i in range(len(self.labels))
                ]
            )
        return out


# ---------------------------------------------------------------------------
# RAM++
# ---------------------------------------------------------------------------


# Symbols that lived in transformers.modeling_utils until ~4.12 and now live
# in transformers.pytorch_utils. RAM vendors BLIP's BERT implementation, which
# still imports the old paths, so on any modern transformers the import fails
# with "cannot import name 'apply_chunking_to_forward'".
#
# Aliasing them back is less brittle than pinning an old transformers — which
# in any case has no wheels for recent Pythons.
_MOVED_TO_PYTORCH_UTILS = (
    "apply_chunking_to_forward",
    "find_pruneable_heads_and_indices",
    "prune_linear_layer",
    "prune_conv1d_layer",
    "prune_layer",
    "Conv1D",
)


def _apply_chunking_to_forward(forward_fn, chunk_size, chunk_dim, *input_tensors):
    """Vendored from transformers, which dropped it from the public API.

    This one genuinely runs during inference — BERT's feed-forward block
    calls it — so it needs a real implementation rather than a stub. With
    chunk_size 0, which is RAM's configuration, it is just a direct call.
    """
    if not input_tensors:
        raise ValueError("expected at least one input tensor")

    if chunk_size > 0:
        first = input_tensors[0].shape[chunk_dim]
        if any(t.shape[chunk_dim] != first for t in input_tensors):
            raise ValueError("chunked tensors disagree on the chunk dimension")
        if first % chunk_size != 0:
            raise ValueError(
                f"chunk dimension {first} is not a multiple of chunk size {chunk_size}"
            )
        num_chunks = first // chunk_size
        chunked_inputs = tuple(
            t.chunk(num_chunks, dim=chunk_dim) for t in input_tensors
        )
        import torch

        return torch.cat(
            [forward_fn(*parts) for parts in zip(*chunked_inputs)], dim=chunk_dim
        )

    return forward_fn(*input_tensors)


def _unavailable(name: str):
    """A stub for head pruning, which inference never reaches.

    RAM's vendored BERT imports these at module scope but only calls them
    from `prune_heads()`. Satisfying the import is enough to load the model;
    raising rather than guessing an implementation means that if some path
    does reach one, it fails loudly instead of returning a wrong tensor.
    """

    def stub(*_args, **_kwargs):
        raise NotImplementedError(
            f"{name} is not available in this transformers version and is "
            "only needed for head pruning, not inference."
        )

    return stub


def patch_transformers_for_ram() -> list[str]:
    """Makes RAM's vendored BERT importable against modern transformers.

    Three generations of the library to cope with: the symbols used to be in
    `modeling_utils`, moved to `pytorch_utils`, and in transformers 5.x were
    removed altogether. So this tries the alias first and supplies the
    symbol itself when there's nothing left to alias.

    Returns what it touched, each marked `=name` for an alias or `+name` for
    a supplied implementation, so the caller can report honestly which of
    the two happened.
    """
    import transformers.modeling_utils as modeling_utils

    try:
        import transformers.pytorch_utils as pytorch_utils
    except Exception:
        pytorch_utils = None  # type: ignore[assignment]

    touched: list[str] = []
    for name in _MOVED_TO_PYTORCH_UTILS:
        if hasattr(modeling_utils, name):
            continue

        source = getattr(pytorch_utils, name, None) if pytorch_utils else None
        if source is not None:
            setattr(modeling_utils, name, source)
            touched.append(f"={name}")
            continue

        if name == "apply_chunking_to_forward":
            replacement = _apply_chunking_to_forward
        else:
            replacement = _unavailable(name)
        setattr(modeling_utils, name, replacement)
        if pytorch_utils is not None and not hasattr(pytorch_utils, name):
            # RAM's BERT may import from either module depending on version.
            setattr(pytorch_utils, name, replacement)
        touched.append(f"+{name}")

    return touched


class RamPlusTagger:
    """Recognize Anything Plus, for photographic content.

    Needs the authors' package and checkpoint, which are not on PyPI:

        pip install torch torchvision timm fairscale
        pip install git+https://github.com/xinyu1205/recognize-anything.git
        # checkpoint: ram_plus_swin_large_14m.pth from the model zoo

    Confidence: the documented entry point returns tag strings only. This
    reads the tagging head's logits directly, which depends on repo
    internals — if that breaks on a newer version, `confidence` comes back
    None and frequency analysis still works. Better a missing number than a
    fabricated one.
    """

    def __init__(
        self,
        checkpoint: Path,
        image_size: int = 384,
        device: str | None = None,
        threshold: float = 0.68,
    ) -> None:
        import torch

        # Must happen before `ram` is imported, since the failing import is
        # at module load time in its vendored BERT.
        patch_transformers_for_ram()

        from ram.models import ram_plus

        self.torch = torch
        self.name = "ram-plus-swin-large-14m"
        self.default_thresholds = {"general": threshold}
        self.image_size = image_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.model = ram_plus(
            pretrained=str(checkpoint), image_size=image_size, vit="swin_l"
        )
        self.model.eval()
        self.model = self.model.to(self.device)

        # Label list lives on the model; attribute name has moved between
        # releases, so look for either.
        labels = getattr(self.model, "tag_list", None)
        if labels is None:
            labels = getattr(self.model, "tag_list_english", [])
        self.labels = [
            t.strip() if isinstance(t, str) else str(t).strip() for t in labels
        ]

    def _prepare(self, image: Image.Image):
        from torchvision import transforms

        if not hasattr(self, "_tf"):
            self._tf = transforms.Compose(
                [
                    transforms.Resize((self.image_size, self.image_size)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
                    ),
                ]
            )
        return self._tf(image.convert("RGB"))

    def predict(self, images: Sequence[Image.Image]) -> list[list[Prediction]]:
        if not images:
            return []

        torch = self.torch
        batch = torch.stack([self._prepare(im) for im in images]).to(self.device)

        with torch.no_grad():
            probs = self._logits(batch)

        if probs is None:
            # Fall back to the documented path: labels, no confidences.
            from ram import inference_ram

            out = []
            for image in images:
                tensor = self._prepare(image).unsqueeze(0).to(self.device)
                tags = inference_ram(tensor, self.model)
                names = [t.strip() for t in str(tags[0]).split("|") if t.strip()]
                out.append(
                    [Prediction(n, None, "general") for n in names]
                )
            return out

        return [
            [
                Prediction(self.labels[i], float(row[i]), "general")
                for i in range(min(len(self.labels), len(row)))
            ]
            for row in probs.cpu().numpy()
        ]

    def _logits(self, batch):
        """Per-label probabilities, or None if the internals have moved."""
        torch = self.torch
        try:
            features = self.model.image_proj(self.model.visual_encoder(batch))
            embeddings = self.model.label_embed.weight.unsqueeze(0).repeat(
                batch.shape[0], 1, 1
            )
            hidden = self.model.tagging_head(
                encoder_embeds=embeddings,
                encoder_hidden_states=features,
                mode="tagging",
            ).last_hidden_state
            return torch.sigmoid(self.model.fc(hidden).squeeze(-1))
        except Exception:
            return None


# ---------------------------------------------------------------------------


def above_threshold(
    predictions: Sequence[Prediction], thresholds: dict[str, float]
) -> list[Prediction]:
    """Keeps predictions clearing their own category's threshold.

    Predictions with no confidence are kept: the backend has already made
    the decision, so there is nothing here to second-guess.
    """
    out = []
    for p in predictions:
        if p.confidence is None:
            out.append(p)
            continue
        limit = thresholds.get(p.category, thresholds.get("general", 0.35))
        if p.confidence >= limit:
            out.append(p)
    return sorted(out, key=lambda p: (-(p.confidence or 1.0), p.label))
