"""Appearance embeddings of detection crops with DINOv2 (ADR 0003).

A crop is the detection's box cut from the upright photo (EXIF orientation applied, as the
boxes assume), padded to a square with a neutral colour. Padding rather than the processor's
centre crop keeps long, thin objects such as knives whole. The model's pooled output (the
normalized CLS token) is scaled to unit length, so a dot product of two embeddings is their
cosine similarity.

The model is loaded from the official Hugging Face repository in safetensors format only.
Photos are processed locally and never sent anywhere.
"""

import re
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

import numpy as np
import numpy.typing as npt
from PIL import Image, ImageOps

from objhist_core import BoundingBox
from objhist_tracking.embedding_store import EMBEDDING_DTYPE

if TYPE_CHECKING:
    from transformers import BaseImageProcessor, PreTrainedModel

DINOV2_MODEL_ID = "facebook/dinov2-small"
PADDING_RGB = (124, 116, 104)
"""ImageNet mean colour, which the model's normalization maps to about zero."""
EMBEDDING_BATCH_SIZE = 32
COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}")


class AppearanceError(RuntimeError):
    """Embeddings cannot be computed as asked."""


class Embedder(Protocol):
    """Turns crops into unit-length embeddings; tests substitute a stub for the real model."""

    @property
    def model_name(self) -> str:
        """Model ID and exact revision, recorded with every embedding."""
        ...

    def embed(self, crops: Sequence[Image.Image]) -> npt.NDArray[np.float32]:
        """One unit-length float32 row per crop, in order."""
        ...


def load_upright_photo(path: Path) -> Image.Image:
    """The photo rotated by its EXIF orientation, in which detection boxes are expressed."""
    with Image.open(path) as image:
        return ImageOps.exif_transpose(image).convert("RGB")


def square_crop(photo: Image.Image, box: BoundingBox) -> Image.Image:
    """The box's contents centred on a square of `PADDING_RGB`."""
    left, top = int(box.x_min), int(box.y_min)
    right = max(left + 1, min(photo.width, round(box.x_max)))
    bottom = max(top + 1, min(photo.height, round(box.y_max)))
    crop = photo.crop((left, top, right, bottom))
    side = max(crop.width, crop.height)
    square = Image.new("RGB", (side, side), PADDING_RGB)
    square.paste(crop, ((side - crop.width) // 2, (side - crop.height) // 2))
    return square


def download_model(model_id: str) -> tuple[Path, str]:
    """The local snapshot of a Hugging Face model and its exact commit.

    Only configuration files and safetensors weights are fetched, never pickle-based
    checkpoints that run code when loaded. The download is bounded by the hub's timeout and
    skipped when the snapshot is already cached.
    """
    from huggingface_hub import snapshot_download

    snapshot = Path(snapshot_download(model_id, allow_patterns=["*.json", "*.safetensors"]))
    # The hub stores each snapshot in a folder named after its commit.
    revision = snapshot.name
    if not COMMIT_PATTERN.fullmatch(revision):
        raise AppearanceError(f"cannot tell which revision of {model_id} was downloaded")
    if not (snapshot / "model.safetensors").is_file():
        raise AppearanceError(f"{model_id} has no safetensors weights at {snapshot}")
    return snapshot, revision


class Dinov2Embedder:
    """DINOv2 from Hugging Face, loaded from a local snapshot of the official repository.
    Needs PyTorch from the workspace's `cpu` or `cu130` extra."""

    def __init__(
        self,
        model_id: str = DINOV2_MODEL_ID,
        device: str = "cpu",
        batch_size: int = EMBEDDING_BATCH_SIZE,
    ) -> None:
        # Imported here so the tracker and its tests never load PyTorch.
        import torch
        from transformers import AutoImageProcessor, AutoModel

        snapshot, revision = download_model(model_id)
        self._torch = torch
        self._device = torch.device(device)
        self._batch_size = batch_size
        self._processor: BaseImageProcessor = AutoImageProcessor.from_pretrained(snapshot)
        model: PreTrainedModel = AutoModel.from_pretrained(snapshot, use_safetensors=True)
        self._model = model.to(self._device).eval()
        self._model_name = f"{model_id}@{revision}"

    @property
    def model_name(self) -> str:
        return self._model_name

    def embed(self, crops: Sequence[Image.Image]) -> npt.NDArray[np.float32]:
        if not crops:
            raise AppearanceError("no crops to embed")
        batches: list[npt.NDArray[np.float32]] = []
        with self._torch.inference_mode():
            for start in range(0, len(crops), self._batch_size):
                batch = list(crops[start : start + self._batch_size])
                inputs = self._processor(images=batch, return_tensors="pt").to(self._device)
                pooled = self._model(**inputs).pooler_output
                unit = self._torch.nn.functional.normalize(pooled, dim=1)
                batches.append(unit.cpu().numpy().astype(EMBEDDING_DTYPE))
        return np.concatenate(batches)
