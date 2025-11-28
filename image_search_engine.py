import glob
import hashlib
import logging
import os
from typing import Any, Dict, List

import numpy as np
from PIL import Image
from uform import Modality, get_model
from usearch.index import Index, MetricKind

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class ImageSearchEngine:
    """Indexes images from one or more directories and provides text and image search."""

    def __init__(self, image_dirs: List[str], supported_extensions: tuple[str, ...] = ("*.jpg", "*.jpeg", "*.png", "*.webp")):
        self.supported_extensions = supported_extensions
        self.managed_image_dirs = [os.path.abspath(d) for d in image_dirs if os.path.isdir(d)]
        if not self.managed_image_dirs:
            raise ValueError("At least one valid image directory is required.")

        logger.info("Loading UForm models...")
        self.processors, self.models = get_model("unum-cloud/uform3-image-text-multilingual-base")
        logger.info("Models loaded.")

        self.index: Index | None = None
        self.image_paths: Dict[int, str] = {}

        self._build_index()

    def _iter_image_paths(self):
        for directory in self.managed_image_dirs:
            for ext in self.supported_extensions:
                yield from glob.glob(os.path.join(directory, "**", ext), recursive=True)

    def _get_sha1_hash_as_int(self, path: str) -> int:
        hasher = hashlib.sha1()
        with open(path, "rb") as f:
            buf = f.read(65536)
            while buf:
                hasher.update(buf)
                buf = f.read(65536)
        return int.from_bytes(hasher.digest()[:8], "big")

    def _build_index(self):
        logger.info("Indexing images...")
        processor = self.processors[Modality.IMAGE_ENCODER]
        model = self.models[Modality.IMAGE_ENCODER]

        all_embeddings: list[np.ndarray] = []
        all_keys: list[int] = []

        for path in self._iter_image_paths():
            if os.path.getsize(path) <= 0:
                continue
            if not os.path.isfile(path):
                continue
            try:
                key = self._get_sha1_hash_as_int(path)
                if key in self.image_paths:
                    continue

                image = Image.open(path)
                image_data = processor(image)
                _, embedding = model.encode(image_data)

                # Ensure numpy array even if model returns torch tensors
                if hasattr(embedding, "detach"):  # torch.Tensor
                    embedding_np = embedding.detach().cpu().numpy()
                else:
                    embedding_np = np.array(embedding)

                embedding_flat = embedding_np.flatten().astype(np.float32)
                if self.index is None:
                    self.index = Index(ndim=embedding_flat.shape[0], metric=MetricKind.Cos)

                all_embeddings.append(embedding_flat)
                all_keys.append(key)
                self.image_paths[key] = path
            except Exception as exc:  # pragma: no cover - defensive logging
                logger.warning("Failed to process %s: %s", path, exc, exc_info=True)

        if all_embeddings:
            if self.index is None:
                self.index = Index(ndim=all_embeddings[0].shape[0], metric=MetricKind.Cos)

            self.index.add(
                keys=np.array(all_keys, dtype=np.uint64),
                vectors=np.vstack(all_embeddings),
                log=True,
            )
            logger.info("Indexed %s images.", len(all_keys))
        else:
            logger.warning("No images indexed. Check IMAGE_DIR content and extensions.")

    def _format_results(self, matches, score_threshold: float) -> List[Dict[str, Any]]:
        results = []
        for match in matches:
            path = self.get_image_path(match.key)
            score = 1 - match.distance
            if path and score >= score_threshold:
                results.append(
                    {
                        "url": f"/image/{int(match.key)}",
                        "filename": os.path.basename(path),
                        "score": f"{score:.2f}",
                        "key": int(match.key),
                    }
                )
        return results

    def search(self, query: str, count: int = 24, score_threshold: float = 0.0) -> List[Dict[str, Any]]:
        if not query or not self.index or len(self.index) == 0:
            return []
        processor = self.processors[Modality.TEXT_ENCODER]
        model = self.models[Modality.TEXT_ENCODER]
        _, embedding = model.encode(processor(query))
        matches = self.index.search(embedding.flatten(), count=count)
        return self._format_results(matches, score_threshold)

    def search_by_image(self, image: Image.Image, count: int = 24, score_threshold: float = 0.0) -> List[Dict[str, Any]]:
        if not self.index or len(self.index) == 0:
            return []
        processor = self.processors[Modality.IMAGE_ENCODER]
        model = self.models[Modality.IMAGE_ENCODER]
        image_data = processor(image)
        _, embedding = model.encode(image_data)
        matches = self.index.search(embedding.flatten(), count=count)
        return self._format_results(matches, score_threshold)

    def get_image_path(self, key: int) -> str | None:
        return self.image_paths.get(key)

    def list_image_directories(self) -> List[str]:
        return self.managed_image_dirs
