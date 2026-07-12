# dinov2_verifier.py

import os
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
from transformers import AutoImageProcessor, AutoModel


class DINOv2Verifier:
    def __init__(
        self,
        database_path,
        model_name="facebook/dinov2-base",
        device="cuda",
        similarity_threshold=0.75,
        image_extensions=(".jpg", ".jpeg", ".png", ".bmp", ".webp"),
        cache_embeddings=True,
        cache_filename="dinov2_reference_cache.pt",
    ):
        self.database_path = database_path
        self.model_name = model_name
        self.device = device
        self.similarity_threshold = similarity_threshold
        self.image_extensions = image_extensions
        self.cache_embeddings = cache_embeddings
        self.cache_filename = cache_filename

        print("Loading DINOv2 model...")

        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.model.eval()

        print("Loading DINOv2 reference database...")

        self.reference_embeddings, self.reference_labels, self.reference_paths = self._load_database(
            database_path
        )

        self.reference_embeddings = self.reference_embeddings.to(device)
        self.reference_embeddings = F.normalize(
            self.reference_embeddings,
            dim=1
        )

        print("DINOv2 database loaded.")
        print("Reference embeddings:", self.reference_embeddings.shape)
        print("Reference labels:", len(self.reference_labels))

    def embed_image(self, image):
        """
        Extract DINOv2 embedding from PIL image, numpy RGB image, or image path.
        """

        if isinstance(image, np.ndarray):
            image = Image.fromarray(
                image.astype(np.uint8)
            ).convert("RGB")

        elif isinstance(image, str):
            image = Image.open(image).convert("RGB")

        elif isinstance(image, Image.Image):
            image = image.convert("RGB")

        else:
            raise TypeError(
                f"Unsupported image type: {type(image)}"
            )

        inputs = self.processor(
            images=image,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

            embedding = outputs.last_hidden_state[:, 0, :]

            embedding = F.normalize(
                embedding,
                dim=1
            )

        return embedding

    def verify_image(self, image):
        """
        Compare one object crop against the DINOv2 database.
        """

        query_embedding = self.embed_image(image)

        similarities = torch.matmul(
            query_embedding,
            self.reference_embeddings.T
        ).squeeze(0)

        best_idx = int(torch.argmax(similarities).item())
        best_score = float(similarities[best_idx].item())
        best_label = self.reference_labels[best_idx]
        best_path = self.reference_paths[best_idx]

        accepted = best_score >= self.similarity_threshold

        return {
            "dinov2_label": best_label,
            "dinov2_score": best_score,
            "dinov2_reference_path": best_path,
            "accepted": accepted
        }

    def verify_crops(self, crop_records):
        """
        Verify crop records.

        Priority:
            1. Use in-memory masked_crop if available.
            2. Fall back to masked_path if available.
        """

        updated = []

        for record in crop_records:

            if "masked_crop" in record and record["masked_crop"] is not None:
                crop_input = record["masked_crop"]

            elif "masked_path" in record and record["masked_path"] is not None:
                crop_input = record["masked_path"]

            else:
                raise ValueError(
                    f"No masked crop or masked path found for record {record.get('index')}"
                )

            result = self.verify_image(crop_input)

            merged = {
                **record,
                **result
            }

            updated.append(merged)

        return updated

    def _load_database(self, database_path):
        """
        Supported database formats:

        Format A:
            database.pt
            {
                "embeddings": Tensor[N, D],
                "labels": List[str],
                optional "paths": List[str]
            }

        Format B:
            folder with .pt embeddings:
            dinov2_database/
                class_name/
                    ref1.pt

        Format C:
            folder with images:
            dinov2_database/
                context/
                    class_name/
                        img1.png
                masked/
                    class_name/
                        img1.png

        If both context/ and masked/ exist, this will recursively load all images.
        """

        if os.path.isfile(database_path):
            return self._load_pt_database(database_path)

        if not os.path.isdir(database_path):
            raise FileNotFoundError(
                f"DINOv2 database path not found: {database_path}"
            )

        cache_path = os.path.join(
            database_path,
            self.cache_filename
        )

        if self.cache_embeddings and os.path.exists(cache_path):
            print(f"Loading cached DINOv2 embeddings: {cache_path}")
            return self._load_pt_database(cache_path)

        pt_files = self._find_files(
            database_path,
            extensions=(".pt",)
        )

        image_files = self._find_files(
            database_path,
            extensions=self.image_extensions
        )

        if len(pt_files) > 0:
            embeddings, labels, paths = self._load_pt_folder_database(
                pt_files=pt_files,
                root=database_path
            )

        elif len(image_files) > 0:
            embeddings, labels, paths = self._build_database_from_images(
                image_files=image_files,
                root=database_path
            )

        else:
            raise ValueError(
                f"No .pt embeddings or image files found in {database_path}"
            )

        if self.cache_embeddings:
            print(f"Saving DINOv2 cache: {cache_path}")

            torch.save(
                {
                    "embeddings": embeddings.cpu(),
                    "labels": labels,
                    "paths": paths
                },
                cache_path
            )

        return embeddings, labels, paths

    def _load_pt_database(self, path):
        data = torch.load(
            path,
            map_location="cpu"
        )

        if not isinstance(data, dict):
            raise ValueError(
                "Database .pt file must contain a dict."
            )

        if "embeddings" not in data or "labels" not in data:
            raise ValueError(
                "Database .pt must contain keys: embeddings, labels"
            )

        embeddings = data["embeddings"]
        labels = data["labels"]
        paths = data.get("paths", [""] * len(labels))

        if not torch.is_tensor(embeddings):
            embeddings = torch.tensor(
                embeddings,
                dtype=torch.float32
            )

        embeddings = embeddings.float()

        return embeddings, labels, paths

    def _load_pt_folder_database(self, pt_files, root):
        embeddings = []
        labels = []
        paths = []

        for path in pt_files:
            item = torch.load(
                path,
                map_location="cpu"
            )

            if isinstance(item, dict):
                if "embedding" in item:
                    emb = item["embedding"]
                elif "embeddings" in item:
                    emb = item["embeddings"]
                else:
                    continue
            else:
                emb = item

            if not torch.is_tensor(emb):
                emb = torch.tensor(
                    emb,
                    dtype=torch.float32
                )

            emb = emb.float()

            if emb.ndim == 1:
                emb = emb.unsqueeze(0)

            label = self._infer_label_from_path(
                path=path,
                root=root
            )

            for i in range(emb.shape[0]):
                embeddings.append(emb[i])
                labels.append(label)
                paths.append(path)

        if len(embeddings) == 0:
            raise ValueError(
                f"No usable .pt embeddings found in {root}"
            )

        embeddings = torch.stack(
            embeddings,
            dim=0
        )

        return embeddings, labels, paths

    def _build_database_from_images(self, image_files, root):
        embeddings = []
        labels = []
        paths = []

        print(f"Building DINOv2 embeddings from {len(image_files)} reference images...")

        for idx, image_path in enumerate(image_files):
            label = self._infer_label_from_path(
                path=image_path,
                root=root
            )

            try:
                embedding = self.embed_image(image_path).detach().cpu().squeeze(0)
            except Exception as e:
                print(f"Skipping failed image: {image_path}")
                print(e)
                continue

            embeddings.append(embedding)
            labels.append(label)
            paths.append(image_path)

            if (idx + 1) % 20 == 0 or (idx + 1) == len(image_files):
                print(f"Embedded [{idx + 1}/{len(image_files)}] images")

        if len(embeddings) == 0:
            raise ValueError(
                f"No usable image references found in {root}"
            )

        embeddings = torch.stack(
            embeddings,
            dim=0
        )

        return embeddings, labels, paths

    def _find_files(self, root, extensions):
        matched = []

        for dirpath, _, filenames in os.walk(root):
            for filename in filenames:
                lower = filename.lower()

                if lower.endswith(extensions):
                    matched.append(
                        os.path.join(dirpath, filename)
                    )

        return sorted(matched)

    def _infer_label_from_path(self, path, root):
        """
        Infer class label from path.

        Handles:
            dinov2_database/context/ac_remotecontrol/img.png
            dinov2_database/masked/ac_remotecontrol/img.png
            dinov2_database/ac_remotecontrol/img.png

        Returns:
            ac_remotecontrol
        """

        rel = os.path.relpath(path, root)
        parts = rel.split(os.sep)

        if len(parts) >= 3 and parts[0] in {"context", "masked"}:
            return parts[1]

        if len(parts) >= 2:
            return parts[0]

        return "unknown"
