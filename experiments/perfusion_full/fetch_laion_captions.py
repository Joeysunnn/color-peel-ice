"""Stream LAION-400M shard-0 caption column and save a reproducible 100K sample."""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
from pathlib import Path
import urllib.request


SOURCE_URL = (
    "https://deploy.laion.ai/8f83b608504d46bb81708ec86e912220/"
    "embeddings/metadata/metadata_0.parquet"
)


def priority(seed: int, caption: str) -> int:
    return int.from_bytes(hashlib.sha256(
        f"laion-shard0\0{seed}\0{caption}".encode("utf-8")
    ).digest(), "big")


def extract(output_dir: Path, seed: int = 42, count: int = 100000) -> dict:
    if count != 100000 or seed < 0:
        raise ValueError("paper covariance requires a seeded 100000-caption sample")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(output_dir)
    import fsspec
    import pyarrow.parquet as pq

    head = urllib.request.urlopen(urllib.request.Request(SOURCE_URL, method="HEAD"), timeout=30)
    source = {"url": SOURCE_URL, "content_length": int(head.headers["Content-Length"]),
              "etag": head.headers.get("ETag"), "last_modified": head.headers.get("Last-Modified")}
    heap: list[tuple[int, str]] = []
    retained: set[str] = set()
    scanned = 0
    with fsspec.open(SOURCE_URL, "rb", block_size=8 << 20).open() as handle:
        parquet = pq.ParquetFile(handle)
        if "caption" not in parquet.schema.names:
            raise ValueError("LAION metadata has no caption column")
        total_rows = parquet.metadata.num_rows
        for batch in parquet.iter_batches(batch_size=8192, columns=["caption"]):
            for raw in batch.column(0).to_pylist():
                scanned += 1
                if not isinstance(raw, str):
                    continue
                caption = raw.strip().replace("\r", " ").replace("\n", " ")
                if not caption or caption in retained:
                    continue
                rank = priority(seed, caption)
                if len(heap) == count:
                    if rank >= -heap[0][0]:
                        continue
                    _, removed = heapq.heapreplace(heap, (-rank, caption))
                    retained.remove(removed)
                else:
                    heapq.heappush(heap, (-rank, caption))
                retained.add(caption)
    if scanned != total_rows or len(heap) != count:
        raise ValueError(f"LAION shard scan incomplete: {scanned}/{total_rows}, selected {len(heap)}")
    selected = [caption for _, caption in sorted(heap, key=lambda row: -row[0])]
    output_dir.mkdir(parents=True)
    captions_file = output_dir / "laion_100k.jsonl"
    with captions_file.open("w", encoding="utf-8") as out:
        for caption in selected:
            out.write(json.dumps({"caption": caption}, ensure_ascii=False) + "\n")
    record = {
        "schema": "perfusion_full_laion_caption_sample/v1", "source": source,
        "source_rows_scanned": scanned, "selection": "smallest seeded SHA256 priority among unique captions",
        "seed": seed, "caption_count": count, "captions_sha256": hashlib.sha256(captions_file.read_bytes()).hexdigest(),
    }
    (output_dir / "provenance.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(json.dumps(extract(args.output_dir, args.seed), indent=2))
