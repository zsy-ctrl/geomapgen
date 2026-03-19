import argparse
import json
from pathlib import Path

from geo_current_dataset_v1_common import build_manifest_for_dataset, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build val-only family manifest for rollout/inference validation.")
    parser.add_argument("--dataset-root", type=str, default="/dataset/zsy/dataset-extracted")
    parser.add_argument("--output-manifest", type=str, required=True)
    parser.add_argument("--image-relpath", type=str, default="patch_tif/0.tif")
    parser.add_argument("--mask-relpath", type=str, default="patch_tif/0_edit_poly.tif")
    parser.add_argument("--lane-relpath", type=str, default="label_check_crop/Lane.geojson")
    parser.add_argument("--intersection-relpath", type=str, default="label_check_crop/Intersection.geojson")
    parser.add_argument("--mask-threshold", type=int, default=127)
    parser.add_argument("--tile-size-px", type=int, default=1024)
    parser.add_argument("--overlap-px", type=int, default=256)
    parser.add_argument("--keep-margin-px", type=int, default=128)
    parser.add_argument("--review-crop-pad-px", type=int, default=64)
    parser.add_argument("--tile-min-mask-ratio", type=float, default=0.02)
    parser.add_argument("--tile-min-mask-pixels", type=int, default=256)
    parser.add_argument("--tile-max-per-sample", type=int, default=0)
    parser.add_argument("--search-within-review-bbox", action="store_true")
    parser.add_argument("--fallback-to-all-if-empty", action="store_true")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_manifest = Path(args.output_manifest).resolve()
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    families = build_manifest_for_dataset(
        dataset_root=Path(args.dataset_root).resolve(),
        splits=["val"],
        image_relpath=str(args.image_relpath),
        mask_relpath=str(args.mask_relpath),
        lane_relpath=str(args.lane_relpath),
        intersection_relpath=str(args.intersection_relpath),
        mask_threshold=int(args.mask_threshold),
        tile_size_px=int(args.tile_size_px),
        overlap_px=int(args.overlap_px),
        keep_margin_px=int(args.keep_margin_px),
        review_crop_pad_px=int(args.review_crop_pad_px),
        tile_min_mask_ratio=float(args.tile_min_mask_ratio),
        tile_min_mask_pixels=int(args.tile_min_mask_pixels),
        tile_max_per_sample=int(args.tile_max_per_sample),
        search_within_review_bbox=bool(args.search_within_review_bbox),
        fallback_to_all_if_empty=bool(args.fallback_to_all_if_empty),
        max_samples_per_split=int(args.max_samples),
        shard_index=int(args.shard_index),
        num_shards=int(args.num_shards),
    )
    count = write_jsonl(output_manifest, families)
    summary = {
        "dataset_root": str(Path(args.dataset_root).resolve()),
        "output_manifest": str(output_manifest),
        "split": "val",
        "family_count": int(count),
        "tile_size_px": int(args.tile_size_px),
        "overlap_px": int(args.overlap_px),
        "keep_margin_px": int(args.keep_margin_px),
        "review_crop_pad_px": int(args.review_crop_pad_px),
        "shard_index": int(args.shard_index),
        "num_shards": int(args.num_shards),
    }
    with output_manifest.with_suffix(".summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"[GeoCurrentV1] built val manifest families={count}")
    print(f"[GeoCurrentV1] manifest={output_manifest}")


if __name__ == "__main__":
    main()
