# Geo Patch Supervision Reorder Mask Cut 20260313_134400

## Updated supervision order

The patch-level supervision path has been tightened to better match the intended training logic:

1. deterministic feature reordering
2. review-mask filtering on supervision geometry
3. patch/state-region clipping
4. cut metadata preservation for clipped local targets
5. quantized coordinate token serialization

## What changed

### 1. Stable target ordering

Raw feature records are now deterministically reordered before local patch target construction.

### 2. Image masking

During train/eval dataset preparation, pixels outside the trusted review mask are zeroed out inside the cropped patch image before resize/pad.

### 3. Line supervision path

Lane geometry is now prepared as:

- split by trusted mask first
- then clipped by patch bbox
- then clipped by state bbox when needed
- then resampled

If a line is broken by mask or crop, its local sequence carries cut metadata.

### 4. Polygon supervision path

Intersection polygons are now:

- checked against the trusted mask before local patch target construction
- clipped to patch/state bbox
- marked as partial when clipping occurs

### 5. Cut-token meaning

New internal cut markers are used when a local object is partial but the cut is not on a patch border:

- `cut_in=internal`
- `cut_out=internal`

This helps the model distinguish complete local objects from partial local fragments.

## Cache compatibility

Because the supervised local target construction logic changed, the default patch-cache namespace was bumped to:

- `geo_patch_cache_v2`

This prevents old cache entries from silently reusing the previous target-building path.
