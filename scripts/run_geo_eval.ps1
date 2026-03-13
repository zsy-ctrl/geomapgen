$Config = "configs/geo_vector_lora.yaml"
$Checkpoint = ""
$Split = "val"
$Output = "outputs/geo_eval_metrics.json"
$MaxSamples = "0"
$DatasetRoot = "/home/zsy/Downloads/dataset-extracted"
$DinoModelPath = ""
$QwenModelPath = ""
$Device = ""
$ImageSize = "448"
$TileSizePx = "1024"
$OverlapPx = "256"
$KeepMarginPx = "128"
$StateBorderMarginPx = "128"
$SampleIntervalMeter = "6.0"
$CoordBins = "1024"
$LaneMaxFeatures = "0"
$IntersectionMaxFeatures = "0"
$StateMaxFeatures = "32"
$StateAnchorMaxPoints = "6"
$PromptMaxTokens = "0"
$StateMaxTokens = "0"
$TargetMaxTokens = "0"
$MaxNewTokens = "0"
$MinNewTokens = "0"
$MaxPropTokens = "0"
$Temperature = "1.0"
$TopK = "1"
$RepetitionPenalty = "1.02"
$UseKvCache = "true"
$ArtifactExportEnabled = "true"
$SaveKeptPatches = "true"
$SaveDiscardedPatches = "true"
$SaveResizedPatchInputs = "true"
$SaveEvalSampleGeojson = "true"
$SavePredictTileGeojson = "true"
$ArtifactMaxPatchImagesPerSample = "0"

function Set-EnvIfValue([string]$Name, [string]$Value) {
    if ($null -ne $Value -and $Value -ne "") {
        Set-Item -Path ("Env:" + $Name) -Value $Value
    }
}

if (-not $Checkpoint) {
    throw "Checkpoint is required."
}

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = Split-Path -Parent $scriptDir
Set-Location $repoRoot

Set-EnvIfValue "UNIMAPGEN_GEO_ROOT" $DatasetRoot
Set-EnvIfValue "DINOV2_BACKBONE_PATH" $DinoModelPath
Set-EnvIfValue "QWEN_MODEL_PATH" $QwenModelPath
Set-EnvIfValue "UNIMAPGEN_DEVICE" $Device
Set-EnvIfValue "UNIMAPGEN_IMAGE_SIZE" $ImageSize
Set-EnvIfValue "UNIMAPGEN_TILE_SIZE_PX" $TileSizePx
Set-EnvIfValue "UNIMAPGEN_OVERLAP_PX" $OverlapPx
Set-EnvIfValue "UNIMAPGEN_KEEP_MARGIN_PX" $KeepMarginPx
Set-EnvIfValue "UNIMAPGEN_STATE_BORDER_MARGIN_PX" $StateBorderMarginPx
Set-EnvIfValue "UNIMAPGEN_SAMPLE_INTERVAL_METER" $SampleIntervalMeter
Set-EnvIfValue "UNIMAPGEN_COORD_BINS" $CoordBins
Set-EnvIfValue "UNIMAPGEN_LANE_MAX_FEATURES" $LaneMaxFeatures
Set-EnvIfValue "UNIMAPGEN_INTERSECTION_MAX_FEATURES" $IntersectionMaxFeatures
Set-EnvIfValue "UNIMAPGEN_STATE_MAX_FEATURES" $StateMaxFeatures
Set-EnvIfValue "UNIMAPGEN_STATE_ANCHOR_MAX_POINTS" $StateAnchorMaxPoints
Set-EnvIfValue "UNIMAPGEN_PROMPT_MAX_TOKENS" $PromptMaxTokens
Set-EnvIfValue "UNIMAPGEN_STATE_MAX_TOKENS" $StateMaxTokens
Set-EnvIfValue "UNIMAPGEN_TARGET_MAX_TOKENS" $TargetMaxTokens
Set-EnvIfValue "UNIMAPGEN_MAX_NEW_TOKENS" $MaxNewTokens
Set-EnvIfValue "UNIMAPGEN_MIN_NEW_TOKENS" $MinNewTokens
Set-EnvIfValue "UNIMAPGEN_MAX_PROP_TOKENS" $MaxPropTokens
Set-EnvIfValue "UNIMAPGEN_TEMPERATURE" $Temperature
Set-EnvIfValue "UNIMAPGEN_TOP_K" $TopK
Set-EnvIfValue "UNIMAPGEN_REPETITION_PENALTY" $RepetitionPenalty
Set-EnvIfValue "UNIMAPGEN_USE_KV_CACHE" $UseKvCache
Set-EnvIfValue "UNIMAPGEN_ARTIFACT_EXPORT_ENABLED" $ArtifactExportEnabled
Set-EnvIfValue "UNIMAPGEN_SAVE_KEPT_PATCHES" $SaveKeptPatches
Set-EnvIfValue "UNIMAPGEN_SAVE_DISCARDED_PATCHES" $SaveDiscardedPatches
Set-EnvIfValue "UNIMAPGEN_SAVE_RESIZED_PATCH_INPUTS" $SaveResizedPatchInputs
Set-EnvIfValue "UNIMAPGEN_SAVE_EVAL_SAMPLE_GEOJSON" $SaveEvalSampleGeojson
Set-EnvIfValue "UNIMAPGEN_SAVE_PREDICT_TILE_GEOJSON" $SavePredictTileGeojson
Set-EnvIfValue "UNIMAPGEN_ARTIFACT_MAX_PATCH_IMAGES_PER_SAMPLE" $ArtifactMaxPatchImagesPerSample

$args = @(
    "-m", "unimapgen.eval_geo_vector",
    "--config", $Config,
    "--checkpoint", $Checkpoint,
    "--split", $Split,
    "--output", $Output
)
if ($MaxSamples -ne "0") {
    $args += @("--max_samples", $MaxSamples)
}

python @args
