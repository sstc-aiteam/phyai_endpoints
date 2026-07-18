# SAM2 fp16 + torch.compile speedup

SAM2 inference in `SAM2ROISegmenter` (used by `WardObjectPipeline` /
`WardObjectPipelineService`) is sped up with fp16 autocast + `torch.compile`,
following [sam2-speedup](https://github.com/sstc-aiteam/sam2-speedup).
Enabled by default (`use_fp16=True, use_compile=True`); falls back to eager
fp16 automatically if `torch.compile` fails to warm up (e.g. Triton
misconfigured). See `README.md` for the Jetson env vars this needs.

## Full-pipeline (WardObjectPipelineService) timing summary

Tested on Jetson AGX Orin, running the full pipeline (RF-DETR detector → SAM2
fp16+compile → DINOv2 verification → classification fusion) against the 7
images under `images/`, with SAM2 kept warm across calls.

**One-time load**: 97.12s (RF-DETR weights + SAM2 fp16/compile warmup +
DINOv2 weights + reference DB with 1376 embeddings)

| Image | Detector | ROI+objects | SAM2 | Match+crop | DINOv2 | Fusion | **Total predict** | Result |
|---|---|---|---|---|---|---|---|---|
| cars.jpg | 3.040s* | 0.012s | 2.684s | 0.004s | 0.181s | 0.000s | **5.967s** | 7 objects |
| groceries.jpg | 0.089s | 0.002s | 2.746s | 0.025s | 0.582s | 0.000s | **3.452s** | 22 objects |
| rgb_20260701_123720 | 0.107s | 0.004s | 1.919s | 0.012s | 0.055s | 0.000s | **2.120s** | 2 objects |
| rgb_20260707_192013 | 0.090s | 0.002s | 2.621s | 0.006s | 0.041s | 0.000s | **2.783s** | 2 objects |
| rgb_20260707_220449 | 0.096s | 0.002s | 2.031s | 0.012s | 0.078s | 0.000s | **2.243s** | 4 objects |
| rgb_20260708_010855 | 0.093s | 0.003s | 3.567s | 0.018s | 0.074s | 0.000s | **3.778s** | 3 objects |
| truck.jpg | 0.155s | — | — | — | — | — | **0.170s** | failed: no chair_surface ROI |

\* cars.jpg's detector call includes a one-off ~2.9s CUDA/graph warmup on the
very first RF-DETR inference; every subsequent image drops to ~0.09–0.16s.

**Observations:**
- SAM2 dominates per-image latency (1.9–3.6s), consistent with standalone
  SAM2-only testing — the fp16+compile path stays warm and doesn't recompile
  between calls.
- DINOv2 verification is cheap (0.04–0.58s), scaling with object count.
- `truck.jpg` correctly short-circuits at 0.17s once no `chair_surface` ROI
  is found — no wasted SAM2/DINOv2 work.
- The three generic stock photos (cars/groceries/truck) still triggered a
  `chair_surface` detection on two of them (cars, groceries) — expected
  model behavior given RF-DETR was trained on the ward-item domain and
  wasn't given out-of-domain images to reject; not a pipeline bug.
- Steady-state per-image total (excluding the one-time load and first-call
  CUDA warmup): **~2.1–3.8s**.

This reflects the fp16+torch.compile path being actively engaged (confirmed
via the "Warming up SAM2 (fp16 + torch.compile)... / SAM2 warmup complete."
log lines with no eager-mode fallback triggered, and `autocast_dtype ==
torch.float16`), not the original eager fp32 baseline. No side-by-side
before/after comparison has been run yet.
