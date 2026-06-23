# zssim

**Perceptual image similarity for Zig.** A clean-room, from-scratch
implementation of a multi-scale, L\*a\*b\*-based structural-similarity metric,
built independently from the public research literature and designed to be
embedded in [ImProc](../ImProc).

> **Status:** early development. See the [Roadmap](#roadmap) and
> [Progress log](#progress-log).

It returns two numbers per image pair:

- **SSIM** in `0..1` (1.0 = identical) — the structural similarity index, and
- **distance** = `1/SSIM − 1` in `0..∞` (0.0 = identical, larger = more
  different; unbounded) — a convenient perceptual distance derived from SSIM.

---

## Why this exists / legal note

`zssim` implements image-quality algorithms that come from public, peer-reviewed
papers and open standards. Those algorithms are **not copyrightable** — only a
particular implementation is.

`zssim` is therefore written **clean-room**: from the papers and standards listed
below, with its own architecture, data structures, and optimizations. No source
code is copied or translated from any other implementation. It is MIT-licensed
(see [`LICENSE`](LICENSE)), so it can be embedded freely.

### References (all public)

- Z. Wang, A. C. Bovik, H. R. Sheikh, E. P. Simoncelli, *"Image Quality
  Assessment: From Error Visibility to Structural Similarity"*, IEEE TIP, 2004.
  (SSIM)
- Z. Wang, E. P. Simoncelli, A. C. Bovik, *"Multi-Scale Structural Similarity for
  Image Quality Assessment"*, IEEE Asilomar, 2003. (MS-SSIM + scale weights)
- Z. Wang, Q. Li, *"Information Content Weighting for Perceptual Image Quality
  Assessment"*, IEEE TIP, 2011. (IW-SSIM — inspiration for weighted scales)
- CIE 15:2004, *Colorimetry* — CIE 1976 L\*a\*b\*.
- IEC 61966-2-1 — sRGB transfer function and primaries.
- ITU-R BT.2100 — PQ (SMPTE ST 2084) and HLG transfer functions; ICtCp.
- SMPTE ST 2065-1 — ACES2065-1 / AP0 primaries.
- Bruce Lindbloom / SMPTE RP 177 — RGB↔XYZ matrix derivation and L\*a\*b\*
  continuity constants (public reference values).

---

## Algorithm overview

The pipeline is a multi-scale, color-aware SSIM:

1. **Decode → linear light.** Input pixels (8/16-bit integer or float) are
   converted to linear-light RGB in `f32`. The transfer function is a parameter
   (sRGB by default; also `linear`, `gamma(g)`, PQ, HLG).
2. **Multi-scale pyramid.** The linear image is repeatedly halved by 2×2-box
   averaging (done in linear light, which is physically correct for viewing
   distance / lens blur). Default: up to 5 scales.
3. **Perceptual encode.** Each scale is converted to a perceptually-uniform,
   channel-separated form:
   - **SDR** → CIE L\*a\*b\* (using the working-space primaries → XYZ matrix),
     normalized to `~[0,1]`.
   - **HDR / wide-gamut** (PQ/HLG transfers, or scene-referred input) → **ICtCp**,
     which stays perceptually uniform across a high dynamic range where CIELAB's
     SDR cube-root would break down.
   Chroma channels are optionally blurred (the eye has lower spatial acuity for
   color than luminance).
4. **Per-channel SSIM.** For each channel and scale, compute local statistics
   with a separable Gaussian window:
   `μ = G∗I`, `σ² = G∗I² − μ²`, `σ₁₂ = G∗(I₁I₂) − μ₁μ₂`, then
   `SSIM = (2μ₁μ₂+C₁)(2σ₁₂+C₂) / ((μ₁²+μ₂²+C₁)(σ₁²+σ₂²+C₂))`,
   with `C₁=(0.01)²`, `C₂=(0.03)²`.
5. **Pool + combine.** Per-scale SSIM is pooled (mean), channels combined with
   perceptual weights (luma-weighted), and scales combined with the published
   MS-SSIM-style weights.
6. **Output.** `SSIM` and `distance = 1/SSIM − 1`, plus (planned) per-scale SSIM
   maps for visualization.

### Color management: division of labor (important)

zssim does **not** parse ICC profiles and needs **no color-management library**.
This fits ImProc cleanly and avoids duplicating its color stack:

- **The host (ImProc) owns ICC.** ImProc already has a full lcms2 engine
  (`improc_lcms_transform_rgba16`) that converts any embedded source profile —
  sRGB, Display P3, Rec.2020, Adobe RGB, CMYK, N-color — to a *known working
  space* in its native 16-bit RGBA buffer.
- **zssim consumes a known working space.** The caller tags the buffer with its
  `{transfer, primaries, scene_referred}` and zssim does the correct
  linearization and perceptual encoding from built-in constant tables. No ICC
  blobs ever reach zssim.

**Wide gamut & HDR.** Primaries are derived at comptime from raw chromaticities
with Bradford chromatic adaptation to D65, so non-D65 white points are handled.
Presets include sRGB, Display P3, Rec.2020, Adobe RGB, and **ACES AP0 / ACEScg
AP1**. For *unbounded scene-linear* spaces (e.g. ACES2065-1), set
`scene_referred = true` to route through the ICtCp encoder; for the cleanest
results, apply a view/output transform in the host first (so highlights are
mapped to display range) and compare the result.

---

## Usage (planned API)

```zig
const zssim = @import("zssim");

var ctx = try zssim.Comparator.init(allocator, .{}); // default config
defer ctx.deinit();

// 8-bit sRGB RGBA:
const a = try ctx.prepareRgba8(pixels_a, w, h, .{});
defer a.deinit();

// ImProc's native 16-bit RGBA (top-down, straight alpha), tagged with its space:
const b = try ctx.prepareRgba16(pixels_b, w, h, .{ .transfer = .srgb, .primaries = .srgb });
defer b.deinit();

const result = try ctx.compare(a, b);
std.debug.print("distance={d:.6} ssim={d:.6}\n", .{ result.distance, result.ssim });
```

Supported inputs: `rgba8`, `rgb8`, `gray8`, `rgba16`, `rgb16`, `gray16`, and
`linear_rgb_f32`. The 16-bit RGBA path matches ImProc's internal format directly.

---

## Build & test

Requires Zig **0.16.0**.

```sh
zig build test     # run the unit test suite
zig build check    # fast type-check
```

Consume from another Zig project (e.g. ImProc) as a local path dependency:

```zig
// build.zig.zon
.dependencies = .{
    .zssim = .{ .path = "../zssim" },
},
```

```zig
// build.zig
const zssim = b.dependency("zssim", .{ .target = target, .optimize = optimize });
exe.root_module.addImport("zssim", zssim.module("zssim"));
```

---

## Roadmap

Legend: ✅ done · 🚧 in progress · ⬜ todo

- ✅ Repo scaffold, build, MIT license, clean-room notice, this roadmap
- ✅ **Color module** — sRGB/linear/gamma/PQ/HLG transfers; sRGB/P3/Rec.2020/
  Adobe-RGB/ACES-AP0/ACEScg primaries (comptime RGB→XYZ + Bradford to D65);
  CIE L\*a\*b\* and ICtCp encodings; `[0,1]` normalization
- ✅ **Separable Gaussian blur** — single-scratch two-pass, plus a fused
  `blur(a·b)` that never materializes the product
- ✅ **Pyramid** — linear-light 2×2 downsample with a min-size floor
- ✅ **Image preprocessing** — input adapters (8/16-bit RGBA/RGB/Gray + linear
  float), planar channel layout, per-channel `μ` and `blur(I²)` precompute
- ✅ **SSIM core + compare** — per-channel SSIM, mean pooling, channel weights,
  multi-scale weighted combine, `compare()` returning SSIM + distance
- ✅ **Grayscale fast path** — single-channel L\* pipeline
- ✅ **Tests** — identity = 0, symmetry, noise monotonicity, 16-bit↔8-bit
  equivalence, edge sizes (1×1, tiny), HDR/ICtCp sanity
- 🚧 **SIMD optimization** — done: per-image LUT linearization + `@Vector`
  blur interior. Todo: vectorize the SSIM combine
- ✅ **ImProc integration** — vendored as a git submodule; bridge
  (`src/imgcompare.zig`) + `improc --compare` CLI; validated on real images
- ⬜ **Optional threading** — per-scale / per-channel parallelism via a thread pool
- ⬜ **C ABI** — `zssim.h` + exported functions for non-Zig callers
- ⬜ **Difference visualization** — export per-scale SSIM heatmap
- ⬜ **Validation** — correlation check against a public IQA dataset (e.g. TID2013)

### Possible improvements to explore

- ICtCp HDR path for scene-referred / wide-gamut content.
- Exact-Gaussian SSIM window with configurable σ.
- Edge handling by weight renormalization (unbiased borders).
- SIMD CIELAB / ICtCp and a fast LUT linearization path.

---

## Progress log

- **2026-06-23** — Project created. Studied the public SSIM/MS-SSIM/CIELAB/ICtCp
  literature and decided the clean-room design: a parameterized
  `{transfer, primaries, scene_referred}` color front-end (so ICC stays in the
  host and HDR/wide-gamut/ACES work without a color library), a separable
  Gaussian SSIM core, and a luma-weighted multi-scale combine.
  Implemented and tested, in order: color science (transfers + comptime
  primaries/Bradford + CIELAB + ICtCp), separable Gaussian blur (with fused
  `blur(a·b)`), linear-light pyramid, the full preprocessing + `compare`
  pipeline, and the grayscale fast path. **32 tests passing**, including
  8-bit↔16-bit equivalence, symmetry, noise monotonicity, and HDR sanity.
- **2026-06-23** — Optimized: per-image transfer LUT (one table vs a `pow()`
  per pixel) and `@Vector` SIMD for the separable-blur interior (validated
  against a scalar reference). **33 tests passing**.
- **2026-06-23** — Integrated into ImProc as a git submodule
  (`vendor/zssim`). Added a bridge (`ImProc/src/imgcompare.zig`) mapping
  ImProc's decoded 16-bit RGBA `Image` to the comparator, and an
  `improc <a> --compare <b>` CLI flag. Validated end-to-end on real images
  (KADID-10k references, PNG/WebP round-trips): identical → distance 0,
  distinct images → larger distance, lossless round-trip → 0.
