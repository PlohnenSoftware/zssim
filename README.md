# zssim

**Perceptual image similarity for Zig.** A clean-room, from-scratch implementation
of a multi-scale, L\*a\*b\*-based structural-similarity metric — a spiritual
sibling of [distance](), built independently from
the public research literature and designed to be embedded in
[ImProc](../ImProc).

> **Status:** early development. See the [Roadmap](#roadmap) and
> [Progress log](#progress-log).

It returns a single number per image pair:

- **SSIM** in `0..1` (1.0 = identical), and
- **distance** = `1/SSIM − 1` in `0..∞` (0.0 = identical, larger = more different).

The distance scale matches the convention used by the original distance tool, so values
are interpreted the same way (though not bit-identical — this is a reinterpretation,
not a port).

---

## Why this exists / legal note

The original `distance` is excellent but is **differently licensed**, so
its *code* cannot be reused in a permissively-licensed project. The *algorithms*
it uses, however, come from public peer-reviewed papers and open standards and are
**not copyrightable** — only a particular implementation is.

`zssim` is therefore written **clean-room**: from the papers and standards listed
below, with its own architecture, data structures, and optimizations. No code is
copied or translated from distance (or any other implementation). See
[`LICENSE`](LICENSE) for the clean-room notice. zssim is MIT-licensed.

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
- Bruce Lindbloom, RGB/XYZ matrices and L\*a\*b\* continuity constants
  (public reference values).

---

## Algorithm overview

The pipeline mirrors the well-known multi-scale, color-aware SSIM approach:

1. **Decode → linear light.** Input pixels (8/16-bit integer or float) are
   converted to linear-light RGB in `f32`. The transfer function is a parameter
   (sRGB by default; also `linear`, `gamma(g)`, PQ, HLG).
2. **Multi-scale pyramid.** The linear image is repeatedly halved by 2×2-box
   averaging (done in linear light, which is physically correct for viewing
   distance / lens blur). Default: 5 scales.
3. **Perceptual encode.** Each scale is converted to a perceptually-uniform,
   channel-separated form:
   - **SDR** → CIE L\*a\*b\* (using the working-space primaries → XYZ matrix),
     normalized to `~[0,1]`.
   - **HDR / wide-gamut** (PQ/HLG transfers) → **ICtCp**, which stays perceptually
     uniform across a high dynamic range where CIELAB's SDR cube-root would
     break down.
   Chroma channels are optionally blurred (the eye has lower spatial acuity for
   color than luminance).
4. **Per-channel SSIM.** For each channel and scale, compute local statistics
   with a separable Gaussian window:
   `μ = G∗I`, `σ² = G∗I² − μ²`, `σ₁₂ = G∗(I₁I₂) − μ₁μ₂`, then
   `SSIM = (2μ₁μ₂+C₁)(2σ₁₂+C₂) / ((μ₁²+μ₂²+C₁)(σ₁²+σ₂²+C₂))`,
   with `C₁=(0.01)²`, `C₂=(0.03)²`.
5. **Pool + combine.** Per-scale SSIM maps are pooled (mean or mean-absolute-
   deviation), channels combined with perceptual weights (luma-weighted), and
   scales combined with the published MS-SSIM-style weights.
6. **Output.** `SSIM` and `distance = 1/SSIM − 1`, plus optional per-scale SSIM maps.

### Color management: division of labor (important)

zssim does **not** parse ICC profiles and needs **no color-management library**.
This is the same separation the original distance uses (it relies on its decoder to
flatten profiles to sRGB), and it is exactly what fits ImProc:

- **The host (ImProc) owns ICC.** ImProc already has a full lcms2 engine
  (`improc_lcms_transform_rgba16`) that converts any embedded source profile —
  sRGB, Display P3, Rec.2020, Adobe RGB, CMYK, N-color — to a *known working
  space* in its native 16-bit RGBA buffer.
- **zssim consumes a known working space.** The caller tags the buffer with its
  `{transfer, primaries}` and zssim does the correct linearization and perceptual
  encoding from built-in constant tables. No ICC blobs ever reach zssim.

This keeps zssim pure Zig with zero external dependencies, and avoids duplicating
ImProc's already-vendored color stack.

---

## Usage (planned API)

```zig
const zssim = @import("zssim");

var ctx = zssim.Comparator.init(allocator, .{}); // default config
defer ctx.deinit();

// 8-bit sRGB RGBA:
const a = try ctx.prepareRgba8(pixels_a, w, h, .srgb);
defer a.deinit();

// ImProc's native 16-bit RGBA (top-down, straight alpha), tagged with its space:
const b = try ctx.prepareRgba16(pixels_b, w, h, .{ .transfer = .srgb, .primaries = .srgb });
defer b.deinit();

const result = try ctx.compare(a, b);
std.debug.print("distance={d:.6} ssim={d:.6}\n", .{ result.distance, result.ssim });
```

Supported inputs (planned): `rgba8`, `rgb8`, `gray8`, `rgba16`, `rgb16`, `gray16`,
and `linear_rgb_f32`. The 16-bit RGBA path matches ImProc's internal format
directly.

---

## Build & test

Requires Zig **0.16.0**.

```sh
zig build test     # run the unit test suite
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
- 🚧 **Color module** — sRGB/linear/gamma/PQ/HLG transfers; sRGB/P3/Rec.2020/
  Adobe-RGB primaries; linear→XYZ→CIELAB; ICtCp (HDR); `[0,1]` normalization
- ⬜ **Separable Gaussian blur** — single-temp-buffer two-pass, plus a fused
  `blur(a·b)` that never materializes the product
- ⬜ **Pyramid** — linear-light 2×2 downsample with a min-size floor
- ⬜ **Image preprocessing** — input adapters (8/16-bit RGBA/RGB/Gray + linear
  float), planar channel layout, per-channel `μ` and `blur(I²)` precompute
- ⬜ **SSIM core** — per-channel SSIM map, pooling (mean / MAD), channel weights,
  multi-scale weighted combine, `compare()` returning SSIM + distance
- ⬜ **Grayscale fast path** — single-channel pipeline
- ⬜ **Tests** — identity = 0, symmetry, monotonicity vs. added noise/blur,
  16-bit↔8-bit equivalence, edge sizes (1×1, tiny, non-square), HDR sanity
- ⬜ **SIMD optimization** — `@Vector` blur + SSIM combine; LUT linearization
- ⬜ **Optional threading** — per-scale / per-channel parallelism via a thread pool
- ⬜ **C ABI** — `zssim.h` + exported functions for non-Zig callers
- ⬜ **ImProc integration** — path dependency, `ColorProfileKind` → zssim preset
  adapter, a `compare` subcommand / GUI hook
- ⬜ **Difference visualization** — export per-scale SSIM heatmap (like `distance -o`)
- ⬜ **Validation** — correlation check against a public IQA dataset (e.g. TID2013)

### Possible improvements over the reference (to explore)

- ICtCp HDR path (the reference is sRGB-only).
- Exact-Gaussian SSIM window with configurable σ (vs. a fixed small kernel).
- Edge handling by weight renormalization (unbiased borders).
- Higher-accuracy / SIMD CIELAB and a wide-gamut-correct matrix per call.
- Fused multiply-blur to cut memory traffic.

---

## Progress log

- **2026-06-23** — Project created. Studied the public SSIM/MS-SSIM/CIELAB/ICtCp
  literature and the reference architectures (distance-core is pure-sRGB; ImProc
  owns ICC via lcms2). Decided the clean-room design, color division of labor,
  and the parameterized transfer/primaries approach for HDR/WCG. Scaffolded the
  repo (builds + trivial test passing).
