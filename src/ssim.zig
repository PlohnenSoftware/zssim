//! The SSIM kernel: turn precomputed local moments into a similarity score.
//!
//! For each pixel and channel, with μ = blur(I), and the blurred second moments
//! `sq = blur(I²)`, `cross = blur(I₁·I₂)`:
//!
//!   σ₁²  = sq₁  − μ₁²
//!   σ₂²  = sq₂  − μ₂²
//!   σ₁₂  = cross − μ₁·μ₂
//!   SSIM = (2μ₁μ₂ + C₁)(2σ₁₂ + C₂) / ((μ₁²+μ₂²+C₁)(σ₁²+σ₂²+C₂))
//!
//! with C₁=(K₁·L)², C₂=(K₂·L)², K₁=0.01, K₂=0.03 and data range L=1 (channels
//! are normalized to ~[0,1]). This is the standard SSIM of Wang et al. (2004).

const std = @import("std");

pub const k1: f32 = 0.01;
pub const k2: f32 = 0.03;
pub const c1: f32 = k1 * k1;
pub const c2: f32 = k2 * k2;

/// Per-pixel SSIM for one channel from precomputed moments. All slices are
/// `n` long. Algebraically, identical inputs yield exactly 1.0 per pixel.
pub fn ssimAt(mu1v: f32, mu2v: f32, sq1v: f32, sq2v: f32, crossv: f32) f32 {
    const mu1mu1 = mu1v * mu1v;
    const mu2mu2 = mu2v * mu2v;
    const mu1mu2 = mu1v * mu2v;
    const sigma1_sq = sq1v - mu1mu1;
    const sigma2_sq = sq2v - mu2mu2;
    const sigma12 = crossv - mu1mu2;
    const num = (2.0 * mu1mu2 + c1) * (2.0 * sigma12 + c2);
    const den = (mu1mu1 + mu2mu2 + c1) * (sigma1_sq + sigma2_sq + c2);
    return num / den;
}

/// Mean SSIM over a channel. Accumulates in `f64` for numerical stability over
/// large images.
pub fn meanSsimChannel(mu1: []const f32, mu2: []const f32, sq1: []const f32, sq2: []const f32, cross: []const f32) f64 {
    const n = mu1.len;
    std.debug.assert(mu2.len == n and sq1.len == n and sq2.len == n and cross.len == n);
    var sum: f64 = 0;
    for (0..n) |i| {
        sum += ssimAt(mu1[i], mu2[i], sq1[i], sq2[i], cross[i]);
    }
    return sum / @as(f64, @floatFromInt(n));
}

/// Write the per-pixel SSIM map for a channel (for difference visualization),
/// returning its mean.
pub fn ssimMapChannel(mu1: []const f32, mu2: []const f32, sq1: []const f32, sq2: []const f32, cross: []const f32, out: []f32) f64 {
    const n = mu1.len;
    std.debug.assert(out.len >= n);
    var sum: f64 = 0;
    for (0..n) |i| {
        const s = ssimAt(mu1[i], mu2[i], sq1[i], sq2[i], cross[i]);
        out[i] = s;
        sum += s;
    }
    return sum / @as(f64, @floatFromInt(n));
}

/// Convert an SSIM score in (0,1] to a perceptual distance `1/SSIM − 1`
/// (0 = identical, larger = more different; unbounded).
pub fn toDistance(ssim: f64) f64 {
    return 1.0 / @max(ssim, std.math.floatEps(f64)) - 1.0;
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

const testing = std.testing;

test "identical moments give SSIM exactly 1" {
    // For I1==I2: mu1==mu2, sq1==sq2, and cross==sq (blur of I*I). Then every
    // term cancels and SSIM == 1 per pixel.
    const mu = [_]f32{ 0.2, 0.5, 0.8, 0.3 };
    const sq = [_]f32{ 0.05, 0.26, 0.66, 0.1 };
    const mean = meanSsimChannel(&mu, &mu, &sq, &sq, &sq);
    try testing.expectApproxEqAbs(@as(f64, 1.0), mean, 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 0.0), toDistance(mean), 1e-12);
}

test "different signals give SSIM below 1" {
    const mu1 = [_]f32{ 0.2, 0.5, 0.8, 0.3 };
    const mu2 = [_]f32{ 0.6, 0.1, 0.4, 0.9 };
    const sq1 = [_]f32{ 0.06, 0.30, 0.70, 0.12 };
    const sq2 = [_]f32{ 0.40, 0.05, 0.20, 0.85 };
    const cross = [_]f32{ 0.12, 0.05, 0.32, 0.27 };
    const mean = meanSsimChannel(&mu1, &mu2, &sq1, &sq2, &cross);
    try testing.expect(mean < 1.0);
    try testing.expect(toDistance(mean) > 0.0);
}

test "toDistance is monotonic decreasing in ssim" {
    try testing.expect(toDistance(0.9) < toDistance(0.5));
    try testing.expect(toDistance(0.5) < toDistance(0.1));
}
