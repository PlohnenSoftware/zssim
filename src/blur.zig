//! Separable Gaussian blur over tightly-packed planar `f32` images.
//!
//! SSIM needs three blurred quantities per channel: `blur(I)` (the local mean
//! μ), `blur(I²)`, and the cross term `blur(I₁·I₂)`. A 2-D Gaussian is
//! separable, so each is two 1-D passes (horizontal then vertical) sharing one
//! scratch buffer. `blurMul` fuses the element-wise product into the horizontal
//! pass, so the product is never materialized as its own buffer.
//!
//! Edges use clamp-to-edge (the image is treated as extended by its border
//! pixel). The kernel stays normalized, so a flat region blurs to itself.

const std = @import("std");

/// A normalized, symmetric 1-D Gaussian kernel of length `2*radius + 1`.
pub const Kernel = struct {
    taps: []const f32,
    radius: usize,

    /// Build a Gaussian kernel for the given standard deviation. The radius is
    /// `ceil(3σ)` (covers ~99.7% of the mass), minimum 1.
    pub fn gaussian(allocator: std.mem.Allocator, sigma: f32) !Kernel {
        std.debug.assert(sigma > 0);
        const radius: usize = @max(1, @as(usize, @intFromFloat(@ceil(3.0 * sigma))));
        const len = 2 * radius + 1;
        const taps = try allocator.alloc(f32, len);
        errdefer allocator.free(taps);

        const inv2s2: f32 = 1.0 / (2.0 * sigma * sigma);
        var sum: f32 = 0;
        for (0..len) |i| {
            const x: f32 = @floatFromInt(@as(isize, @intCast(i)) - @as(isize, @intCast(radius)));
            const w = @exp(-x * x * inv2s2);
            taps[i] = w;
            sum += w;
        }
        const inv_sum = 1.0 / sum;
        for (taps) |*t| t.* *= inv_sum;
        return .{ .taps = taps, .radius = radius };
    }

    pub fn deinit(self: Kernel, allocator: std.mem.Allocator) void {
        allocator.free(self.taps);
    }
};

inline fn clampIndex(i: isize, hi: usize) usize {
    if (i < 0) return 0;
    const u: usize = @intCast(i);
    return @min(u, hi - 1);
}

/// Horizontal pass over `src` → `dst` (both `w*h`, tightly packed).
fn horizontal(src: []const f32, dst: []f32, w: usize, h: usize, k: Kernel) void {
    const r: isize = @intCast(k.radius);
    for (0..h) |y| {
        const row = y * w;
        for (0..w) |x| {
            var acc: f32 = 0;
            const xi: isize = @intCast(x);
            for (k.taps, 0..) |tap, j| {
                const sx = clampIndex(xi + @as(isize, @intCast(j)) - r, w);
                acc += tap * src[row + sx];
            }
            dst[row + x] = acc;
        }
    }
}

/// Horizontal pass over the element-wise product `a·b` → `dst`.
fn horizontalMul(a: []const f32, b: []const f32, dst: []f32, w: usize, h: usize, k: Kernel) void {
    const r: isize = @intCast(k.radius);
    for (0..h) |y| {
        const row = y * w;
        for (0..w) |x| {
            var acc: f32 = 0;
            const xi: isize = @intCast(x);
            for (k.taps, 0..) |tap, j| {
                const sx = clampIndex(xi + @as(isize, @intCast(j)) - r, w);
                acc += tap * (a[row + sx] * b[row + sx]);
            }
            dst[row + x] = acc;
        }
    }
}

/// Vertical pass over `src` → `dst`.
fn vertical(src: []const f32, dst: []f32, w: usize, h: usize, k: Kernel) void {
    const r: isize = @intCast(k.radius);
    for (0..h) |y| {
        const yi: isize = @intCast(y);
        for (0..w) |x| {
            var acc: f32 = 0;
            for (k.taps, 0..) |tap, j| {
                const sy = clampIndex(yi + @as(isize, @intCast(j)) - r, h);
                acc += tap * src[sy * w + x];
            }
            dst[y * w + x] = acc;
        }
    }
}

/// `dst = blur(src)`. `tmp` and `dst` must each be at least `w*h` long.
pub fn blur(src: []const f32, tmp: []f32, dst: []f32, w: usize, h: usize, k: Kernel) void {
    std.debug.assert(src.len >= w * h and tmp.len >= w * h and dst.len >= w * h);
    horizontal(src, tmp, w, h, k);
    vertical(tmp, dst, w, h, k);
}

/// `dst = blur(a · b)`, fusing the product into the horizontal pass.
pub fn blurMul(a: []const f32, b: []const f32, tmp: []f32, dst: []f32, w: usize, h: usize, k: Kernel) void {
    std.debug.assert(a.len >= w * h and b.len >= w * h and tmp.len >= w * h and dst.len >= w * h);
    horizontalMul(a, b, tmp, w, h, k);
    vertical(tmp, dst, w, h, k);
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

const testing = std.testing;

test "kernel is normalized and symmetric" {
    const k = try Kernel.gaussian(testing.allocator, 1.5);
    defer k.deinit(testing.allocator);
    var sum: f32 = 0;
    for (k.taps) |t| sum += t;
    try testing.expectApproxEqAbs(@as(f32, 1.0), sum, 1e-6);
    for (0..k.radius) |d| {
        try testing.expectApproxEqAbs(k.taps[k.radius - 1 - d], k.taps[k.radius + 1 + d], 1e-7);
    }
}

test "blurring a flat image is a no-op" {
    const w = 9;
    const h = 7;
    const k = try Kernel.gaussian(testing.allocator, 1.5);
    defer k.deinit(testing.allocator);

    var src: [w * h]f32 = undefined;
    @memset(&src, 0.42);
    var tmp: [w * h]f32 = undefined;
    var dst: [w * h]f32 = undefined;
    blur(&src, &tmp, &dst, w, h, k);
    for (dst) |v| try testing.expectApproxEqAbs(@as(f32, 0.42), v, 1e-5);
}

test "blur conserves total mass on an interior impulse" {
    const w = 11;
    const h = 11;
    const k = try Kernel.gaussian(testing.allocator, 1.5);
    defer k.deinit(testing.allocator);

    var src: [w * h]f32 = undefined;
    @memset(&src, 0);
    src[5 * w + 5] = 1.0; // centered impulse, far from edges
    var tmp: [w * h]f32 = undefined;
    var dst: [w * h]f32 = undefined;
    blur(&src, &tmp, &dst, w, h, k);

    var total: f32 = 0;
    for (dst) |v| total += v;
    try testing.expectApproxEqAbs(@as(f32, 1.0), total, 1e-4);
    // Peak stays at the center and is the largest value.
    for (dst) |v| try testing.expect(v <= dst[5 * w + 5] + 1e-6);
}

test "blurMul equals blur of the materialized product" {
    const w = 8;
    const h = 6;
    const k = try Kernel.gaussian(testing.allocator, 1.2);
    defer k.deinit(testing.allocator);

    var a: [w * h]f32 = undefined;
    var b: [w * h]f32 = undefined;
    for (0..w * h) |i| {
        a[i] = @as(f32, @floatFromInt((i * 7) % 13)) / 13.0;
        b[i] = @as(f32, @floatFromInt((i * 5) % 11)) / 11.0;
    }
    var prod: [w * h]f32 = undefined;
    for (0..w * h) |i| prod[i] = a[i] * b[i];

    var tmp: [w * h]f32 = undefined;
    var dst_fused: [w * h]f32 = undefined;
    var dst_ref: [w * h]f32 = undefined;
    blurMul(&a, &b, &tmp, &dst_fused, w, h, k);
    blur(&prod, &tmp, &dst_ref, w, h, k);

    for (0..w * h) |i| try testing.expectApproxEqAbs(dst_ref[i], dst_fused[i], 1e-6);
}
