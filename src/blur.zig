//! Separable Gaussian blur over tightly-packed planar `f32` images.
//!
//! SSIM needs three blurred quantities per channel: `blur(I)` (the local mean
//! μ), `blur(I²)`, and the cross term `blur(I₁·I₂)`. A 2-D Gaussian is
//! separable, so each is two 1-D passes (horizontal then vertical) sharing one
//! scratch buffer. `blurMul` fuses the element-wise product into the horizontal
//! pass, so the product is never materialized as its own buffer.
//!
//! The interior of every pass is vectorized with `@Vector`; the radius-wide
//! borders (which need clamp-to-edge index handling) run scalar. Edges use
//! clamp-to-edge (the image is treated as extended by its border pixel), so a
//! flat region blurs to itself.

const std = @import("std");

/// SIMD width for the interior loops. 8×f32 maps to one AVX2 / two NEON regs.
const V = 8;
const Vec = @Vector(V, f32);

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

inline fn loadVec(s: []const f32, off: usize) Vec {
    return s[off..][0..V].*;
}
inline fn storeVec(d: []f32, off: usize, v: Vec) void {
    d[off..][0..V].* = v;
}

// --- Horizontal pass (over one source, or a fused product) -----------------

inline fn hPixel(row: []const f32, x: usize, w: usize, k: Kernel) f32 {
    const r: isize = @intCast(k.radius);
    const xi: isize = @intCast(x);
    var acc: f32 = 0;
    for (k.taps, 0..) |tap, j| {
        acc += tap * row[clampIndex(xi + @as(isize, @intCast(j)) - r, w)];
    }
    return acc;
}

fn horizontal(src: []const f32, dst: []f32, w: usize, h: usize, src_stride: usize, k: Kernel) void {
    const r = k.radius;
    for (0..h) |y| {
        const row = src[y * src_stride ..][0..w];
        const out = dst[y * w ..][0..w];
        // Left border + (if the row is too narrow for a full vector) everything.
        const interior_lo = r;
        const interior_hi = if (w > r) w - r else 0; // last in-bounds center + 1
        var x: usize = 0;
        while (x < @min(interior_lo, w)) : (x += 1) out[x] = hPixel(row, x, w, k);
        // Vectorized interior: all tap offsets stay in-bounds.
        if (interior_hi > interior_lo) {
            x = interior_lo;
            while (x + V <= interior_hi) : (x += V) {
                var acc: Vec = @splat(0.0);
                for (k.taps, 0..) |tap, j| {
                    acc += @as(Vec, @splat(tap)) * loadVec(row, x + j - r);
                }
                storeVec(out, x, acc);
            }
        }
        // Right border + remainder.
        const tail_start = @max(x, @min(interior_lo, w));
        x = tail_start;
        while (x < w) : (x += 1) out[x] = hPixel(row, x, w, k);
    }
}

inline fn hPixelMul(r1: []const f32, r2: []const f32, x: usize, w: usize, k: Kernel) f32 {
    const r: isize = @intCast(k.radius);
    const xi: isize = @intCast(x);
    var acc: f32 = 0;
    for (k.taps, 0..) |tap, j| {
        const sx = clampIndex(xi + @as(isize, @intCast(j)) - r, w);
        acc += tap * (r1[sx] * r2[sx]);
    }
    return acc;
}

fn horizontalMul(a: []const f32, b: []const f32, dst: []f32, w: usize, h: usize, sa: usize, sb: usize, k: Kernel) void {
    const r = k.radius;
    for (0..h) |y| {
        const r1 = a[y * sa ..][0..w];
        const r2 = b[y * sb ..][0..w];
        const out = dst[y * w ..][0..w];
        const interior_lo = r;
        const interior_hi = if (w > r) w - r else 0;
        var x: usize = 0;
        while (x < @min(interior_lo, w)) : (x += 1) out[x] = hPixelMul(r1, r2, x, w, k);
        if (interior_hi > interior_lo) {
            x = interior_lo;
            while (x + V <= interior_hi) : (x += V) {
                var acc: Vec = @splat(0.0);
                for (k.taps, 0..) |tap, j| {
                    const prod = loadVec(r1, x + j - r) * loadVec(r2, x + j - r);
                    acc += @as(Vec, @splat(tap)) * prod;
                }
                storeVec(out, x, acc);
            }
        }
        x = @max(x, @min(interior_lo, w));
        while (x < w) : (x += 1) out[x] = hPixelMul(r1, r2, x, w, k);
    }
}

// --- Vertical pass ---------------------------------------------------------

fn vertical(src: []const f32, dst: []f32, w: usize, h: usize, dst_stride: usize, k: Kernel) void {
    const r: isize = @intCast(k.radius);
    for (0..h) |y| {
        const yi: isize = @intCast(y);
        const out = dst[y * dst_stride ..][0..w];
        var x: usize = 0;
        // Vectorized across the row; the per-tap source row is contiguous.
        while (x + V <= w) : (x += V) {
            var acc: Vec = @splat(0.0);
            for (k.taps, 0..) |tap, j| {
                const sy = clampIndex(yi + @as(isize, @intCast(j)) - r, h);
                acc += @as(Vec, @splat(tap)) * loadVec(src, sy * w + x);
            }
            storeVec(out, x, acc);
        }
        // Remainder columns.
        while (x < w) : (x += 1) {
            var acc: f32 = 0;
            for (k.taps, 0..) |tap, j| {
                const sy = clampIndex(yi + @as(isize, @intCast(j)) - r, h);
                acc += tap * src[sy * w + x];
            }
            out[x] = acc;
        }
    }
}

// --- Public entry points ---------------------------------------------------

/// `dst = blur(src)`. `tmp` and `dst` must each be at least `w*h` long.
pub fn blur(src: []const f32, tmp: []f32, dst: []f32, w: usize, h: usize, k: Kernel) void {
    std.debug.assert(src.len >= w * h and tmp.len >= w * h and dst.len >= w * h);
    horizontal(src, tmp, w, h, w, k);
    vertical(tmp, dst, w, h, w, k);
}

/// `dst = blur(a · b)`, fusing the product into the horizontal pass.
pub fn blurMul(a: []const f32, b: []const f32, tmp: []f32, dst: []f32, w: usize, h: usize, k: Kernel) void {
    std.debug.assert(a.len >= w * h and b.len >= w * h and tmp.len >= w * h and dst.len >= w * h);
    horizontalMul(a, b, tmp, w, h, w, w, k);
    vertical(tmp, dst, w, h, w, k);
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

const testing = std.testing;

/// Naive O(n·k) reference used only to validate the vectorized passes.
fn blurReference(allocator: std.mem.Allocator, src: []const f32, w: usize, h: usize, k: Kernel) ![]f32 {
    const r: isize = @intCast(k.radius);
    const mid = try allocator.alloc(f32, w * h);
    defer allocator.free(mid);
    const out = try allocator.alloc(f32, w * h);
    for (0..h) |y| {
        for (0..w) |x| {
            var acc: f32 = 0;
            for (k.taps, 0..) |tap, j| {
                acc += tap * src[y * w + clampIndex(@as(isize, @intCast(x)) + @as(isize, @intCast(j)) - r, w)];
            }
            mid[y * w + x] = acc;
        }
    }
    for (0..h) |y| {
        for (0..w) |x| {
            var acc: f32 = 0;
            for (k.taps, 0..) |tap, j| {
                acc += tap * mid[clampIndex(@as(isize, @intCast(y)) + @as(isize, @intCast(j)) - r, h) * w + x];
            }
            out[y * w + x] = acc;
        }
    }
    return out;
}

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
    const w = 21;
    const h = 17;
    const k = try Kernel.gaussian(testing.allocator, 1.5);
    defer k.deinit(testing.allocator);
    var src: [w * h]f32 = undefined;
    @memset(&src, 0);
    src[8 * w + 10] = 1.0;
    var tmp: [w * h]f32 = undefined;
    var dst: [w * h]f32 = undefined;
    blur(&src, &tmp, &dst, w, h, k);
    var total: f32 = 0;
    for (dst) |v| total += v;
    try testing.expectApproxEqAbs(@as(f32, 1.0), total, 1e-4);
}

test "vectorized blur matches the scalar reference (incl. borders/remainder)" {
    // Sweep widths that exercise the vector body, remainder, and narrow rows.
    const k = try Kernel.gaussian(testing.allocator, 1.7);
    defer k.deinit(testing.allocator);
    var rng = std.Random.DefaultPrng.init(99);
    const rnd = rng.random();
    for ([_][2]usize{ .{ 1, 1 }, .{ 3, 5 }, .{ 8, 8 }, .{ 17, 13 }, .{ 32, 9 }, .{ 40, 36 } }) |dim| {
        const w = dim[0];
        const h = dim[1];
        const src = try testing.allocator.alloc(f32, w * h);
        defer testing.allocator.free(src);
        for (src) |*s| s.* = rnd.float(f32);
        const tmp = try testing.allocator.alloc(f32, w * h);
        defer testing.allocator.free(tmp);
        const got = try testing.allocator.alloc(f32, w * h);
        defer testing.allocator.free(got);
        blur(src, tmp, got, w, h, k);
        const ref = try blurReference(testing.allocator, src, w, h, k);
        defer testing.allocator.free(ref);
        for (0..w * h) |i| try testing.expectApproxEqAbs(ref[i], got[i], 1e-5);
    }
}

test "blurMul equals blur of the materialized product" {
    const w = 17;
    const h = 11;
    const k = try Kernel.gaussian(testing.allocator, 1.2);
    defer k.deinit(testing.allocator);
    const a = try testing.allocator.alloc(f32, w * h);
    defer testing.allocator.free(a);
    const b = try testing.allocator.alloc(f32, w * h);
    defer testing.allocator.free(b);
    var rng = std.Random.DefaultPrng.init(7);
    const rnd = rng.random();
    for (0..w * h) |i| {
        a[i] = rnd.float(f32);
        b[i] = rnd.float(f32);
    }
    const prod = try testing.allocator.alloc(f32, w * h);
    defer testing.allocator.free(prod);
    for (0..w * h) |i| prod[i] = a[i] * b[i];

    const tmp = try testing.allocator.alloc(f32, w * h);
    defer testing.allocator.free(tmp);
    const dst_fused = try testing.allocator.alloc(f32, w * h);
    defer testing.allocator.free(dst_fused);
    const dst_ref = try testing.allocator.alloc(f32, w * h);
    defer testing.allocator.free(dst_ref);
    blurMul(a, b, tmp, dst_fused, w, h, k);
    blur(prod, tmp, dst_ref, w, h, k);
    for (0..w * h) |i| try testing.expectApproxEqAbs(dst_ref[i], dst_fused[i], 1e-6);
}
