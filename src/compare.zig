//! Public API: `Comparator`, image preparation, and `compare`.
//!
//! Pipeline per image (`prepare*`):
//!   raw pixels → linear-light RGB planes (premultiplied alpha)
//!             → multi-scale pyramid (2×2 box average, linear light)
//!             → per-scale perceptual encode (gray L* / L*a*b* / ICtCp)
//!             → per-channel precompute of μ=blur(I) and sq=blur(I²)
//!
//! Then `compare(a, b)` computes, per scale and channel, the cross term
//! `blur(Iₐ·I_b)`, the mean SSIM, combines channels (luma-weighted) and scales
//! (weighted), and returns the SSIM score plus the derived distance.

const std = @import("std");
const color = @import("color.zig");
const blur = @import("blur.zig");
const pyramid = @import("pyramid.zig");
const ssim = @import("ssim.zig");

pub const Transfer = color.Transfer;
pub const Primaries = color.Primaries;

/// The working color space a prepared buffer is in. The host (e.g. ImProc) is
/// responsible for flattening ICC profiles to one of these before calling.
pub const Space = struct {
    transfer: Transfer = .srgb,
    primaries: Primaries = .srgb,
    /// Unbounded scene-linear input (e.g. ACES2065-1). Forces the HDR-capable
    /// ICtCp encoder instead of SDR CIELAB.
    scene_referred: bool = false,
};

pub const Encoding = enum { gray, lab, ictcp };

pub const Pooling = enum {
    /// Standard mean of the SSIM map.
    mean,
};

/// Default per-scale weights: the published MS-SSIM weights (Wang 2003),
/// used here as relative weights for a weighted arithmetic mean (renormalized).
pub const default_scale_weights = [_]f32{ 0.0448, 0.2856, 0.3001, 0.2363, 0.1333 };

pub const Config = struct {
    /// Gaussian window standard deviation (in pixels) for the SSIM statistics.
    sigma: f32 = 1.5,
    /// Stop building scales before any side drops below this.
    min_side: usize = 8,
    /// Pre-blur chroma channels once (eye has lower spatial acuity for color).
    chroma_blur: bool = true,
    pooling: Pooling = .mean,
    /// Relative weight of each scale (renormalized). Length sets the max scales.
    scale_weights: []const f32 = &default_scale_weights,
    /// Per-channel weights for color: [L|I, a|Ct, b|Cp]. Luma-weighted default.
    channel_weights: [3]f32 = .{ 0.8, 0.1, 0.1 },
};

pub const Result = struct {
    /// Overall SSIM in (0,1]; 1.0 == identical.
    ssim: f64,
    /// Perceptual distance = 1/SSIM − 1 in [0,∞); 0.0 == identical, unbounded.
    distance: f64,
    /// Number of scales actually used (capped by image size).
    scales: usize,
};

pub const Error = error{
    SizeMismatch,
    EncodingMismatch,
    EmptyImage,
    BadPixelBuffer,
} || std.mem.Allocator.Error;

const ChannelPlanes = struct {
    img: []f32, // perceptual channel value I
    mu: []f32, // blur(I)
    sq: []f32, // blur(I²)
};

const Scale = struct {
    w: usize,
    h: usize,
    channels: []ChannelPlanes,
};

/// A preprocessed image ready for `compare`. Owns all its plane data via an
/// arena; free with `deinit`.
pub const PreparedImage = struct {
    arena: *std.heap.ArenaAllocator,
    parent: std.mem.Allocator,
    width: usize,
    height: usize,
    encoding: Encoding,
    channel_count: usize,
    scales: []Scale,

    pub fn deinit(self: PreparedImage) void {
        self.arena.deinit();
        self.parent.destroy(self.arena);
    }
};

pub const Comparator = struct {
    allocator: std.mem.Allocator,
    cfg: Config,
    kernel: blur.Kernel,

    pub fn init(allocator: std.mem.Allocator, cfg: Config) !Comparator {
        return .{
            .allocator = allocator,
            .cfg = cfg,
            .kernel = try blur.Kernel.gaussian(allocator, cfg.sigma),
        };
    }

    pub fn deinit(self: *Comparator) void {
        self.kernel.deinit(self.allocator);
    }

    // --- Input entry points -------------------------------------------------

    /// 8-bit sRGB-style RGBA (R,G,B,A order, straight alpha), `w*h*4` bytes.
    pub fn prepareRgba8(self: *Comparator, pixels: []const u8, w: usize, h: usize, space: Space) Error!PreparedImage {
        if (pixels.len < w * h * 4) return Error.BadPixelBuffer;
        return self.prepareColorInt(u8, 4, true, pixels, w, h, space);
    }

    /// 8-bit RGB (no alpha), `w*h*3` bytes.
    pub fn prepareRgb8(self: *Comparator, pixels: []const u8, w: usize, h: usize, space: Space) Error!PreparedImage {
        if (pixels.len < w * h * 3) return Error.BadPixelBuffer;
        return self.prepareColorInt(u8, 3, false, pixels, w, h, space);
    }

    /// 8-bit grayscale, `w*h` bytes.
    pub fn prepareGray8(self: *Comparator, pixels: []const u8, w: usize, h: usize, space: Space) Error!PreparedImage {
        if (pixels.len < w * h) return Error.BadPixelBuffer;
        return self.prepareGrayInt(u8, pixels, w, h, space);
    }

    /// 16-bit RGBA — **ImProc's native internal format** (R,G,B,A, straight
    /// alpha, top-down, tightly packed), `w*h*4` samples.
    pub fn prepareRgba16(self: *Comparator, pixels: []const u16, w: usize, h: usize, space: Space) Error!PreparedImage {
        if (pixels.len < w * h * 4) return Error.BadPixelBuffer;
        return self.prepareColorInt(u16, 4, true, pixels, w, h, space);
    }

    /// 16-bit RGB (no alpha), `w*h*3` samples.
    pub fn prepareRgb16(self: *Comparator, pixels: []const u16, w: usize, h: usize, space: Space) Error!PreparedImage {
        if (pixels.len < w * h * 3) return Error.BadPixelBuffer;
        return self.prepareColorInt(u16, 3, false, pixels, w, h, space);
    }

    /// 16-bit grayscale, `w*h` samples.
    pub fn prepareGray16(self: *Comparator, pixels: []const u16, w: usize, h: usize, space: Space) Error!PreparedImage {
        if (pixels.len < w * h) return Error.BadPixelBuffer;
        return self.prepareGrayInt(u16, pixels, w, h, space);
    }

    /// Already-linear interleaved RGB float, `w*h*3` samples. The `space`'s
    /// transfer is ignored (input is linear); its primaries/scene_referred flag
    /// still select the encode matrix and SDR-vs-HDR encoder.
    pub fn prepareLinearRgbF32(self: *Comparator, pixels: []const f32, w: usize, h: usize, space: Space) Error!PreparedImage {
        if (pixels.len < w * h * 3) return Error.BadPixelBuffer;
        if (w == 0 or h == 0) return Error.EmptyImage;
        var scratch = std.heap.ArenaAllocator.init(self.allocator);
        defer scratch.deinit();
        const sa = scratch.allocator();
        const n = w * h;
        var lin = try sa.alloc([]f32, 3);
        for (0..3) |c| lin[c] = try sa.alloc(f32, n);
        for (0..n) |i| {
            lin[0][i] = pixels[i * 3 + 0];
            lin[1][i] = pixels[i * 3 + 1];
            lin[2][i] = pixels[i * 3 + 2];
        }
        const enc: Encoding = if (space.scene_referred or space.transfer == .pq or space.transfer == .hlg) .ictcp else .lab;
        return self.prepareCommon(&scratch, lin, w, h, space.primaries, enc);
    }

    // --- Internal preparation ----------------------------------------------

    fn prepareColorInt(self: *Comparator, comptime T: type, comptime stride: usize, comptime has_alpha: bool, pixels: []const T, w: usize, h: usize, space: Space) Error!PreparedImage {
        if (w == 0 or h == 0) return Error.EmptyImage;
        var scratch = std.heap.ArenaAllocator.init(self.allocator);
        defer scratch.deinit();
        const sa = scratch.allocator();
        const n = w * h;
        const maxv: f32 = @floatFromInt(std.math.maxInt(T));

        var lin = try sa.alloc([]f32, 3);
        for (0..3) |c| lin[c] = try sa.alloc(f32, n);

        for (0..n) |i| {
            const b = i * stride;
            const a: f32 = if (has_alpha) @as(f32, @floatFromInt(pixels[b + 3])) / maxv else 1.0;
            lin[0][i] = space.transfer.decode(@as(f32, @floatFromInt(pixels[b + 0])) / maxv) * a;
            lin[1][i] = space.transfer.decode(@as(f32, @floatFromInt(pixels[b + 1])) / maxv) * a;
            lin[2][i] = space.transfer.decode(@as(f32, @floatFromInt(pixels[b + 2])) / maxv) * a;
        }

        const enc: Encoding = if (space.scene_referred or space.transfer == .pq or space.transfer == .hlg) .ictcp else .lab;
        return self.prepareCommon(&scratch, lin, w, h, space.primaries, enc);
    }

    fn prepareGrayInt(self: *Comparator, comptime T: type, pixels: []const T, w: usize, h: usize, space: Space) Error!PreparedImage {
        if (w == 0 or h == 0) return Error.EmptyImage;
        var scratch = std.heap.ArenaAllocator.init(self.allocator);
        defer scratch.deinit();
        const sa = scratch.allocator();
        const n = w * h;
        const maxv: f32 = @floatFromInt(std.math.maxInt(T));

        var lin = try sa.alloc([]f32, 1);
        lin[0] = try sa.alloc(f32, n);
        for (0..n) |i| {
            lin[0][i] = space.transfer.decode(@as(f32, @floatFromInt(pixels[i])) / maxv);
        }
        return self.prepareCommon(&scratch, lin, w, h, space.primaries, .gray);
    }

    /// Common pipeline once linear planes exist. `lin` lives in `scratch` (freed
    /// by the caller's `defer`); the returned image owns its own arena.
    fn prepareCommon(self: *Comparator, scratch: *std.heap.ArenaAllocator, lin: [][]f32, w: usize, h: usize, primaries: Primaries, encoding: Encoding) Error!PreparedImage {
        const lin_count = lin.len;
        const channel_count: usize = if (encoding == .gray) 1 else 3;
        const num_scales = pyramid.usableScales(w, h, self.cfg.scale_weights.len, self.cfg.min_side);

        const arena = try self.allocator.create(std.heap.ArenaAllocator);
        arena.* = std.heap.ArenaAllocator.init(self.allocator);
        errdefer {
            arena.deinit();
            self.allocator.destroy(arena);
        }
        const a = arena.allocator();
        const scales = try a.alloc(Scale, num_scales);

        const sa = scratch.allocator();
        // `cur` holds the current scale's linear planes; scale 0 aliases `lin`.
        var cur = lin;
        var cw = w;
        var ch = h;
        // Scratch reused across scales (largest at scale 0).
        const tmp = try sa.alloc(f32, w * h);
        const tmp2 = try sa.alloc(f32, w * h);

        for (0..num_scales) |s| {
            if (s > 0) {
                const nw = pyramid.halfDim(cw);
                const nh = pyramid.halfDim(ch);
                var next = try sa.alloc([]f32, lin_count);
                for (0..lin_count) |c| {
                    next[c] = try sa.alloc(f32, nw * nh);
                    pyramid.downsamplePlane(cur[c], cw, ch, next[c]);
                }
                cur = next;
                cw = nw;
                ch = nh;
            }

            const n = cw * ch;
            const chans = try a.alloc(ChannelPlanes, channel_count);
            for (0..channel_count) |c| {
                chans[c] = .{
                    .img = try a.alloc(f32, n),
                    .mu = try a.alloc(f32, n),
                    .sq = try a.alloc(f32, n),
                };
            }

            encodeScale(encoding, primaries, cur, n, chans);

            for (0..channel_count) |c| {
                // Chroma channels get an extra blur (lower spatial acuity).
                if (self.cfg.chroma_blur and encoding != .gray and c >= 1) {
                    blur.blur(chans[c].img, tmp, chans[c].img, cw, ch, self.kernel);
                }
                blur.blur(chans[c].img, tmp[0 .. cw * ch], chans[c].mu, cw, ch, self.kernel);
                blur.blurMul(chans[c].img, chans[c].img, tmp2[0 .. cw * ch], chans[c].sq, cw, ch, self.kernel);
            }

            scales[s] = .{ .w = cw, .h = ch, .channels = chans };
        }

        return .{
            .arena = arena,
            .parent = self.allocator,
            .width = w,
            .height = h,
            .encoding = encoding,
            .channel_count = channel_count,
            .scales = scales,
        };
    }

    // --- Comparison ---------------------------------------------------------

    pub fn compare(self: *Comparator, a: PreparedImage, b: PreparedImage) Error!Result {
        if (a.width != b.width or a.height != b.height) return Error.SizeMismatch;
        if (a.encoding != b.encoding) return Error.EncodingMismatch;

        const num = @min(a.scales.len, b.scales.len);
        const max_n = a.scales[0].w * a.scales[0].h;
        const tmp = try self.allocator.alloc(f32, max_n);
        defer self.allocator.free(tmp);
        const cross = try self.allocator.alloc(f32, max_n);
        defer self.allocator.free(cross);

        var ssim_acc: f64 = 0;
        var weight_sum: f64 = 0;

        for (0..num) |s| {
            const sa = a.scales[s];
            const sb = b.scales[s];
            const n = sa.w * sa.h;

            var scale_ssim: f64 = 0;
            var chan_w_sum: f64 = 0;
            for (0..a.channel_count) |c| {
                blur.blurMul(sa.channels[c].img, sb.channels[c].img, tmp[0..n], cross[0..n], sa.w, sa.h, self.kernel);
                const m = ssim.meanSsimChannel(sa.channels[c].mu, sb.channels[c].mu, sa.channels[c].sq, sb.channels[c].sq, cross[0..n]);
                const w: f64 = if (a.channel_count == 1) 1.0 else self.cfg.channel_weights[c];
                scale_ssim += w * m;
                chan_w_sum += w;
            }
            scale_ssim /= chan_w_sum;

            const sw: f64 = self.cfg.scale_weights[s];
            ssim_acc += sw * scale_ssim;
            weight_sum += sw;
        }

        const overall = ssim_acc / weight_sum;
        return .{ .ssim = overall, .distance = ssim.toDistance(overall), .scales = num };
    }
};

/// Fill the perceptual channel `img` planes from linear planes `cur`.
fn encodeScale(encoding: Encoding, primaries: Primaries, cur: [][]f32, n: usize, chans: []ChannelPlanes) void {
    switch (encoding) {
        .gray => {
            for (0..n) |i| chans[0].img[i] = color.grayToLNorm(cur[0][i]);
        },
        .lab => switch (primaries) {
            inline else => |p| {
                for (0..n) |i| {
                    const v = color.labNormalized(p, .{ cur[0][i], cur[1][i], cur[2][i] });
                    chans[0].img[i] = v[0];
                    chans[1].img[i] = v[1];
                    chans[2].img[i] = v[2];
                }
            },
        },
        .ictcp => switch (primaries) {
            inline else => |p| {
                for (0..n) |i| {
                    const v = color.ictcpNormalized(p, .{ cur[0][i], cur[1][i], cur[2][i] });
                    chans[0].img[i] = v[0];
                    chans[1].img[i] = v[1];
                    chans[2].img[i] = v[2];
                }
            },
        },
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

const testing = std.testing;

fn makePattern(allocator: std.mem.Allocator, w: usize, h: usize, shift: u8) ![]u8 {
    const buf = try allocator.alloc(u8, w * h * 4);
    for (0..h) |y| {
        for (0..w) |x| {
            const i = (y * w + x) * 4;
            buf[i + 0] = @intCast((x * 7 + y * 3 + shift) % 256);
            buf[i + 1] = @intCast((x * 3 + y * 11 + shift) % 256);
            buf[i + 2] = @intCast((x * 5 + y * 5 + shift) % 256);
            buf[i + 3] = 255;
        }
    }
    return buf;
}

test "identical images give distance ~ 0" {
    var ctx = try Comparator.init(testing.allocator, .{});
    defer ctx.deinit();
    const w = 48;
    const h = 40;
    const px = try makePattern(testing.allocator, w, h, 0);
    defer testing.allocator.free(px);

    const a = try ctx.prepareRgba8(px, w, h, .{});
    defer a.deinit();
    const b = try ctx.prepareRgba8(px, w, h, .{});
    defer b.deinit();

    const r = try ctx.compare(a, b);
    try testing.expectApproxEqAbs(@as(f64, 1.0), r.ssim, 1e-9);
    try testing.expectApproxEqAbs(@as(f64, 0.0), r.distance, 1e-9);
    try testing.expect(r.scales >= 1);
}

test "different images give distance > 0" {
    var ctx = try Comparator.init(testing.allocator, .{});
    defer ctx.deinit();
    const w = 48;
    const h = 40;
    const pa = try makePattern(testing.allocator, w, h, 0);
    defer testing.allocator.free(pa);
    const pb = try makePattern(testing.allocator, w, h, 60);
    defer testing.allocator.free(pb);

    const a = try ctx.prepareRgba8(pa, w, h, .{});
    defer a.deinit();
    const b = try ctx.prepareRgba8(pb, w, h, .{});
    defer b.deinit();

    const r = try ctx.compare(a, b);
    try testing.expect(r.distance > 0.0);
    try testing.expect(r.ssim < 1.0);
}

test "size mismatch is an error" {
    var ctx = try Comparator.init(testing.allocator, .{});
    defer ctx.deinit();
    const pa = try makePattern(testing.allocator, 32, 32, 0);
    defer testing.allocator.free(pa);
    const pb = try makePattern(testing.allocator, 16, 16, 0);
    defer testing.allocator.free(pb);
    const a = try ctx.prepareRgba8(pa, 32, 32, .{});
    defer a.deinit();
    const b = try ctx.prepareRgba8(pb, 16, 16, .{});
    defer b.deinit();
    try testing.expectError(Error.SizeMismatch, ctx.compare(a, b));
}

test "tiny and 1x1 images do not crash and identity holds" {
    var ctx = try Comparator.init(testing.allocator, .{});
    defer ctx.deinit();
    inline for (.{ .{ 1, 1 }, .{ 2, 3 }, .{ 5, 1 } }) |dim| {
        const w = dim[0];
        const h = dim[1];
        const px = try makePattern(testing.allocator, w, h, 0);
        defer testing.allocator.free(px);
        const a = try ctx.prepareRgba8(px, w, h, .{});
        defer a.deinit();
        const b = try ctx.prepareRgba8(px, w, h, .{});
        defer b.deinit();
        const r = try ctx.compare(a, b);
        try testing.expectApproxEqAbs(@as(f64, 1.0), r.ssim, 1e-9);
    }
}
