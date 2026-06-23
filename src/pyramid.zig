//! Image-pyramid downsampling for multi-scale SSIM.
//!
//! Each step halves a plane by averaging 2×2 blocks. Downsampling is done in
//! **linear light** (this module operates on the linear-RGB planes, before the
//! L*a*b*/ICtCp encode) so that blending colors models physical viewing-distance
//! blur correctly — averaging gamma-encoded or Lab values would be wrong.
//!
//! Odd trailing rows/columns are dropped (floor division), matching the common
//! multi-scale-SSIM convention.

const std = @import("std");

/// Downsampled dimension: floor(n / 2).
pub inline fn halfDim(n: usize) usize {
    return n / 2;
}

/// Average 2×2 blocks of `src` (`sw×sh`) into `dst` (`halfDim(sw)×halfDim(sh)`).
/// `dst` must be at least `halfDim(sw)*halfDim(sh)` long.
pub fn downsamplePlane(src: []const f32, sw: usize, sh: usize, dst: []f32) void {
    const dw = halfDim(sw);
    const dh = halfDim(sh);
    std.debug.assert(dst.len >= dw * dh);
    for (0..dh) |dy| {
        const top = (2 * dy) * sw;
        const bot = (2 * dy + 1) * sw;
        const drow = dy * dw;
        for (0..dw) |dx| {
            const sx = 2 * dx;
            const sum = src[top + sx] + src[top + sx + 1] + src[bot + sx] + src[bot + sx + 1];
            dst[drow + dx] = sum * 0.25;
        }
    }
}

/// How many scales are usable for a `w×h` image given a desired count and a
/// minimum side length: stop before any side would drop below `min_side`.
pub fn usableScales(w: usize, h: usize, desired: usize, min_side: usize) usize {
    var count: usize = 1; // scale 0 = full resolution
    var cw = w;
    var ch = h;
    while (count < desired) {
        const nw = halfDim(cw);
        const nh = halfDim(ch);
        if (nw < min_side or nh < min_side) break;
        cw = nw;
        ch = nh;
        count += 1;
    }
    return count;
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

const testing = std.testing;

test "downsample a flat plane preserves the value" {
    const src = [_]f32{0.3} ** 16; // 4x4
    var dst: [4]f32 = undefined; // 2x2
    downsamplePlane(&src, 4, 4, &dst);
    for (dst) |v| try testing.expectApproxEqAbs(@as(f32, 0.3), v, 1e-7);
}

test "downsample averages 2x2 blocks" {
    // 2x2 image -> single pixel = mean.
    const src = [_]f32{ 0.0, 0.4, 0.8, 1.0 };
    var dst: [1]f32 = undefined;
    downsamplePlane(&src, 2, 2, &dst);
    try testing.expectApproxEqAbs(@as(f32, 0.55), dst[0], 1e-7);
}

test "downsample drops odd trailing row/col" {
    // 3x3 -> 1x1, using only the top-left 2x2 block.
    const src = [_]f32{
        1.0, 1.0, 9.0,
        1.0, 1.0, 9.0,
        9.0, 9.0, 9.0,
    };
    var dst: [1]f32 = undefined;
    downsamplePlane(&src, 3, 3, &dst);
    try testing.expectApproxEqAbs(@as(f32, 1.0), dst[0], 1e-7);
}

test "usableScales respects min side" {
    // 64x64, want 5 scales, min side 8: 64,32,16,8 -> 4 scales (next is 4 < 8).
    try testing.expectEqual(@as(usize, 4), usableScales(64, 64, 5, 8));
    // Plenty of room.
    try testing.expectEqual(@as(usize, 5), usableScales(1024, 1024, 5, 8));
    // Tiny image: only full res.
    try testing.expectEqual(@as(usize, 1), usableScales(10, 10, 5, 8));
}
