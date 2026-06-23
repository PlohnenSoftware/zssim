//! Property and equivalence tests for the end-to-end metric. These validate
//! that the metric behaves like a sensible perceptual distance, independent of
//! the exact algorithm internals.

const std = @import("std");
const zssim = @import("compare.zig");
const testing = std.testing;

const W = 40;
const H = 36;

fn pattern8(allocator: std.mem.Allocator, shift: u32) ![]u8 {
    const buf = try allocator.alloc(u8, W * H * 4);
    for (0..H) |y| {
        for (0..W) |x| {
            const i = (y * W + x) * 4;
            buf[i + 0] = @intCast((x * 9 + y * 4 + shift) % 256);
            buf[i + 1] = @intCast((x * 4 + y * 13 + shift) % 256);
            buf[i + 2] = @intCast((x * 6 + y * 6 + shift) % 256);
            buf[i + 3] = 255;
        }
    }
    return buf;
}

/// Same logical image as `src8`, promoted to 16-bit by *257 (exact: v/255 ==
/// v*257/65535), so the linear values are bit-identical to the 8-bit path.
fn promote16(allocator: std.mem.Allocator, src8: []const u8) ![]u16 {
    const buf = try allocator.alloc(u16, src8.len);
    for (src8, 0..) |v, i| buf[i] = @as(u16, v) * 257;
    return buf;
}

/// Reference plus additive noise of a given amplitude (deterministic).
fn noisy8(allocator: std.mem.Allocator, base: []const u8, amp: i32, seed: u64) ![]u8 {
    const buf = try allocator.alloc(u8, base.len);
    var rng = std.Random.DefaultPrng.init(seed);
    const r = rng.random();
    for (base, 0..) |v, i| {
        if (i % 4 == 3) {
            buf[i] = v; // keep alpha
            continue;
        }
        const delta = r.intRangeAtMost(i32, -amp, amp);
        buf[i] = @intCast(std.math.clamp(@as(i32, v) + delta, 0, 255));
    }
    return buf;
}

test "8-bit and 16-bit paths agree (v/255 == v*257/65535)" {
    var ctx = try zssim.Comparator.init(testing.allocator, .{});
    defer ctx.deinit();

    const p8 = try pattern8(testing.allocator, 0);
    defer testing.allocator.free(p8);
    const p16 = try promote16(testing.allocator, p8);
    defer testing.allocator.free(p16);

    const a8 = try ctx.prepareRgba8(p8, W, H, .{});
    defer a8.deinit();
    const a16 = try ctx.prepareRgba16(p16, W, H, .{});
    defer a16.deinit();

    // The 16-bit promotion is exact, so comparing the two encodings of the same
    // image must yield (near-)perfect similarity.
    const r = try ctx.compare(a8, a16);
    try testing.expectApproxEqAbs(@as(f64, 0.0), r.distance, 1e-6);
}

test "compare is symmetric" {
    var ctx = try zssim.Comparator.init(testing.allocator, .{});
    defer ctx.deinit();
    const pa = try pattern8(testing.allocator, 0);
    defer testing.allocator.free(pa);
    const pb = try pattern8(testing.allocator, 40);
    defer testing.allocator.free(pb);

    const a = try ctx.prepareRgba8(pa, W, H, .{});
    defer a.deinit();
    const b = try ctx.prepareRgba8(pb, W, H, .{});
    defer b.deinit();

    const ab = try ctx.compare(a, b);
    const ba = try ctx.compare(b, a);
    try testing.expectApproxEqAbs(ab.distance, ba.distance, 1e-9);
}

test "distance grows monotonically with noise amplitude" {
    var ctx = try zssim.Comparator.init(testing.allocator, .{});
    defer ctx.deinit();
    const base = try pattern8(testing.allocator, 0);
    defer testing.allocator.free(base);

    const ref = try ctx.prepareRgba8(base, W, H, .{});
    defer ref.deinit();

    var prev: f64 = -1.0;
    for ([_]i32{ 0, 6, 16, 40, 90 }) |amp| {
        const np = try noisy8(testing.allocator, base, amp, 12345);
        defer testing.allocator.free(np);
        const n = try ctx.prepareRgba8(np, W, H, .{});
        defer n.deinit();
        const r = try ctx.compare(ref, n);
        try testing.expect(r.distance >= prev - 1e-9); // non-decreasing
        prev = r.distance;
    }
    try testing.expect(prev > 0.0); // the largest noise is clearly different
}

test "grayscale path: identity and difference" {
    var ctx = try zssim.Comparator.init(testing.allocator, .{});
    defer ctx.deinit();
    const n = W * H;
    const g1 = try testing.allocator.alloc(u8, n);
    defer testing.allocator.free(g1);
    const g2 = try testing.allocator.alloc(u8, n);
    defer testing.allocator.free(g2);
    for (0..n) |i| {
        g1[i] = @intCast((i * 7) % 256);
        g2[i] = @intCast((i * 7 + 30) % 256);
    }

    const a = try ctx.prepareGray8(g1, W, H, .{});
    defer a.deinit();
    const a_copy = try ctx.prepareGray8(g1, W, H, .{});
    defer a_copy.deinit();
    const b = try ctx.prepareGray8(g2, W, H, .{});
    defer b.deinit();

    const same = try ctx.compare(a, a_copy);
    try testing.expectApproxEqAbs(@as(f64, 0.0), same.distance, 1e-9);
    const diff = try ctx.compare(a, b);
    try testing.expect(diff.distance > 0.0);
}

test "HDR scene-referred path (ICtCp) runs and is sane" {
    var ctx = try zssim.Comparator.init(testing.allocator, .{});
    defer ctx.deinit();
    const n = W * H;
    // Unbounded scene-linear values (some > 1.0), Rec.2020 primaries.
    const lin = try testing.allocator.alloc(f32, n * 3);
    defer testing.allocator.free(lin);
    for (0..n) |i| {
        lin[i * 3 + 0] = @as(f32, @floatFromInt(i % 17)) * 0.3;
        lin[i * 3 + 1] = @as(f32, @floatFromInt(i % 11)) * 0.5;
        lin[i * 3 + 2] = @as(f32, @floatFromInt(i % 7)) * 0.8;
    }
    const space: zssim.Space = .{ .transfer = .linear, .primaries = .rec2020, .scene_referred = true };
    const a = try ctx.prepareLinearRgbF32(lin, W, H, space);
    defer a.deinit();
    try testing.expectEqual(zssim.Encoding.ictcp, a.encoding);
    const a2 = try ctx.prepareLinearRgbF32(lin, W, H, space);
    defer a2.deinit();
    const r = try ctx.compare(a, a2);
    try testing.expectApproxEqAbs(@as(f64, 0.0), r.distance, 1e-6);
}
