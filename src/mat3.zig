//! Tiny 3×3 / 3-vector linear-algebra helpers used to derive RGB→XYZ matrices
//! and chromatic-adaptation transforms at comptime.
//!
//! All math is `f64` for derivation precision; callers cast the resulting
//! constant matrices to `f32` for the runtime per-pixel path.

const std = @import("std");

pub const Vec3 = [3]f64;

/// Row-major 3×3 matrix.
pub const Mat3 = struct {
    m: [3][3]f64,

    pub const identity: Mat3 = .{ .m = .{
        .{ 1, 0, 0 },
        .{ 0, 1, 0 },
        .{ 0, 0, 1 },
    } };

    pub fn mulVec(a: Mat3, v: Vec3) Vec3 {
        return .{
            a.m[0][0] * v[0] + a.m[0][1] * v[1] + a.m[0][2] * v[2],
            a.m[1][0] * v[0] + a.m[1][1] * v[1] + a.m[1][2] * v[2],
            a.m[2][0] * v[0] + a.m[2][1] * v[1] + a.m[2][2] * v[2],
        };
    }

    pub fn mul(a: Mat3, b: Mat3) Mat3 {
        var out: Mat3 = undefined;
        for (0..3) |i| {
            for (0..3) |j| {
                var s: f64 = 0;
                for (0..3) |k| s += a.m[i][k] * b.m[k][j];
                out.m[i][j] = s;
            }
        }
        return out;
    }

    /// Multiply on the right by a diagonal matrix given as a vector.
    pub fn mulDiag(a: Mat3, d: Vec3) Mat3 {
        var out: Mat3 = a;
        for (0..3) |i| {
            for (0..3) |j| out.m[i][j] = a.m[i][j] * d[j];
        }
        return out;
    }

    pub fn det(a: Mat3) f64 {
        return a.m[0][0] * (a.m[1][1] * a.m[2][2] - a.m[1][2] * a.m[2][1]) -
            a.m[0][1] * (a.m[1][0] * a.m[2][2] - a.m[1][2] * a.m[2][0]) +
            a.m[0][2] * (a.m[1][0] * a.m[2][1] - a.m[1][1] * a.m[2][0]);
    }

    pub fn inverse(a: Mat3) Mat3 {
        const d = a.det();
        std.debug.assert(d != 0);
        const inv = 1.0 / d;
        var out: Mat3 = undefined;
        out.m[0][0] = (a.m[1][1] * a.m[2][2] - a.m[1][2] * a.m[2][1]) * inv;
        out.m[0][1] = (a.m[0][2] * a.m[2][1] - a.m[0][1] * a.m[2][2]) * inv;
        out.m[0][2] = (a.m[0][1] * a.m[1][2] - a.m[0][2] * a.m[1][1]) * inv;
        out.m[1][0] = (a.m[1][2] * a.m[2][0] - a.m[1][0] * a.m[2][2]) * inv;
        out.m[1][1] = (a.m[0][0] * a.m[2][2] - a.m[0][2] * a.m[2][0]) * inv;
        out.m[1][2] = (a.m[0][2] * a.m[1][0] - a.m[0][0] * a.m[1][2]) * inv;
        out.m[2][0] = (a.m[1][0] * a.m[2][1] - a.m[1][1] * a.m[2][0]) * inv;
        out.m[2][1] = (a.m[0][1] * a.m[2][0] - a.m[0][0] * a.m[2][1]) * inv;
        out.m[2][2] = (a.m[0][0] * a.m[1][1] - a.m[0][1] * a.m[1][0]) * inv;
        return out;
    }

    /// Cast to a row-major `[3][3]f32` for the runtime path.
    pub fn toF32(a: Mat3) [3][3]f32 {
        var out: [3][3]f32 = undefined;
        for (0..3) |i| {
            for (0..3) |j| out[i][j] = @floatCast(a.m[i][j]);
        }
        return out;
    }
};

test "inverse round-trips to identity" {
    const a: Mat3 = .{ .m = .{
        .{ 0.5, 0.1, 0.2 },
        .{ 0.0, 1.3, 0.4 },
        .{ 0.7, 0.2, 1.1 },
    } };
    const id = a.mul(a.inverse());
    for (0..3) |i| {
        for (0..3) |j| {
            const expected: f64 = if (i == j) 1.0 else 0.0;
            try std.testing.expectApproxEqAbs(expected, id.m[i][j], 1e-12);
        }
    }
}

test "mulVec matches identity" {
    const v: Vec3 = .{ 0.2, 0.7, 0.4 };
    const r = Mat3.identity.mulVec(v);
    try std.testing.expectEqual(v, r);
}
