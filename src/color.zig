//! Color science for zssim: transfer functions, RGB primaries, and the
//! perceptually-uniform encodings used by the SSIM core.
//!
//! Design (see README "Color management: division of labor"): zssim never
//! touches ICC profiles. The caller (e.g. ImProc via lcms2/OCIO) flattens any
//! source profile to a *known working space*, then tags the buffer with a
//! `Space { transfer, primaries, scene_referred }`. From that tag, this module:
//!
//!   1. linearizes encoded values (sRGB / linear / gamma / PQ / HLG),
//!   2. converts working-primaries linear RGB → CIE XYZ (D65) using a matrix
//!      derived at comptime from raw chromaticities, with Bradford chromatic
//!      adaptation so non-D65 white points (e.g. ACES ~D60) are handled,
//!   3. encodes to a channel-separated, perceptually-uniform triple:
//!        - SDR  → CIE L*a*b* normalized to ~[0,1]
//!        - HDR/scene-referred → ICtCp (PQ-based), which stays uniform across a
//!          high dynamic range where CIELAB's SDR cube-root would clip.
//!
//! All algorithms are from public standards (sRGB IEC 61966-2-1, CIE 15,
//! ITU-R BT.2100, SMPTE ST 2084/2065-1, Bradford CAT). Nothing here is derived
//! from another implementation.

const std = @import("std");
const Mat3 = @import("mat3.zig").Mat3;
const Vec3 = @import("mat3.zig").Vec3;

// ---------------------------------------------------------------------------
// Transfer functions (encoded signal -> scene/display linear light)
// ---------------------------------------------------------------------------

/// How encoded pixel values map to linear light.
pub const Transfer = union(enum) {
    /// Already linear; identity.
    linear,
    /// sRGB EOTF (IEC 61966-2-1).
    srgb,
    /// Pure power law with the given display gamma exponent (e.g. 2.2, 2.4).
    gamma: f32,
    /// SMPTE ST 2084 "PQ". Output is normalized so 1.0 == 10000 cd/m².
    pq,
    /// ITU-R BT.2100 HLG inverse-OETF (scene linear, 1.0 == reference white).
    hlg,

    /// Convert a single encoded component in [0,1] (PQ/HLG use the full signal)
    /// to linear light.
    pub fn decode(self: Transfer, v: f32) f32 {
        return switch (self) {
            .linear => v,
            .srgb => srgbToLinear(v),
            .gamma => |g| if (v <= 0) 0 else std.math.pow(f32, v, g),
            .pq => pqToLinear(v),
            .hlg => hlgToLinear(v),
        };
    }
};

pub fn srgbToLinear(s: f32) f32 {
    if (s <= 0.04045) return s / 12.92;
    return std.math.pow(f32, (s + 0.055) / 1.055, 2.4);
}

pub fn linearToSrgb(l: f32) f32 {
    if (l <= 0.0031308) return l * 12.92;
    return 1.055 * std.math.pow(f32, l, 1.0 / 2.4) - 0.055;
}

/// SMPTE ST 2084 (PQ) EOTF. Input is the encoded signal in [0,1]; output is
/// display luminance normalized so 1.0 == 10000 cd/m².
pub fn pqToLinear(e: f32) f32 {
    const m1: f32 = 0.1593017578125;
    const m2: f32 = 78.84375;
    const c1: f32 = 0.8359375;
    const c2: f32 = 18.8515625;
    const c3: f32 = 18.6875;
    if (e <= 0) return 0;
    const ep = std.math.pow(f32, e, 1.0 / m2);
    const num = @max(ep - c1, 0.0);
    const den = c2 - c3 * ep;
    if (den <= 0) return 1.0;
    return std.math.pow(f32, num / den, 1.0 / m1);
}

/// Inverse of the PQ EOTF: linear (1.0 == 10000 cd/m²) -> encoded signal.
pub fn linearToPq(y: f32) f32 {
    const m1: f32 = 0.1593017578125;
    const m2: f32 = 78.84375;
    const c1: f32 = 0.8359375;
    const c2: f32 = 18.8515625;
    const c3: f32 = 18.6875;
    if (y <= 0) return 0;
    const yp = std.math.pow(f32, y, m1);
    return std.math.pow(f32, (c1 + c2 * yp) / (1.0 + c3 * yp), m2);
}

/// ITU-R BT.2100 HLG inverse-OETF: encoded [0,1] -> scene linear [0,1].
pub fn hlgToLinear(e: f32) f32 {
    const a: f32 = 0.17883277;
    const b: f32 = 0.28466892; // 1 - 4a
    const c: f32 = 0.55991073; // 0.5 - a*ln(4a)
    if (e <= 0.5) return (e * e) / 3.0;
    return (@exp((e - c) / a) + b) / 12.0;
}

// ---------------------------------------------------------------------------
// White points & RGB primaries
// ---------------------------------------------------------------------------

pub const WhitePoint = struct {
    x: f64,
    y: f64,
    /// CIE XYZ of the white with Y normalized to 1.
    pub fn xyz(self: WhitePoint) Vec3 {
        return .{ self.x / self.y, 1.0, (1.0 - self.x - self.y) / self.y };
    }
};

pub const white_d65: WhitePoint = .{ .x = 0.31270, .y = 0.32900 };
pub const white_d60_aces: WhitePoint = .{ .x = 0.32168, .y = 0.33767 };
pub const white_d50: WhitePoint = .{ .x = 0.34570, .y = 0.35850 };

/// RGB primaries as CIE xy chromaticities plus a reference white.
pub const Chromaticities = struct {
    rx: f64,
    ry: f64,
    gx: f64,
    gy: f64,
    bx: f64,
    by: f64,
    white: WhitePoint,
};

/// Built-in working-space primaries. ICC handling lives in the host; these are
/// the spaces a host typically normalizes to.
pub const Primaries = enum {
    /// sRGB / Rec.709 (D65).
    srgb,
    /// Display P3 (DCI-P3 primaries, D65 white).
    display_p3,
    /// ITU-R BT.2020 / Rec.2100 (D65).
    rec2020,
    /// Adobe RGB (1998) (D65).
    adobe_rgb,
    /// ACES2065-1 AP0 (D60-ish ACES white) — ultra-wide, encloses the locus.
    aces_ap0,
    /// ACEScg AP1 (D60-ish ACES white).
    aces_ap1,

    pub fn chromaticities(self: Primaries) Chromaticities {
        return switch (self) {
            .srgb => .{ .rx = 0.640, .ry = 0.330, .gx = 0.300, .gy = 0.600, .bx = 0.150, .by = 0.060, .white = white_d65 },
            .display_p3 => .{ .rx = 0.680, .ry = 0.320, .gx = 0.265, .gy = 0.690, .bx = 0.150, .by = 0.060, .white = white_d65 },
            .rec2020 => .{ .rx = 0.708, .ry = 0.292, .gx = 0.170, .gy = 0.797, .bx = 0.131, .by = 0.046, .white = white_d65 },
            .adobe_rgb => .{ .rx = 0.640, .ry = 0.330, .gx = 0.210, .gy = 0.710, .bx = 0.150, .by = 0.060, .white = white_d65 },
            .aces_ap0 => .{ .rx = 0.7347, .ry = 0.2653, .gx = 0.0000, .gy = 1.0000, .bx = 0.0001, .by = -0.0770, .white = white_d60_aces },
            .aces_ap1 => .{ .rx = 0.713, .ry = 0.293, .gx = 0.165, .gy = 0.830, .bx = 0.128, .by = 0.044, .white = white_d60_aces },
        };
    }
};

/// Derive the RGB→XYZ matrix for a set of primaries (in their native white),
/// following the standard method (SMPTE RP 177): build the primary matrix,
/// solve for per-channel luminance scalars from the white point.
fn deriveRgbToXyz(c: Chromaticities) Mat3 {
    const zr = 1.0 - c.rx - c.ry;
    const zg = 1.0 - c.gx - c.gy;
    const zb = 1.0 - c.bx - c.by;
    const primary: Mat3 = .{ .m = .{
        .{ c.rx / c.ry, c.gx / c.gy, c.bx / c.by },
        .{ 1.0, 1.0, 1.0 },
        .{ zr / c.ry, zg / c.gy, zb / c.by },
    } };
    const s = primary.inverse().mulVec(c.white.xyz());
    return primary.mulDiag(s);
}

/// Bradford chromatic-adaptation transform from `src` white to `dst` white.
fn bradford(src: WhitePoint, dst: WhitePoint) Mat3 {
    const b: Mat3 = .{ .m = .{
        .{ 0.8951, 0.2664, -0.1614 },
        .{ -0.7502, 1.7135, 0.0367 },
        .{ 0.0389, -0.0685, 1.0296 },
    } };
    const src_lms = b.mulVec(src.xyz());
    const dst_lms = b.mulVec(dst.xyz());
    const ratio: Vec3 = .{ dst_lms[0] / src_lms[0], dst_lms[1] / src_lms[1], dst_lms[2] / src_lms[2] };
    // A = B^-1 * diag(ratio) * B
    return b.inverse().mulDiag(ratio).mul(b);
}

/// RGB→XYZ for the given primaries, Bradford-adapted to D65 (the reference
/// white the CIELAB step expects). Computed once at comptime per primaries.
pub fn rgbToXyzD65(comptime p: Primaries) [3][3]f32 {
    return comptime blk: {
        const c = p.chromaticities();
        const native = deriveRgbToXyz(c);
        const adapted = bradford(c.white, white_d65).mul(native);
        break :blk adapted.toF32();
    };
}

/// XYZ(D65)→linear Rec.2020 RGB, for routing arbitrary working spaces into the
/// ICtCp encoder (which is defined on Rec.2020 linear).
pub fn xyzD65ToRec2020() [3][3]f32 {
    return comptime blk: {
        const c = Primaries.rec2020.chromaticities();
        const fwd = deriveRgbToXyz(c); // rec2020 white is already D65
        break :blk fwd.inverse().toF32();
    };
}

fn matVec(m: [3][3]f32, v: [3]f32) [3]f32 {
    return .{
        m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
        m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
        m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
    };
}

// ---------------------------------------------------------------------------
// CIE L*a*b* (D65) — the SDR perceptual encoding
// ---------------------------------------------------------------------------

const lab_eps: f32 = 216.0 / 24389.0; // (6/29)^3
const lab_kappa: f32 = 24389.0 / 27.0; // (29/3)^3

fn labF(t: f32) f32 {
    if (t > lab_eps) return std.math.cbrt(t);
    return (lab_kappa * t + 16.0) / 116.0;
}

/// Working-primaries linear RGB → CIE L*a*b*. Returns `.{ L*, a*, b* }` with
/// L* in [0,100], a*/b* roughly in [-128,128].
pub fn linearRgbToLab(comptime p: Primaries, rgb: [3]f32) [3]f32 {
    const xyz = matVec(rgbToXyzD65(p), rgb);
    // D65 reference white, derived from the *same* chromaticity constant used to
    // build the matrices, so neutral RGB maps to a*=b*=0 exactly (no drift).
    const wn = comptime white_d65.xyz();
    const xn: f32 = @floatCast(wn[0]);
    const yn: f32 = @floatCast(wn[1]);
    const zn: f32 = @floatCast(wn[2]);
    const fx = labF(@max(xyz[0] / xn, 0.0));
    const fy = labF(@max(xyz[1] / yn, 0.0));
    const fz = labF(@max(xyz[2] / zn, 0.0));
    return .{
        116.0 * fy - 16.0,
        500.0 * (fx - fy),
        200.0 * (fy - fz),
    };
}

/// L*a*b* normalized to ~[0,1] per channel for the SSIM core (range = 1, so the
/// SSIM constants C1=(0.01)², C2=(0.03)² are meaningful). a*/b* are mapped from
/// [-128,128] → [0,1] and clamped (wide-gamut colors may exceed the box).
pub fn labNormalized(comptime p: Primaries, rgb: [3]f32) [3]f32 {
    const lab = linearRgbToLab(p, rgb);
    return .{
        std.math.clamp(lab[0] / 100.0, 0.0, 1.0),
        std.math.clamp((lab[1] + 128.0) / 255.0, 0.0, 1.0),
        std.math.clamp((lab[2] + 128.0) / 255.0, 0.0, 1.0),
    };
}

/// Grayscale fast path: a linear gray value equals relative luminance Y for a
/// neutral pixel, so its perceptual lightness is L* normalized to [0,1].
pub fn grayToLNorm(y: f32) f32 {
    const l = 116.0 * labF(@max(y, 0.0)) - 16.0;
    return std.math.clamp(l / 100.0, 0.0, 1.0);
}

// ---------------------------------------------------------------------------
// ICtCp (ITU-R BT.2100) — the HDR / scene-referred perceptual encoding
// ---------------------------------------------------------------------------

/// Working-primaries linear RGB → ICtCp, normalized to ~[0,1] per channel.
/// Input linear is assumed scaled so 1.0 == 10000 cd/m² (PQ normalization);
/// scene-referred callers scale by their peak before calling.
pub fn ictcpNormalized(comptime p: Primaries, rgb: [3]f32) [3]f32 {
    // Route any working space into Rec.2020 linear via XYZ.
    const xyz = matVec(rgbToXyzD65(p), rgb);
    const r2020 = matVec(xyzD65ToRec2020(), xyz);
    // Rec.2020 linear RGB → LMS (BT.2100 crosstalk matrix).
    const l = (1688.0 * r2020[0] + 2146.0 * r2020[1] + 262.0 * r2020[2]) / 4096.0;
    const m = (683.0 * r2020[0] + 2951.0 * r2020[1] + 462.0 * r2020[2]) / 4096.0;
    const s = (99.0 * r2020[0] + 309.0 * r2020[1] + 3688.0 * r2020[2]) / 4096.0;
    const lp = linearToPq(@max(l, 0.0));
    const mp = linearToPq(@max(m, 0.0));
    const sp = linearToPq(@max(s, 0.0));
    const i = 0.5 * lp + 0.5 * mp;
    const ct = (6610.0 * lp - 13613.0 * mp + 7003.0 * sp) / 4096.0;
    const cp = (17933.0 * lp - 17390.0 * mp - 543.0 * sp) / 4096.0;
    // I is already [0,1]. Ct/Cp are roughly [-0.5,0.5]; shift to [0,1].
    return .{
        std.math.clamp(i, 0.0, 1.0),
        std.math.clamp(ct + 0.5, 0.0, 1.0),
        std.math.clamp(cp + 0.5, 0.0, 1.0),
    };
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

test "srgb transfer round-trips" {
    var v: f32 = 0.0;
    while (v <= 1.0) : (v += 0.05) {
        const back = linearToSrgb(srgbToLinear(v));
        try std.testing.expectApproxEqAbs(v, back, 1e-5);
    }
}

test "srgb known points" {
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), srgbToLinear(0.0), 1e-7);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), srgbToLinear(1.0), 1e-6);
    // 0.5 sRGB ≈ 0.214 linear
    try std.testing.expectApproxEqAbs(@as(f32, 0.21404), srgbToLinear(0.5), 1e-4);
}

test "pq transfer round-trips" {
    var e: f32 = 0.0;
    while (e <= 1.0) : (e += 0.05) {
        const back = linearToPq(pqToLinear(e));
        try std.testing.expectApproxEqAbs(e, back, 1e-4);
    }
}

test "sRGB white maps to L*≈100, neutral a*b*" {
    const lab = linearRgbToLab(.srgb, .{ 1.0, 1.0, 1.0 });
    try std.testing.expectApproxEqAbs(@as(f32, 100.0), lab[0], 1e-2);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), lab[1], 1e-2);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), lab[2], 1e-2);
}

test "black maps to L*=0" {
    const lab = linearRgbToLab(.srgb, .{ 0.0, 0.0, 0.0 });
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), lab[0], 1e-4);
}

test "ACES AP0 white maps to L*≈100, near-neutral chroma" {
    // AP0 (1,1,1) is the ACES white; after Bradford D60→D65 it should read as a
    // near-neutral, full-lightness color — proving the wide-gamut + non-D65
    // path is wired correctly rather than mangled by an sRGB assumption.
    const lab = linearRgbToLab(.aces_ap0, .{ 1.0, 1.0, 1.0 });
    try std.testing.expectApproxEqAbs(@as(f32, 100.0), lab[0], 0.5);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), lab[1], 2.0);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), lab[2], 2.0);
}

test "rec2020 primaries derive a sane matrix (white -> XYZ D65)" {
    const xyz = matVec(rgbToXyzD65(.rec2020), .{ 1.0, 1.0, 1.0 });
    try std.testing.expectApproxEqAbs(@as(f32, 0.95047), xyz[0], 1e-3);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), xyz[1], 1e-3);
    try std.testing.expectApproxEqAbs(@as(f32, 1.08883), xyz[2], 1e-3);
}

test "ictcp neutral has near-zero chroma offsets" {
    // A neutral mid-gray should land near Ct=Cp=0.5 (the shifted zero point).
    const v = ictcpNormalized(.rec2020, .{ 0.1, 0.1, 0.1 });
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), v[1], 1e-3);
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), v[2], 1e-3);
}
