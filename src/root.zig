//! zssim — perceptual image similarity for Zig.
//!
//! A clean-room, from-scratch implementation of a multi-scale, perceptually
//! weighted structural-similarity metric (SSIM / MS-SSIM family) for comparing
//! images. It returns an `SSIM` score in `0..1` (1 = identical) and the
//! `distance = 1/SSIM - 1` dissimilarity in `0..∞` (0 = identical).
//!
//! This file is the module root: it re-exports the public API and pulls in
//! every source file so `zig build test` exercises the whole tree.
//!
//! See README.md for the algorithm overview, references, and roadmap.

const std = @import("std");

/// Library version (keep in sync with build.zig.zon).
pub const version = "0.1.0-dev";

// --- Public submodules ------------------------------------------------------
pub const color = @import("color.zig");
pub const mat3 = @import("mat3.zig");
pub const blur = @import("blur.zig");
pub const pyramid = @import("pyramid.zig");
pub const ssim = @import("ssim.zig");

test {
    // Aggregate tests from every source file. Add new modules here as they land.
    _ = color;
    _ = mat3;
    _ = blur;
    _ = pyramid;
    _ = ssim;
    std.testing.refAllDecls(@This());
}

test "version string" {
    try std.testing.expectEqualStrings("0.1.0-dev", version);
}
