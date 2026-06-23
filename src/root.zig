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

test {
    // Aggregate tests from every source file. Add new modules here as they land.
    std.testing.refAllDecls(@This());
}

test "version string" {
    try std.testing.expectEqualStrings("0.1.0-dev", version);
}
