const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // Public library module: dependents import it as `@import("zssim")`.
    const mod = b.addModule("zssim", .{
        .root_source_file = b.path("src/root.zig"),
        .target = target,
        .optimize = optimize,
    });

    // Unit tests cover the whole module tree (root.zig pulls in every file).
    const mod_tests = b.addTest(.{ .root_module = mod });
    const run_mod_tests = b.addRunArtifact(mod_tests);

    const test_step = b.step("test", "Run unit tests");
    test_step.dependOn(&run_mod_tests.step);

    // `zig build check` — type-check quickly without running anything.
    const check = b.step("check", "Type-check the library");
    check.dependOn(&mod_tests.step);
}
