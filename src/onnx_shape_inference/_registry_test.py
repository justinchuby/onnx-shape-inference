# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Tests for OpShapeInferenceRegistry."""

from __future__ import annotations

import unittest

from onnx_shape_inference._registry import OpShapeInferenceRegistry


class OpShapeInferenceRegistryTest(unittest.TestCase):
    """Tests for OpShapeInferenceRegistry."""

    def setUp(self):
        # Use a fresh registry for each test
        self.registry = OpShapeInferenceRegistry()

    def test_register_with_since_version(self):
        @self.registry.register("", "TestOp", since_version=7)
        def infer_test(ctx, node):
            pass

        # Should work for version 7 and above
        self.assertIsNotNone(self.registry.get("", "TestOp", version=7))
        self.assertIsNotNone(self.registry.get("", "TestOp", version=10))
        self.assertIsNotNone(self.registry.get("", "TestOp", version=20))
        # Should not work below version 7
        self.assertIsNone(self.registry.get("", "TestOp", version=6))

    def test_register_default_since_version(self):
        @self.registry.register("", "TestOp")
        def infer_test(ctx, node):
            pass

        # Default since_version=1, so should work for version 1 and above
        self.assertIsNotNone(self.registry.get("", "TestOp", version=1))
        self.assertIsNotNone(self.registry.get("", "TestOp", version=100))
        # Should not work below version 1
        self.assertIsNone(self.registry.get("", "TestOp", version=0))

    def test_has(self):
        @self.registry.register("", "TestOp", since_version=1)
        def infer_test(ctx, node):
            pass

        self.assertTrue(self.registry.has("", "TestOp"))
        self.assertFalse(self.registry.has("", "NonExistent"))

    def test_multiple_version_registrations(self):
        @self.registry.register("", "TestOp", since_version=7)
        def infer_v7(ctx, node):
            return "v7"

        @self.registry.register("", "TestOp", since_version=14)
        def infer_v14(ctx, node):
            return "v14"

        # Version 6 should return None (below all registrations)
        self.assertIsNone(self.registry.get("", "TestOp", version=6))

        # Version 7-13 should get v7 handler
        func7 = self.registry.get("", "TestOp", version=7)
        self.assertEqual(func7(None, None), "v7")

        func10 = self.registry.get("", "TestOp", version=10)
        self.assertEqual(func10(None, None), "v7")

        func13 = self.registry.get("", "TestOp", version=13)
        self.assertEqual(func13(None, None), "v7")

        # Version 14 and above should get v14 handler
        func14 = self.registry.get("", "TestOp", version=14)
        self.assertEqual(func14(None, None), "v14")

        func20 = self.registry.get("", "TestOp", version=20)
        self.assertEqual(func20(None, None), "v14")

    def test_lookup_is_o1_after_first_access(self):
        """Test that lookup uses cached dict for O(1) access."""

        @self.registry.register("", "TestOp", since_version=7)
        def infer_v7(ctx, node):
            return "v7"

        @self.registry.register("", "TestOp", since_version=14)
        def infer_v14(ctx, node):
            return "v14"

        # First access builds the cache
        self.registry.get("", "TestOp", version=10)

        # Verify cache was built
        key = ("", "TestOp")
        self.assertIn(key, self.registry._cache)
        self.assertIn(key, self.registry._max_version)

        # Cache should have versions 7-13 mapped to v7
        cache = self.registry._cache[key]
        for v in range(7, 14):
            self.assertIn(v, cache)

        # Max version should be (14, infer_v14)
        max_since, max_func = self.registry._max_version[key]
        self.assertEqual(max_since, 14)
        self.assertEqual(max_func(None, None), "v14")

    def test_cache_invalidation_on_new_registration(self):
        """Test that cache is invalidated when new registration is added."""

        @self.registry.register("", "TestOp", since_version=7)
        def infer_v7(ctx, node):
            return "v7"

        # Build cache
        self.registry.get("", "TestOp", version=10)

        key = ("", "TestOp")
        self.assertIn(key, self.registry._cache)

        # Add new registration
        @self.registry.register("", "TestOp", since_version=14)
        def infer_v14(ctx, node):
            return "v14"

        # Cache should be invalidated
        self.assertNotIn(key, self.registry._cache)

        # New lookup should work correctly
        func20 = self.registry.get("", "TestOp", version=20)
        self.assertEqual(func20(None, None), "v14")

    def test_version_boundaries_returns_sorted_since_versions(self):
        @self.registry.register("", "TestOp", since_version=14)
        def infer_v14(ctx, node):
            pass

        @self.registry.register("", "TestOp", since_version=7)
        def infer_v7(ctx, node):
            pass

        self.assertEqual(self.registry.version_boundaries("", "TestOp"), (7, 14))

    def test_version_boundaries_normalizes_domain(self):
        @self.registry.register("", "TestOp", since_version=7)
        def infer_test(ctx, node):
            pass

        # "ai.onnx" must normalize to "" like get()/register() do.
        self.assertEqual(self.registry.version_boundaries("ai.onnx", "TestOp"), (7,))

    def test_version_boundaries_unregistered_is_empty(self):
        self.assertEqual(self.registry.version_boundaries("", "Missing"), ())

    def test_iter_supported_is_deterministic_and_sorted(self):
        @self.registry.register("com.microsoft", "ZOp", since_version=1)
        def infer_z(ctx, node):
            pass

        @self.registry.register("", "BOp", since_version=13)
        def infer_b13(ctx, node):
            pass

        @self.registry.register("", "BOp", since_version=1)
        def infer_b1(ctx, node):
            pass

        @self.registry.register("", "AOp", since_version=1)
        def infer_a(ctx, node):
            pass

        supported = list(self.registry.iter_supported())
        self.assertEqual(
            supported,
            [
                ("", "AOp", (1,)),
                ("", "BOp", (1, 13)),
                ("com.microsoft", "ZOp", (1,)),
            ],
        )
        # Deterministic across calls (seed reproducibility contract).
        self.assertEqual(supported, list(self.registry.iter_supported()))

    def test_iter_supported_empty_registry(self):
        self.assertEqual(list(self.registry.iter_supported()), [])

    def test_iter_supported_matches_version_boundaries(self):
        @self.registry.register("", "TestOp", since_version=7)
        def infer_v7(ctx, node):
            pass

        @self.registry.register("", "TestOp", since_version=14)
        def infer_v14(ctx, node):
            pass

        for domain, op_type, versions in self.registry.iter_supported():
            self.assertEqual(versions, self.registry.version_boundaries(domain, op_type))


if __name__ == "__main__":
    unittest.main()
