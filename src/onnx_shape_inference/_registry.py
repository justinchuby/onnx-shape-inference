# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Registry for shape inference functions."""

from __future__ import annotations

__all__ = [
    "OpShapeInferenceRegistry",
    "registry",
]

import logging
from collections.abc import Callable, Iterator
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import onnx_ir as ir

    from onnx_shape_inference._context import ShapeInferenceContext

logger = logging.getLogger(__name__)

# Type alias for shape inference functions
ShapeInferenceFunc = Callable[["ShapeInferenceContext", "ir.Node"], None]


def _normalize_domain(domain: str) -> str:
    """Normalize domain so that 'ai.onnx' is treated the same as ''."""
    if domain == "ai.onnx":
        return ""
    return domain


class OpShapeInferenceRegistry:
    """Registry for operator shape inference functions.

    Supports registration by (domain, op_type) with since_version semantics.
    When looking up a function, dispatches to the correct version where
    target_version >= since_version and target_version < next_since_version.

    Example::

        from onnx_shape_inference import registry

        # Register with decorator
        @registry.register("", "Add", since_version=7)
        def infer_add_v7(ctx, node):
            ...

        @registry.register("", "Add", since_version=14)  # 14 and above
        def infer_add_v14(ctx, node):
            ...

        # Lookup
        func = registry.get("", "Add", version=13)  # Returns infer_add_v7
        func = registry.get("", "Add", version=14)  # Returns infer_add_v14
    """

    def __init__(self) -> None:
        # Raw registrations: {(domain, op_type): [(since_version, func), ...]}
        # Sorted by since_version ascending
        self._registrations: dict[tuple[str, str], list[tuple[int, ShapeInferenceFunc]]] = {}
        # Cached lookup table: {(domain, op_type): {version: func}}
        # Built on first lookup for each (domain, op_type)
        self._cache: dict[tuple[str, str], dict[int, ShapeInferenceFunc]] = {}
        # Track max since_version per key for O(1) lookup beyond cache
        self._max_version: dict[tuple[str, str], tuple[int, ShapeInferenceFunc]] = {}

    def register(
        self,
        domain: str,
        op_type: str,
        since_version: int = 1,
    ) -> Callable[[ShapeInferenceFunc], ShapeInferenceFunc]:
        """Register a shape inference function for an operator.

        Can be used as a decorator or called directly.

        Args:
            domain: ONNX domain (e.g., "", "com.microsoft").
            op_type: Operator type (e.g., "Add", "Transpose").
            since_version: The minimum opset version this function applies to.
                The function will be used for all versions >= since_version
                until a newer registration with a higher since_version exists.

        Returns:
            A decorator that registers the function.

        Example::

            @registry.register("", "Add", since_version=7)
            def infer_add_v7(ctx, node):
                ...

            @registry.register("", "Add", since_version=14)
            def infer_add_v14(ctx, node):
                ...
        """

        def decorator(func: ShapeInferenceFunc) -> ShapeInferenceFunc:
            key = (_normalize_domain(domain), op_type)

            if key not in self._registrations:
                self._registrations[key] = []

            self._registrations[key].append((since_version, func))
            # Keep sorted by since_version ascending
            self._registrations[key].sort(key=lambda x: x[0])

            # Invalidate cache for this key since registrations changed
            self._cache.pop(key, None)
            self._max_version.pop(key, None)

            logger.debug(
                "Registered shape inference for %s::%s (since_version=%s)",
                domain,
                op_type,
                since_version,
            )
            return func

        return decorator

    def _build_cache(self, key: tuple[str, str]) -> None:
        """Build the O(1) lookup cache for a given (domain, op_type) key."""
        if key not in self._registrations:
            return

        registrations = self._registrations[key]
        if not registrations:
            return

        cache: dict[int, ShapeInferenceFunc] = {}

        # Build cache: for each version from the lowest since_version to the highest,
        # map to the appropriate function
        for i, (since_ver, func) in enumerate(registrations):
            # Determine the end version (exclusive) for this registration
            if i + 1 < len(registrations):
                end_ver = registrations[i + 1][0]
            else:
                # Last registration - use since_ver as end (will handle larger versions specially)
                end_ver = since_ver + 1

            # Fill cache from since_ver to end_ver (exclusive)
            for ver in range(since_ver, end_ver):
                cache[ver] = func

        self._cache[key] = cache

        # Track the highest registration for O(1) lookup of versions beyond cache
        max_since, max_func = registrations[-1]
        self._max_version[key] = (max_since, max_func)

    def get(
        self,
        domain: str,
        op_type: str,
        version: int,
    ) -> ShapeInferenceFunc | None:
        """Get the shape inference function for an operator.

        Args:
            domain: ONNX domain.
            op_type: Operator type.
            version: Opset version to look up.

        Returns:
            The shape inference function, or None if not found.

        Complexity: O(1) after first lookup for a given (domain, op_type).
        """
        key = (_normalize_domain(domain), op_type)

        # Build cache if not already built
        if key not in self._cache and key in self._registrations:
            self._build_cache(key)

        # Check if we have any registrations for this key
        if key not in self._cache:
            return None

        cache = self._cache[key]

        # O(1) lookup in cache
        if version in cache:
            return cache[version]

        # Check if version is >= max since_version (O(1))
        if key in self._max_version:
            max_since, max_func = self._max_version[key]
            if version >= max_since:
                return max_func

        # Version is below all registered since_versions
        return None

    def has(self, domain: str, op_type: str) -> bool:
        """Check if any shape inference function is registered for an operator."""
        key = (_normalize_domain(domain), op_type)
        return key in self._registrations and len(self._registrations[key]) > 0

    def version_boundaries(self, domain: str, op_type: str) -> tuple[int, ...]:
        """Return the sorted ``since_version`` boundaries for an operator.

        Each value is a version at which the dispatched inference function
        changes.  Consumers such as the fuzzer use these to sample versions on
        and around dispatch edges, a common source of ``since_version`` bugs.

        Args:
            domain: ONNX domain (``"ai.onnx"`` is normalized to ``""``).
            op_type: Operator type (e.g. ``"Add"``).

        The built-in operator modules are imported automatically (via
        :meth:`collect`) so callers never observe a spuriously empty result
        just because inference has not run yet in this process.

        Returns:
            A tuple of ``since_version`` integers in ascending order, or an
            empty tuple if the operator is not registered.
        """
        self.collect()
        key = (_normalize_domain(domain), op_type)
        registrations = self._registrations.get(key)
        if not registrations:
            return ()
        # ``register`` keeps registrations sorted by since_version ascending.
        return tuple(since_version for since_version, _ in registrations)

    def iter_supported(self) -> Iterator[tuple[str, str, tuple[int, ...]]]:
        """Yield every registered operator with its version boundaries.

        Yields ``(domain, op_type, since_versions)`` triples in deterministic
        (sorted) order so that seed-driven consumers (e.g. the fuzzer) reproduce
        the same op sampling across runs.  ``since_versions`` is the ascending
        tuple of registered ``since_version`` values (see
        :meth:`version_boundaries`).  Domains are already normalized (so
        ``"ai.onnx"`` appears as ``""``).

        The built-in operator modules are imported automatically (via
        :meth:`collect`), so seed-driven consumers get the full supported set
        without having to call :meth:`collect` (or run inference) first.

        Example::

            for domain, op_type, versions in registry.iter_supported():
                ...
        """
        self.collect()
        for domain, op_type in sorted(self._registrations):
            registrations = self._registrations[(domain, op_type)]
            yield (
                domain,
                op_type,
                tuple(since_version for since_version, _ in registrations),
            )

    def collect(self) -> None:
        """Import all built-in op modules to populate the registry.

        Call this before using :meth:`get` outside of
        :func:`~onnx_shape_inference.infer_symbolic_shapes` (which calls it
        automatically).

        Example::

            from onnx_shape_inference import registry

            registry.collect()
            func = registry.get("", "Relu", version=21)
        """
        from onnx_shape_inference import _ops  # ruff:ignore[unused-import]

    def clear(self) -> None:
        """Clear all registered functions (mainly for testing)."""
        self._registrations.clear()
        self._cache.clear()
        self._max_version.clear()


# Global registry instance
registry = OpShapeInferenceRegistry()
