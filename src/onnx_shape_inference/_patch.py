"""Patch onnx-ir.SymbolicDim to support arithmetic operations for easier shape inference."""

from __future__ import annotations

from collections.abc import Mapping

import onnx_ir as ir
import onnx_ir._display
import sympy

from onnx_shape_inference import _symbolic_shapes


class SymbolicDim(ir.SymbolicDimProtocol, onnx_ir._display.PrettyPrintable):
    """Immutable symbolic dimension that can be shared across multiple shapes.

    SymbolicDim is used to represent a symbolic (non-integer) dimension in a tensor shape.
    It is immutable and can be compared or hashed. It supports SymPy expressions for
    symbolic arithmetic.

    Example::

        >>> import onnx_ir as ir
        >>> # Simple symbolic dimension (backward compatible)
        >>> batch = ir.SymbolicDim("batch")
        >>> batch.value
        'batch'
        >>> # Arithmetic operations return new SymbolicDim with SymPy expression
        >>> seq_plus_one = ir.SymbolicDim("seq_len") + 1
        >>> seq_plus_one.value
        'seq_len + 1'
        >>> # Evaluate with concrete values
        >>> seq_plus_one.evaluate({"seq_len": 128})
        129
    """

    __slots__ = ("_expr_cache", "_value")

    def __init__(self, value: str | sympy.Expr | None) -> None:
        """Initialize a symbolic dimension.

        Args:
            value: The value of the dimension. Can be:
                - A string: Represents a named symbolic dimension
                - None: Represents an unknown dimension
                - A SymPy expression: Used directly for symbolic arithmetic

        Raises:
            TypeError: If value is an int (use int directly in Shape instead).
        """
        if isinstance(value, int):
            raise TypeError(
                "The value of a SymbolicDim cannot be an int. "
                "If you are creating a Shape, use int directly instead of SymbolicDim."
            )

        # Lazy initialization - don't create sympy expression unless needed
        self._expr_cache: sympy.Expr | None = None

        if value is None:
            self._value: str | None = None
        elif isinstance(value, str):
            self._value = value
        elif isinstance(value, sympy.Expr):
            # For sympy expressions, store both value string and expression
            self._value = str(value)
            self._expr_cache = value
        else:
            raise TypeError(f"Expected str, None, or sympy.Expr, got {type(value).__name__}")

    def __eq__(self, other: object) -> bool:
        """Check equality with another SymbolicDim, string, or None."""
        if isinstance(other, SymbolicDim):
            return self._value == other._value
        if isinstance(other, str):
            return self._value == other
        if other is None:
            return self._value is None
        # TODO(justinchuby): Consider supporting equality with sympy.Expr directly
        return False

    def __hash__(self) -> int:
        """Return the hash of the symbolic dimension."""
        return hash(self._value)

    @property
    def value(self) -> str | None:
        """The value of the symbolic dimension as a string.

        Returns the string representation, or None for unknown dimensions.
        """
        return self._value

    @property
    def _expr(self) -> sympy.Expr | None:
        """The underlying SymPy expression (lazily created).

        Returns the SymPy expression for this dimension, or None for unknown dimensions.

        The expression is parsed from the string representation and supports:
            - Basic arithmetic: +, -, *, /, //, **
            - Functions: max(), min(), floor(), sqrt()
            - Symbolic variables (identifiers)
            - Integer literals
        """
        if self._value is None:
            return None
        if self._expr_cache is None:
            self._expr_cache = _symbolic_shapes.parse_symbolic_expression(self._value)
        return self._expr_cache

    def __add__(self, other: int | SymbolicDim) -> SymbolicDim:
        """Add an integer or another SymbolicDim to this dimension."""
        if self._expr is None:
            return SymbolicDim(None)
        if isinstance(other, int):
            return SymbolicDim(sympy.sympify(self._expr + other))
        if isinstance(other, SymbolicDim):
            if other._value is None:
                return SymbolicDim(None)
            return SymbolicDim(sympy.sympify(self._expr + other._expr))
        return NotImplemented

    def __radd__(self, other: int) -> SymbolicDim:
        """Support int + SymbolicDim."""
        if isinstance(other, int):
            return self.__add__(other)
        return NotImplemented

    def __sub__(self, other: int | SymbolicDim) -> SymbolicDim:
        """Subtract an integer or another SymbolicDim from this dimension."""
        if self._expr is None:
            return SymbolicDim(None)
        if isinstance(other, int):
            return SymbolicDim(sympy.sympify(self._expr - other))
        if isinstance(other, SymbolicDim):
            if other._value is None:
                return SymbolicDim(None)
            return SymbolicDim(sympy.sympify(self._expr - other._expr))
        return NotImplemented

    def __rsub__(self, other: int) -> SymbolicDim:
        """Support int - SymbolicDim."""
        if self._expr is None:
            return SymbolicDim(None)
        if isinstance(other, int):
            return SymbolicDim(sympy.sympify(other - self._expr))
        return NotImplemented

    def __mul__(self, other: int | SymbolicDim) -> SymbolicDim:
        """Multiply this dimension by an integer or another SymbolicDim."""
        if self._expr is None:
            return SymbolicDim(None)
        if isinstance(other, int):
            return SymbolicDim(sympy.sympify(self._expr * other))
        if isinstance(other, SymbolicDim):
            if other._value is None:
                return SymbolicDim(None)
            return SymbolicDim(sympy.sympify(self._expr * other._expr))
        return NotImplemented

    def __rmul__(self, other: int) -> SymbolicDim:
        """Support int * SymbolicDim."""
        if isinstance(other, int):
            return self.__mul__(other)
        return NotImplemented

    def __floordiv__(self, other: int | SymbolicDim) -> SymbolicDim:
        """Floor divide this dimension by an integer or another SymbolicDim."""
        if self._expr is None:
            return SymbolicDim(None)
        if isinstance(other, int):
            return SymbolicDim(sympy.sympify(self._expr // other))
        if isinstance(other, SymbolicDim):
            if other._value is None:
                return SymbolicDim(None)
            return SymbolicDim(sympy.sympify(self._expr // other._expr))
        return NotImplemented

    def __truediv__(self, other: int | SymbolicDim) -> SymbolicDim:
        """Divide this dimension by an integer or another SymbolicDim (rational)."""
        if self._expr is None:
            return SymbolicDim(None)
        if isinstance(other, int):
            return SymbolicDim(sympy.sympify(sympy.Rational(1, other) * self._expr))
        if isinstance(other, SymbolicDim):
            if other._value is None:
                return SymbolicDim(None)
            return SymbolicDim(sympy.sympify(self._expr / other._expr))
        return NotImplemented

    def __rtruediv__(self, other: int) -> SymbolicDim:
        """Support int / SymbolicDim."""
        if self._expr is None:
            return SymbolicDim(None)
        if isinstance(other, int):
            return SymbolicDim(sympy.sympify(other / self._expr))
        return NotImplemented

    def __mod__(self, other: int | SymbolicDim) -> SymbolicDim:
        """Compute modulo of this dimension by an integer or another SymbolicDim."""
        if self._expr is None:
            return SymbolicDim(None)
        if isinstance(other, int):
            return SymbolicDim(sympy.sympify(self._expr % other))
        if isinstance(other, SymbolicDim):
            if other._value is None:
                return SymbolicDim(None)
            return SymbolicDim(sympy.sympify(self._expr % other._expr))
        return NotImplemented

    def __ceil__(self) -> SymbolicDim:
        """Support math.ceil(dim). Returns a SymbolicDim with ceiling expression."""
        if self._expr is None:
            return SymbolicDim(None)
        return SymbolicDim(sympy.ceiling(self._expr))

    def __floor__(self) -> SymbolicDim:
        """Support math.floor(dim). Returns a SymbolicDim with floor expression."""
        if self._expr is None:
            return SymbolicDim(None)
        return SymbolicDim(sympy.floor(self._expr))

    def __trunc__(self) -> SymbolicDim:
        """Support math.trunc(dim). Returns a SymbolicDim truncated toward zero."""
        if self._expr is None:
            return SymbolicDim(None)
        return SymbolicDim(sympy.sign(self._expr) * sympy.floor(sympy.Abs(self._expr)))

    def __neg__(self) -> SymbolicDim:
        """Negate this dimension."""
        if self._expr is None:
            return SymbolicDim(None)
        return SymbolicDim(sympy.sympify(-self._expr))

    def simplify(self) -> SymbolicDim:
        """Return a new SymbolicDim with the expression simplified.

        Uses SymPy's simplify function to reduce the expression.

        Returns:
            A new SymbolicDim with simplified expression.
        """
        if self._expr is None:
            return SymbolicDim(None)
        return SymbolicDim(sympy.simplify(self._expr))

    def evaluate(self, bindings: Mapping[str, int]) -> int | SymbolicDim:
        """Evaluate the symbolic dimension with concrete values.

        Args:
            bindings: A mapping from symbol names to integer values.

        Returns:
            The concrete integer value if fully evaluated, or a SymbolicDim
            containing the partially evaluated expression or None.

        Example::

            >>> dim = ir.SymbolicDim("N") + 1
            >>> dim.evaluate({"N": 10})
            11
        """
        if self._expr is None:
            return SymbolicDim(None)
        # Build substitution map using the actual symbols present in the expression
        subs = {
            symbol: bindings[str(symbol)]
            for symbol in self._expr.free_symbols
            if str(symbol) in bindings
        }
        result = self._expr.subs(subs)
        if result.is_number and result.is_integer:
            return int(result)
        return SymbolicDim(result)

    def free_symbols(self) -> frozenset[str]:
        """Return the set of free symbol names in this dimension.

        Returns:
            A frozenset of symbol names that appear in the expression.
        """
        if self._expr is None:
            return frozenset()
        return frozenset(str(s) for s in self._expr.free_symbols)

    def __str__(self) -> str:
        return str(self._value) if self._value is not None else "None"

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self._value!r})"


ir.SymbolicDim = SymbolicDim  # Patch the onnx_ir.SymbolicDim to our enhanced version
ir._core.SymbolicDim = SymbolicDim
