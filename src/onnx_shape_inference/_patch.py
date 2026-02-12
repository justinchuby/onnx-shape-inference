"""Patch onnx-ir.SymbolicDim to support arithmetic operations for easier shape inference."""

from __future__ import annotations

import onnx_ir as ir
import sympy


def _add__(self, other: int | ir.SymbolicDim) -> ir.SymbolicDim:
    """Add an integer or another ir.SymbolicDim to this dimension."""
    if self._expr is None:
        return ir.SymbolicDim(None)
    if isinstance(other, int):
        return ir.SymbolicDim(sympy.sympify(self._expr + other))
    if isinstance(other, ir.SymbolicDim):
        if other._value is None:
            return ir.SymbolicDim(None)
        return ir.SymbolicDim(sympy.sympify(self._expr + other._expr))
    return NotImplemented

def _radd__(self, other: int) -> ir.SymbolicDim:
    """Support int + ir.SymbolicDim."""
    if isinstance(other, int):
        return self.__add__(other)
    return NotImplemented

def _sub__(self, other: int | ir.SymbolicDim) -> ir.SymbolicDim:
    """Subtract an integer or another ir.SymbolicDim from this dimension."""
    if self._expr is None:
        return ir.SymbolicDim(None)
    if isinstance(other, int):
        return ir.SymbolicDim(sympy.sympify(self._expr - other))
    if isinstance(other, ir.SymbolicDim):
        if other._value is None:
            return ir.SymbolicDim(None)
        return ir.SymbolicDim(sympy.sympify(self._expr - other._expr))
    return NotImplemented

def _rsub__(self, other: int) -> ir.SymbolicDim:
    """Support int - ir.SymbolicDim."""
    if self._expr is None:
        return ir.SymbolicDim(None)
    if isinstance(other, int):
        return ir.SymbolicDim(sympy.sympify(other - self._expr))
    return NotImplemented

def _mul__(self, other: int | ir.SymbolicDim) -> ir.SymbolicDim:
    """Multiply this dimension by an integer or another ir.SymbolicDim."""
    if self._expr is None:
        return ir.SymbolicDim(None)
    if isinstance(other, int):
        return ir.SymbolicDim(sympy.sympify(self._expr * other))
    if isinstance(other, ir.SymbolicDim):
        if other._value is None:
            return ir.SymbolicDim(None)
        return ir.SymbolicDim(sympy.sympify(self._expr * other._expr))
    return NotImplemented

def _rmul__(self, other: int) -> ir.SymbolicDim:
    """Support int * ir.SymbolicDim."""
    if isinstance(other, int):
        return self.__mul__(other)
    return NotImplemented

def _floordiv__(self, other: int | ir.SymbolicDim) -> ir.SymbolicDim:
    """Floor divide this dimension by an integer or another ir.SymbolicDim."""
    if self._expr is None:
        return ir.SymbolicDim(None)
    if isinstance(other, int):
        return ir.SymbolicDim(sympy.sympify(self._expr // other))
    if isinstance(other, ir.SymbolicDim):
        if other._value is None:
            return ir.SymbolicDim(None)
        return ir.SymbolicDim(sympy.sympify(self._expr // other._expr))
    return NotImplemented

def _truediv__(self, other: int | ir.SymbolicDim) -> ir.SymbolicDim:
    """Divide this dimension by an integer or another ir.SymbolicDim (rational)."""
    if self._expr is None:
        return ir.SymbolicDim(None)
    if isinstance(other, int):
        return ir.SymbolicDim(sympy.sympify(sympy.Rational(1, other) * self._expr))
    if isinstance(other, ir.SymbolicDim):
        if other._value is None:
            return ir.SymbolicDim(None)
        return ir.SymbolicDim(sympy.sympify(self._expr / other._expr))
    return NotImplemented

def _rtruediv__(self, other: int) -> ir.SymbolicDim:
    """Support int / ir.SymbolicDim."""
    if self._expr is None:
        return ir.SymbolicDim(None)
    if isinstance(other, int):
        return ir.SymbolicDim(sympy.sympify(other / self._expr))
    return NotImplemented

def _mod__(self, other: int | ir.SymbolicDim) -> ir.SymbolicDim:
    """Compute modulo of this dimension by an integer or another ir.SymbolicDim."""
    if self._expr is None:
        return ir.SymbolicDim(None)
    if isinstance(other, int):
        return ir.SymbolicDim(sympy.sympify(self._expr % other))
    if isinstance(other, ir.SymbolicDim):
        if other._value is None:
            return ir.SymbolicDim(None)
        return ir.SymbolicDim(sympy.sympify(self._expr % other._expr))
    return NotImplemented

def _ceil__(self) -> ir.SymbolicDim:
    """Support math.ceil(dim). Returns a ir.SymbolicDim with ceiling expression."""
    if self._expr is None:
        return ir.SymbolicDim(None)
    return ir.SymbolicDim(sympy.ceiling(self._expr))

def _floor__(self) -> ir.SymbolicDim:
    """Support math.floor(dim). Returns a ir.SymbolicDim with floor expression."""
    if self._expr is None:
        return ir.SymbolicDim(None)
    return ir.SymbolicDim(sympy.floor(self._expr))

def _trunc__(self) -> ir.SymbolicDim:
    """Support math.trunc(dim). Returns a ir.SymbolicDim truncated toward zero."""
    if self._expr is None:
        return ir.SymbolicDim(None)
    return ir.SymbolicDim(sympy.sign(self._expr) * sympy.floor(sympy.Abs(self._expr)))

def _neg__(self) -> ir.SymbolicDim:
    """Negate this dimension."""
    if self._expr is None:
        return ir.SymbolicDim(None)
    return ir.SymbolicDim(sympy.sympify(-self._expr))


# Patch the methods onto ir.SymbolicDim
ir.SymbolicDim.__add__ = _add__
ir.SymbolicDim.__radd__ = _radd__
ir.SymbolicDim.__sub__ = _sub__
ir.SymbolicDim.__rsub__ = _rsub__
ir.SymbolicDim.__mul__ = _mul__
ir.SymbolicDim.__rmul__ = _rmul__
ir.SymbolicDim.__floordiv__ = _floordiv__
ir.SymbolicDim.__truediv__ = _truediv__
ir.SymbolicDim.__rtruediv__ = _rtruediv__
ir.SymbolicDim.__mod__ = _mod__
ir.SymbolicDim.__ceil__ = _ceil__
ir.SymbolicDim.__floor__ = _floor__
ir.SymbolicDim.__trunc__ = _trunc__
ir.SymbolicDim.__neg__ = _neg__
