# Parker op-inference fixes

Branch `fix/elementwise-matmul`, commit `1a1c17e`, pushed to origin.

Fixed symbolic Max/Min crash, integer Div truncation, MatMul/Gemm K validation, Dropout 7-9 mask dtype, and Pad-1 `paddings`. Added regressions. Validation: ruff passed; targeted 117 passed; full suite 3221 passed.
