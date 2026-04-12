# q-Fibonomial Unimodality Tests

Computational verification that q-Fibonomial polynomials are palindromic and unimodal for all tested parameters.

The q-Fibonomial coefficients are defined via Fibonacci numbers as a q-analogue of the binomial coefficient, replacing factorial ratios with products of q-integers built from Fibonacci values.

## Files

| File | Description |
|---|---|
| `fibonomial_test.py` | Verifies unimodality and palindromicity of q-Fibonomial polynomials |
| `conjecture_test.py` | Tests the broader q-analogue conjecture |

## Usage

```bash
python fibonomial_test.py [max_sum]   # default max_sum=20
python conjecture_test.py
```

## Notes

Results support the conjecture that these polynomials are unimodal for all $m, n \geq 1$. No counterexamples were found up to the tested parameter range.