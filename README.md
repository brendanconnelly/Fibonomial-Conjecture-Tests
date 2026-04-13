# Fibonomial Conjecture Tests

Computational verification of unimodality and symmetry conjectures for q-Fibonomial polynomials. Coded with the assistance of Claude (Anthropic).

## Scripts

### `fibonomial_test.py`
Tests that the q-Fibonomial polynomial $\binom{m+n}{n}_F$ is **symmetric and unimodal** for all $m, n \geq 1$.

- Exhaustive grid: all pairs with $m + n \leq 20$
- Large selected pairs: $(m,n) = (k,k)$ up to $k = 16$ (degree 5,697,720)

```
python3 fibonomial_test.py [max_sum]   # default max_sum=20
```

### `conjecture_test.py`
Tests the following conjecture about products of q-integers and q-combs:

> Let $k, n \geq 2$, with $a_i$ positive integers and $n \nmid a_i$ for $i < k$. If
> $$a_k \leq 1 + \sum_{i=1}^{k-1} \left\lfloor \frac{a_i}{n} \right\rfloor,$$
> then $P(a_1, \dots, a_k; n) = [a_1]_q \cdots [a_{k-1}]_q [a_k]_{q^n}$ is unimodal.
> Moreover, for $k \leq 4$ or $n \leq 3$, this condition is also necessary.

Verified for $k, n \leq 6$ and $\max a_i \leq 15$ (~7.6 million polynomials). No counterexamples found.

```
python3 conjecture_test.py              # default: k,n<=6, max_a<=15
python3 conjecture_test.py --full       # full:    k,n<=10, max_a<=20
python3 conjecture_test.py 8 8 18       # custom:  max_k max_n max_a
```

## Requirements

Python 3 with numpy.

```
pip install numpy
```
