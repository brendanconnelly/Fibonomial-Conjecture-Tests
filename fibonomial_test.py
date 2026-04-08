#!/usr/bin/env python3
"""
Computational verification that q-Fibonomial polynomials binom(m+n,n)_F are
symmetric (palindromic) and unimodal for all tested m, n >= 1.

Coded with the assistance of Claude (Anthropic).

Definition:
    F_0=0, F_1=1, F_k = F_{k-1} + F_{k-2}
    [r]_q = 1 + q + ... + q^{r-1}
    binom(m+n, n)_F = prod_{i=1}^n [F_{m+i}]_q / prod_{i=1}^n [F_i]_q

Degree: F_{m+n+2} - F_{m+2} - F_{n+2} + 1

Usage:
    python fibonomial_test.py [max_sum]   # default max_sum=20
"""

import sys
import time
from typing import List, Optional, Tuple
import numpy as np


def make_fibonacci(max_idx: int) -> List[int]:
    F = [0] * (max_idx + 1)
    if max_idx >= 1:
        F[1] = 1
    for i in range(2, max_idx + 1):
        F[i] = F[i - 1] + F[i - 2]
    return F


def poly_mul_all_ones(P: List[int], r: int) -> List[int]:
    """
    Multiply P by [r]_q = 1 + q + ... + q^{r-1} via sliding-window prefix sum.
    O(n + r). Falls back to Python bigints if numpy int64 overflows.
    """
    n = len(P)
    result_len = n + r - 1

    try:
        P64 = np.array(P, dtype=np.int64)
        prefix64 = np.zeros(n + 1, dtype=np.int64)
        np.cumsum(P64, out=prefix64[1:])
        j = np.arange(result_len, dtype=np.int64)
        hi = np.minimum(j + 1, n).astype(np.intp)
        lo = np.maximum(j - r + 1, 0).astype(np.intp)
        result64 = prefix64[hi] - prefix64[lo]
        if not np.any(result64 < 0):
            return result64.tolist()
    except (OverflowError, ValueError):
        pass

    prefix = [0] * (n + 1)
    for i in range(n):
        prefix[i + 1] = prefix[i] + P[i]
    result = [0] * result_len
    for j in range(result_len):
        result[j] = prefix[min(j + 1, n)] - prefix[max(j - r + 1, 0)]
    return result


def poly_mul(a: List[int], b: List[int]) -> List[int]:
    """General polynomial multiplication; numpy int64 with bigint fallback."""
    try:
        a64 = np.asarray(a, dtype=np.int64)
        b64 = np.asarray(b, dtype=np.int64)
    except (OverflowError, ValueError):
        la, lb = len(a), len(b)
        result = [0] * (la + lb - 1)
        for i in range(la):
            if a[i]:
                for j in range(lb):
                    result[i + j] += a[i] * b[j]
        return result
    out = np.convolve(a64, b64)
    if np.any(out < 0):
        la, lb = len(a), len(b)
        result = [0] * (la + lb - 1)
        for i in range(la):
            if a[i]:
                for j in range(lb):
                    result[i + j] += int(a[i]) * int(b[j])
        return result
    return out.tolist()


def poly_div_geometric(P: List[int], k: int) -> List[int]:
    """
    Exact division of P by [k]_q. O(len(P)).

    Uses [k]_q * (q-1) = q^k - 1:
      Step 1: R = P*(q-1)  =>  R[j] = P[j-1] - P[j]
      Step 2: Q*(q^k-1) = R  =>  Q[m] = Q[m-k] - R[m]
    """
    if k == 1:
        return list(P)
    n = len(P)
    deg_Q = n - k
    if deg_Q < 0:
        raise ValueError(f"Degree of divisor {k-1} exceeds degree of P {n-1}")
    R = [0] * (n + 1)
    R[0] = -P[0]
    for j in range(1, n):
        R[j] = P[j - 1] - P[j]
    R[n] = P[n - 1]
    Q = [0] * (deg_Q + 1)
    for m in range(deg_Q + 1):
        Q[m] = (Q[m - k] if m >= k else 0) - R[m]
    return Q


def degree_formula(m: int, n: int, F: List[int]) -> int:
    return F[m + n + 2] - F[m + 2] - F[n + 2] + 1


def fibonomial_poly(m: int, n: int, F: List[int]) -> List[int]:
    """
    Compute binom(m+n,n)_F via interleaved multiply-divide.
    At step i: multiply by [F_{m+i}]_q then divide by [F_i]_q.
    The intermediate result is always an exact polynomial with integer coefficients.
    """
    P = [1]
    for i in range(1, n + 1):
        P = poly_mul_all_ones(P, F[m + i])
        fi = F[i]
        if fi > 1:
            P = poly_div_geometric(P, fi)
    return P


def is_unimodal(coeffs: List[int]) -> bool:
    n = len(coeffs)
    if n <= 2:
        return True
    i = 0
    while i < n - 1 and coeffs[i] <= coeffs[i + 1]:
        i += 1
    while i < n - 1:
        if coeffs[i] < coeffs[i + 1]:
            return False
        i += 1
    return True


def is_palindrome(coeffs: List[int]) -> bool:
    n = len(coeffs)
    for i in range(n // 2):
        if coeffs[i] != coeffs[n - 1 - i]:
            return False
    return True


def _result_line(m, n, coeffs, elapsed, F):
    d = len(coeffs) - 1
    uni = is_unimodal(coeffs)
    pal = is_palindrome(coeffs)
    expected_deg = degree_formula(m, n, F)
    flags = []
    if not uni:
        flags.append("FAIL:unimodal")
    if not pal:
        flags.append("FAIL:palindrome")
    if d != expected_deg:
        flags.append(f"FAIL:degree(got {d}, expected {expected_deg})")
    status = "  ".join(flags) if flags else "ok"
    return (
        f"  C({m+n},{n})_F  m={m:2d} n={n:2d}  "
        f"deg={d:7d}  t={elapsed:6.3f}s  {status}"
    ), uni, pal


def run_grid(max_sum: int = 20, F: Optional[List[int]] = None, verbose: bool = True):
    """Test all (m,n) with m,n >= 1 and m+n <= max_sum."""
    if F is None:
        F = make_fibonacci(max_sum + 3)
    failures_uni, failures_pal, results = [], [], []
    total_t0 = time.perf_counter()
    if verbose:
        print(f"\n{'='*70}")
        print(f"  Grid test: m,n >= 1,  m+n <= {max_sum}")
        print(f"{'='*70}")
    for s in range(2, max_sum + 1):
        for n in range(1, s):
            m = s - n
            t0 = time.perf_counter()
            poly = fibonomial_poly(m, n, F)
            elapsed = time.perf_counter() - t0
            line, uni, pal = _result_line(m, n, poly, elapsed, F)
            results.append((m, n, len(poly) - 1, uni, pal, elapsed))
            if not uni:
                failures_uni.append((m, n))
            if not pal:
                failures_pal.append((m, n))
            if verbose:
                print(line)
    total_elapsed = time.perf_counter() - total_t0
    if verbose:
        print(f"\n  Pairs tested  : {len(results)}")
        print(f"  Total time    : {total_elapsed:.2f}s")
        print(f"  Unimodal fail : {len(failures_uni)} {failures_uni}")
        print(f"  Palindrome fail: {len(failures_pal)} {failures_pal}")
    return results, failures_uni, failures_pal


def run_selected(pairs: List[Tuple[int, int]], F: Optional[List[int]] = None,
                 verbose: bool = True):
    """Test specific (m,n) pairs."""
    if not pairs:
        return
    max_idx = max(m + n + 3 for m, n in pairs)
    if F is None or len(F) <= max_idx:
        F = make_fibonacci(max_idx)
    failures_uni, failures_pal = [], []
    if verbose:
        print(f"\n{'='*70}")
        print("  Selected large pairs")
        print(f"{'='*70}")
    for m, n in pairs:
        expected = degree_formula(m, n, F)
        print(f"  Computing C({m+n},{n})_F  m={m} n={n}  "
              f"expected degree={expected:,}  ...", flush=True)
        t0 = time.perf_counter()
        poly = fibonomial_poly(m, n, F)
        elapsed = time.perf_counter() - t0
        line, uni, pal = _result_line(m, n, poly, elapsed, F)
        if verbose:
            print(line)
        if not uni:
            failures_uni.append((m, n))
        if not pal:
            failures_pal.append((m, n))
    return failures_uni, failures_pal


def run_sanity_checks(F: List[int]) -> bool:
    print("\n--- Sanity checks ---")
    p = fibonomial_poly(1, 1, F)
    assert p == [1], f"binom(2,1)_F failed: {p}"
    print("  binom(2,1)_F = 1                           OK")
    p = fibonomial_poly(2, 1, F)
    assert p == [1, 1], f"binom(3,1)_F failed: {p}"
    print("  binom(3,1)_F = 1 + q                       OK")
    p = fibonomial_poly(1, 2, F)
    assert p == [1, 1], f"binom(3,2)_F failed: {p}"
    print("  binom(3,2)_F = 1+q                         OK")
    p = fibonomial_poly(2, 2, F)
    assert p == [1, 2, 2, 1], f"binom(4,2)_F failed: {p}"
    print("  binom(4,2)_F = 1+2q+2q^2+q^3              OK")
    p25 = poly_mul([1, 1, 1], [1, 1, 1, 1, 1])
    p = fibonomial_poly(3, 2, F)
    assert p == p25, f"binom(5,2)_F failed: {p}"
    print(f"  binom(5,2)_F = {p}   OK")
    for m in range(1, 6):
        for n in range(1, 6):
            poly = fibonomial_poly(m, n, F)
            d = len(poly) - 1
            assert d == degree_formula(m, n, F)
    print("  Degree formula matches for all m,n in 1..5  OK")
    for m in range(1, 6):
        for n in range(1, 6):
            poly = fibonomial_poly(m, n, F)
            assert is_unimodal(poly) and is_palindrome(poly)
    print("  All m,n in 1..5 are unimodal and palindromic OK")
    print("--- All sanity checks passed ---\n")
    return True


if __name__ == "__main__":
    max_sum = 20
    if len(sys.argv) > 1:
        try:
            max_sum = int(sys.argv[1])
        except ValueError:
            print(f"Usage: {sys.argv[0]} [max_sum]")
            sys.exit(1)

    F = make_fibonacci(max(max_sum + 3, 36))
    run_sanity_checks(F)

    results, fail_uni, fail_pal = run_grid(max_sum=max_sum, F=F, verbose=True)
    degs = [r[2] for r in results]
    print(f"\n  Degree range: min={min(degs)}, max={max(degs)}")
    print(f"  All unimodal: {len(fail_uni) == 0}")
    print(f"  All palindromic: {len(fail_pal) == 0}")

    large_pairs = [(11,11),(12,12),(13,13),(14,14),(15,15),(16,16)]
    print(f"\n{'='*70}")
    print("  Estimated degrees for large selected pairs:")
    for m, n in large_pairs:
        d = degree_formula(m, n, F)
        note = "  (may take several seconds)" if d > 5_000_000 else ""
        print(f"    C({m+n},{n})_F  m={m} n={n}  expected degree={d:,}{note}")
    print()
    answer = input("  Run large pairs? [y/N]: ").strip().lower()
    if answer == 'y':
        run_selected(large_pairs, F=F, verbose=True)
    else:
        print("  Skipping large pairs.")
    print("\nDone.")
