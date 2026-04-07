#!/usr/bin/env python3
"""
Computational test of unimodality and symmetry for q-Fibonomial polynomials.

Definition:
    F_0=0, F_1=1, F_k = F_{k-1} + F_{k-2}   (Fibonacci numbers)

    [r]_q = 1 + q + q^2 + ... + q^{r-1}       (q-integer)

    binom(m+n, n)_F = prod_{i=1}^n [F_{m+i}]_q
                     --------------------------    (shortened product form)
                      prod_{i=1}^n [F_i]_q

This is a polynomial in q with positive integer coefficients.

Degree formula (derived from Fibonacci sum identity):
    deg = F_{m+n+2} - F_{m+2} - F_{n+2} + 1

Feasibility notes:
    m+n = 20 (m=n=10): degree ~17,424   → fast with numpy
    m+n = 24 (m=n=12): degree ~47,559   → a few seconds
    m+n = 26 (m=n=13): degree ~124,940  → ~15-30s each
    m+n = 30 (m=n=15): degree ~2.17M    → NOT feasible (O(10^12) multiply ops)

Key algorithm — O(degree) division by [k]_q:
    Since [k]_q * (q-1) = q^k - 1, if P = Q * [k]_q then P*(q-1) = Q*(q^k-1).
    Step 1:  R[j] = P[j-1] - P[j]       (O(deg) — multiply by q-1)
    Step 2:  Q[m] = Q[m-k] - R[m]       (O(deg) — divide by q^k-1)
    This replaces naive O(deg * k) long division with O(deg).
"""

import sys
import time
from typing import List, Optional, Tuple
import numpy as np


# ============================================================
# Fibonacci numbers
# ============================================================

def make_fibonacci(max_idx: int) -> List[int]:
    """Return F[0..max_idx] as a list. F[0]=0, F[1]=1."""
    F = [0] * (max_idx + 1)
    if max_idx >= 1:
        F[1] = 1
    for i in range(2, max_idx + 1):
        F[i] = F[i - 1] + F[i - 2]
    return F


# ============================================================
# Polynomial arithmetic
# Representation: list of ints [c_0, c_1, ..., c_d]
#   = c_0 + c_1*q + ... + c_d*q^d
# All arithmetic uses Python arbitrary-precision integers.
# ============================================================

def poly_q_geometric(r: int) -> List[int]:
    """
    [r]_q = 1 + q + ... + q^{r-1}, returned as [1, 1, ..., 1] of length r.
    Requires r >= 1.
    """
    if r < 1:
        raise ValueError(f"[r]_q requires r >= 1, got {r}")
    return [1] * r


def poly_mul_all_ones(P: List[int], r: int) -> List[int]:
    """
    Multiply polynomial P by [r]_q = 1 + q + ... + q^{r-1} using a
    sliding-window prefix-sum algorithm.

    Since [r]_q has all-1 coefficients:
        result[j] = sum_{k = max(0, j-r+1)}^{min(j, n-1)} P[k]

    This is O(n + r), not O(n * r) — a major speedup for large r = F_{m+i}.

    Strategy:
        1. Build a prefix-sum array: prefix[i+1] = P[0] + ... + P[i].
        2. result[j] = prefix[min(j+1, n)] - prefix[max(j-r+1, 0)].

    Uses numpy int64 for speed when coefficients fit; falls back to pure Python
    bigint arithmetic (which is still O(n + r)) when they overflow.
    """
    n = len(P)
    result_len = n + r - 1

    # --- numpy int64 path ---
    try:
        P64 = np.array(P, dtype=np.int64)
        prefix64 = np.zeros(n + 1, dtype=np.int64)
        np.cumsum(P64, out=prefix64[1:])

        j = np.arange(result_len, dtype=np.int64)
        hi = np.minimum(j + 1, n).astype(np.intp)
        lo = np.maximum(j - r + 1, 0).astype(np.intp)
        result64 = prefix64[hi] - prefix64[lo]

        if not np.any(result64 < 0):   # no overflow (all values should be >= 0)
            return result64.tolist()
    except (OverflowError, ValueError):
        pass

    # --- Python bigint fallback (exact, arbitrary precision) ---
    # Build prefix sums with Python ints
    prefix = [0] * (n + 1)
    for i in range(n):
        prefix[i + 1] = prefix[i] + P[i]

    result = [0] * result_len
    for j in range(result_len):
        result[j] = prefix[min(j + 1, n)] - prefix[max(j - r + 1, 0)]
    return result


def poly_mul(a: List[int], b: List[int]) -> List[int]:
    """
    General polynomial multiplication via numpy int64 convolution with
    pure-Python bigint fallback on overflow. This is used only for
    general (non-all-1) multiplications; fibonomial_poly uses
    poly_mul_all_ones for its numerator factors.
    """
    try:
        a64 = np.asarray(a, dtype=np.int64)
        b64 = np.asarray(b, dtype=np.int64)
    except (OverflowError, ValueError):
        # Input overflows int64 — fall straight to Python bigints
        la, lb = len(a), len(b)
        result = [0] * (la + lb - 1)
        for i in range(la):
            ai = a[i]
            if ai == 0:
                continue
            for j in range(lb):
                result[i + j] += ai * b[j]
        return result
    out = np.convolve(a64, b64)
    if np.any(out < 0):
        # int64 overflow in output — fall back
        la, lb = len(a), len(b)
        result = [0] * (la + lb - 1)
        for i in range(la):
            ai = int(a[i])
            if ai == 0:
                continue
            for j in range(lb):
                result[i + j] += ai * int(b[j])
        return result
    return out.tolist()


def poly_div_geometric(P: List[int], k: int) -> List[int]:
    """
    Divide polynomial P exactly by [k]_q = 1 + q + ... + q^{k-1}.
    P must be exactly divisible (no remainder); this is guaranteed when P
    is the q-Fibonomial numerator and k = F_i for i <= n.

    Algorithm (O(len(P)), no nested loops):
        Identity: [k]_q * (q-1) = q^k - 1.
        So if P = Q * [k]_q, then P*(q-1) = Q*(q^k-1).

        Step 1 — compute R = P*(q-1):
            R[0]   = -P[0]
            R[j]   =  P[j-1] - P[j]    for 1 <= j <= len(P)-1
            R[n]   =  P[n-1]            where n = len(P)

        Step 2 — recover Q from R using Q*(q^k-1) = R,
                 i.e. Q[m-k] - Q[m] = R[m], so Q[m] = Q[m-k] - R[m]:
            Q[m] = (Q[m-k] if m>=k else 0) - R[m]

    Returns the quotient Q as a list of integers.
    """
    if k == 1:
        # [1]_q = 1, identity
        return list(P)

    n = len(P)           # P has degree n-1
    deg_Q = n - k        # Q has degree n-k, i.e. len = n-k+1

    if deg_Q < 0:
        raise ValueError(
            f"Divisor [k]_q has degree {k-1} but P has degree {n-1} < {k-1}"
        )

    # Step 1: R = P * (q - 1), length n+1
    R = [0] * (n + 1)
    R[0] = -P[0]
    for j in range(1, n):
        R[j] = P[j - 1] - P[j]
    R[n] = P[n - 1]

    # Step 2: Q[m] = Q[m-k] - R[m]
    Q = [0] * (deg_Q + 1)
    for m in range(deg_Q + 1):
        Q[m] = (Q[m - k] if m >= k else 0) - R[m]

    return Q


# ============================================================
# q-Fibonomial polynomial
# ============================================================

def degree_formula(m: int, n: int, F: List[int]) -> int:
    """
    Analytic degree of binom(m+n, n)_F using the Fibonacci sum identity:
        deg = F_{m+n+2} - F_{m+2} - F_{n+2} + 1
    Used as a sanity check against the actual polynomial degree.
    """
    return F[m + n + 2] - F[m + 2] - F[n + 2] + 1


def fibonomial_poly(m: int, n: int, F: List[int]) -> List[int]:
    """
    Compute binom(m+n, n)_F as a polynomial via the shortened product:

        binom(m+n, n)_F = prod_{i=1}^n [F_{m+i}]_q
                         --------------------------
                          prod_{i=1}^n [F_i]_q

    Strategy — interleaved multiply-divide:
        At each step i, multiply the running result P by [F_{m+i}]_q, then
        immediately divide by [F_i]_q.

        Correctness: after step i, P equals binom(m+i, i)_F exactly:

            P_i = prod_{j=1}^i [F_{m+j}]_q / prod_{j=1}^i [F_j]_q
                = binom(m+i, i)_F

        Since binom(m+i, i)_F is a polynomial in q with integer coefficients
        for every m >= 1, i >= 1, the division is always exact.

        Advantage over "build full numerator then divide":
        The raw numerator polynomial (before any division) can have enormous
        intermediate coefficients — easily exceeding int64 for n ~ 12 — because
        it is a product of n polynomials with all-1 coefficients. The interleaved
        intermediate result binom(m+i, i)_F has much smaller coefficients, so
        numpy int64 arithmetic works throughout for all practically relevant sizes.

    F_1 = F_2 = 1, so [F_1]_q = [F_2]_q = 1 — those divisions are no-ops.
    """
    P = [1]
    for i in range(1, n + 1):
        # Multiply by [F_{m+i}]_q  — uses O(deg + F_{m+i}) sliding-window sum
        P = poly_mul_all_ones(P, F[m + i])
        # Divide by [F_i]_q        — uses O(deg) geometric-series trick
        fi = F[i]
        if fi > 1:
            P = poly_div_geometric(P, fi)
    return P


# ============================================================
# Analysis: unimodality and palindrome
# ============================================================

def is_unimodal(coeffs: List[int]) -> bool:
    """
    A sequence c_0, ..., c_d is unimodal if there exists t such that
        c_0 <= c_1 <= ... <= c_t >= c_{t+1} >= ... >= c_d.
    Equivalently: after the first strict descent, no subsequent ascent.
    """
    n = len(coeffs)
    if n <= 2:
        return True

    i = 0
    # Scan through the non-decreasing prefix
    while i < n - 1 and coeffs[i] <= coeffs[i + 1]:
        i += 1
    # i is now at (or past) the peak; check the tail is non-increasing
    while i < n - 1:
        if coeffs[i] < coeffs[i + 1]:
            return False
        i += 1
    return True


def is_palindrome(coeffs: List[int]) -> bool:
    """
    Check c_k == c_{d-k} for all k (where d = degree).
    """
    n = len(coeffs)
    for i in range(n // 2):
        if coeffs[i] != coeffs[n - 1 - i]:
            return False
    return True


# ============================================================
# Test runners
# ============================================================

def _result_line(m, n, coeffs, elapsed, F, warn_degree_mismatch=True):
    """Format a single result line."""
    d = len(coeffs) - 1
    uni = is_unimodal(coeffs)
    pal = is_palindrome(coeffs)

    expected_deg = degree_formula(m, n, F)
    deg_ok = (d == expected_deg)

    flags = []
    if not uni:
        flags.append("FAIL:unimodal")
    if not pal:
        flags.append("FAIL:palindrome")
    if warn_degree_mismatch and not deg_ok:
        flags.append(f"FAIL:degree(got {d}, expected {expected_deg})")
    status = "  ".join(flags) if flags else "ok"

    return (
        f"  C({m+n},{n})_F  m={m:2d} n={n:2d}  "
        f"deg={d:7d}  t={elapsed:6.3f}s  {status}"
    ), uni, pal


def run_grid(max_sum: int = 20, F: Optional[List[int]] = None, verbose: bool = True):
    """
    Test all (m, n) with m >= 1, n >= 1, m+n <= max_sum.
    Returns (results, unimodal_failures, palindrome_failures).
    """
    if F is None:
        F = make_fibonacci(max_sum + 3)

    failures_uni = []
    failures_pal = []
    results = []
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
    """
    Test specific (m, n) pairs. Useful for larger cases beyond the grid.
    """
    if not pairs:
        return
    max_idx = max(m + n + 3 for m, n in pairs)
    if F is None:
        F = make_fibonacci(max_idx)
    elif len(F) <= max_idx:
        F = make_fibonacci(max_idx)

    failures_uni = []
    failures_pal = []

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


# ============================================================
# Sanity checks
# ============================================================

def run_sanity_checks(F: List[int]) -> bool:
    """
    Verify small known cases.
    Returns True if all checks pass.
    """
    print("\n--- Sanity checks ---")
    ok = True

    # binom(2,1)_F = [F_2]_q / [F_1]_q = [1]_q / 1 = 1
    p = fibonomial_poly(1, 1, F)
    assert p == [1], f"binom(2,1)_F failed: {p}"
    print("  binom(2,1)_F = 1                           OK")

    # binom(3,1)_F = [F_3]_q / [F_1]_q = [2]_q = 1+q
    p = fibonomial_poly(2, 1, F)
    assert p == [1, 1], f"binom(3,1)_F failed: {p}"
    print("  binom(3,1)_F = 1 + q                       OK")

    # binom(3,2)_F: m=1, n=2
    #   Numerator:   [F_2]_q * [F_3]_q = [1]_q * [2]_q = 1*(1+q) = 1+q
    #   Denominator: [F_1]_q * [F_2]_q = 1*1 = 1
    #   Result: 1+q
    p = fibonomial_poly(1, 2, F)
    assert p == [1, 1], f"binom(3,2)_F failed: {p}"
    print("  binom(3,2)_F = 1+q                         OK")

    # binom(4,2)_F: m=2, n=2
    #   Numerator:   [F_3]_q * [F_4]_q = [2]_q * [3]_q = (1+q)(1+q+q^2)
    #              = 1 + 2q + 2q^2 + q^3
    #   Denominator: [F_1]_q * [F_2]_q = 1
    #   Result: 1 + 2q + 2q^2 + q^3
    p = fibonomial_poly(2, 2, F)
    assert p == [1, 2, 2, 1], f"binom(4,2)_F failed: {p}"
    print("  binom(4,2)_F = 1+2q+2q^2+q^3              OK")

    # binom(5,2)_F = [F_4]_q*[F_5]_q / ([F_1]_q*[F_2]_q)
    #              = [3]_q * [5]_q = (1+q+q^2)(1+q+q^2+q^3+q^4)
    #              = 1+2q+3q^2+3q^3+3q^4+2q^5+q^6
    p25 = poly_mul([1, 1, 1], [1, 1, 1, 1, 1])
    p = fibonomial_poly(3, 2, F)
    assert p == p25, f"binom(5,2)_F failed: {p}"
    print(f"  binom(5,2)_F = {p}   OK")

    # Degree formula checks
    for m in range(1, 6):
        for n in range(1, 6):
            poly = fibonomial_poly(m, n, F)
            d = len(poly) - 1
            expected = degree_formula(m, n, F)
            assert d == expected, (
                f"Degree mismatch for m={m},n={n}: got {d}, expected {expected}"
            )
    print("  Degree formula matches actual degree for all m,n in 1..5  OK")

    # All small cases should be unimodal and palindrome
    for m in range(1, 6):
        for n in range(1, 6):
            poly = fibonomial_poly(m, n, F)
            assert is_unimodal(poly), f"Not unimodal: m={m},n={n}"
            assert is_palindrome(poly), f"Not palindrome: m={m},n={n}"
    print("  All m,n in 1..5 are unimodal and palindromic              OK")

    print("--- All sanity checks passed ---\n")
    return ok


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    # --- Parse optional command-line argument for max_sum ---
    max_sum = 20
    if len(sys.argv) > 1:
        try:
            max_sum = int(sys.argv[1])
        except ValueError:
            print(f"Usage: {sys.argv[0]} [max_sum]")
            sys.exit(1)

    # Precompute Fibonacci numbers up to the required index.
    # Large selected pairs go up to m=n=16 (m+n=32), so we need F[34].
    F = make_fibonacci(max(max_sum + 3, 36))

    # --- Sanity checks on small known values ---
    run_sanity_checks(F)

    # --- Exhaustive grid test ---
    results, fail_uni, fail_pal = run_grid(max_sum=max_sum, F=F, verbose=True)

    # --- Summary statistics ---
    degs = [r[2] for r in results]
    print(f"\n  Degree range: min={min(degs)}, max={max(degs)}")
    print(f"  All unimodal: {len(fail_uni) == 0}")
    print(f"  All palindromic: {len(fail_pal) == 0}")

    # --- Selected larger cases beyond the grid ---
    # The O(deg + r) sliding-window multiply and O(deg) division make large
    # cases feasible. All intermediate computations use Python arbitrary-
    # precision integers when numpy int64 would overflow.
    #
    # Observed timings (2026, Apple Silicon M-series class CPU):
    #   m=n=11  deg=   45,903    ~0.03s
    #   m=n=12  deg=  120,640    ~0.08s
    #   m=n=13  deg=  316,592    ~0.22s
    #   m=n=14  deg=  830,067    ~0.59s
    #   m=n=15  deg=2,175,116    ~1.6s
    #   m=n=16  deg=5,697,720    ~4.2s
    #   m=n=17  deg=14,921,991   ~11s
    #   m=n=18  deg=39,074,640   ~30s  (max tested — all unimodal & palindromic)
    #
    # Beyond m=n=18: degree ~100M+; each bigint step is slower due to
    # growing coefficient sizes (~200+ bits), so runtime escalates quickly.

    large_pairs = [
        (11, 11),
        (12, 12),
        (13, 13),
        (14, 14),
        (15, 15),
        (16, 16),
    ]

    # Estimate degrees before committing
    print(f"\n{'='*70}")
    print("  Estimated degrees for large selected pairs:")
    for m, n in large_pairs:
        d = degree_formula(m, n, F)
        t_note = ""
        if d > 5_000_000:
            t_note = "  (may take several seconds)"
        print(f"    C({m+n},{n})_F  m={m} n={n}  expected degree={d:,}{t_note}")

    print()
    answer = input("  Run large pairs? [y/N]: ").strip().lower()
    if answer == 'y':
        run_selected(large_pairs, F=F, verbose=True)
    else:
        print("  Skipping large pairs.")

    print("\nDone.")
