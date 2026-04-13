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
    python fibonomial_test.py [max_sum]   # default max_sum=26
"""

import os
import sys
import time
import multiprocessing as mp
from typing import List, Optional, Tuple
import numpy as np

# Set True to enable exact-division verification in poly_div_geometric (slow).
DEBUG = False


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
    O(n + r). Falls back to Python bigints on overflow.

    Overflow detection: sum of output must equal sum(P) * r (since [r]_q at q=1
    equals r). This catches silent positive wrapping that a < 0 check misses.
    """
    n = len(P)
    result_len = n + r - 1
    expected_sum = sum(P) * r   # Python bigint — exact, no overflow possible

    try:
        P64 = np.array(P, dtype=np.int64)
        prefix64 = np.zeros(n + 1, dtype=np.int64)
        np.cumsum(P64, out=prefix64[1:])
        j = np.arange(result_len, dtype=np.int64)
        hi = np.minimum(j + 1, n).astype(np.intp)
        lo = np.maximum(j - r + 1, 0).astype(np.intp)
        result64 = prefix64[hi] - prefix64[lo]
        result_list = result64.tolist()
        if sum(result_list) == expected_sum:
            return result_list
    except (OverflowError, ValueError):
        pass

    # Python bigint fallback (exact, arbitrary precision)
    prefix = [0] * (n + 1)
    for i in range(n):
        prefix[i + 1] = prefix[i] + P[i]
    result = [0] * result_len
    for j in range(result_len):
        result[j] = prefix[min(j + 1, n)] - prefix[max(j - r + 1, 0)]
    return result


def poly_mul(a: List[int], b: List[int]) -> List[int]:
    """
    General polynomial multiplication; numpy int64 with bigint fallback.
    Overflow detection uses sum(result) == sum(a) * sum(b).
    """
    expected_sum = sum(a) * sum(b)
    try:
        a64 = np.asarray(a, dtype=np.int64)
        b64 = np.asarray(b, dtype=np.int64)
        out = np.convolve(a64, b64)
        result_list = out.tolist()
        if sum(result_list) == expected_sum:
            return result_list
    except (OverflowError, ValueError):
        pass

    la, lb = len(a), len(b)
    result = [0] * (la + lb - 1)
    for i in range(la):
        if a[i]:
            for j in range(lb):
                result[i + j] += a[i] * b[j]
    return result


def poly_div_geometric(P: List[int], k: int) -> List[int]:
    """
    Exact division of P by [k]_q. O(len(P)).

    Uses [k]_q * (q-1) = q^k - 1:
      Step 1: R = P*(q-1)  =>  R[j] = P[j-1] - P[j]
      Step 2: Q*(q^k-1) = R  =>  Q[m] = Q[m-k] - R[m]

    When DEBUG=True, verifies Q * [k]_q == P afterward.
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

    if DEBUG:
        check = poly_mul_all_ones(Q, k)
        assert check == list(P), (
            f"poly_div_geometric: division by [k]_q not exact (k={k})\n"
            f"  P={P}\n  Q={Q}\n  Q*[k]_q={check}"
        )

    return Q


def degree_formula(m: int, n: int, F: List[int]) -> int:
    return F[m + n + 2] - F[m + 2] - F[n + 2] + 1


def fibonomial_poly(m: int, n: int, F: List[int]) -> List[int]:
    """
    Compute binom(m+n,n)_F via interleaved multiply-divide.
    At step i: multiply by [F_{m+i}]_q then divide by [F_i]_q.
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


def run_grid(max_sum: int = 26, F: Optional[List[int]] = None, verbose: bool = True):
    """
    Test all unique (m,n) with m >= n >= 1 and m+n <= max_sum.
    Skips the mirror (n,m) since binom(m+n,n)_F = binom(m+n,m)_F.
    """
    if F is None:
        F = make_fibonacci(max_sum + 3)
    failures_uni, failures_pal, results = [], [], []
    total_t0 = time.perf_counter()
    if verbose:
        print(f"\n{'='*70}")
        print(f"  Grid test: m >= n >= 1, m+n <= {max_sum}  (unique pairs by symmetry)")
        print(f"{'='*70}")
    for s in range(2, max_sum + 1):
        for n in range(1, s // 2 + 1):
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
        print(f"\n  Unique pairs tested : {len(results)}")
        print(f"  Total time          : {total_elapsed:.2f}s")
        print(f"  Unimodal fail       : {len(failures_uni)} {failures_uni}")
        print(f"  Palindrome fail     : {len(failures_pal)} {failures_pal}")
    return results, failures_uni, failures_pal


# ── parallel worker for run_selected ────────────────────────────────────────

def _fib_pair_worker(args: tuple) -> dict:
    """Compute and check one (m,n) pair; runs in a subprocess."""
    m, n, F = args
    t0 = time.perf_counter()
    poly = fibonomial_poly(m, n, F)
    elapsed = time.perf_counter() - t0
    return {
        "m": m, "n": n,
        "deg": len(poly) - 1,
        "uni": is_unimodal(poly),
        "pal": is_palindrome(poly),
        "elapsed": elapsed,
    }


def run_selected(pairs: List[Tuple[int, int]], F: Optional[List[int]] = None,
                 verbose: bool = True):
    """
    Test specific (m,n) pairs in parallel across all CPU cores.
    Pairs with the same m+n parity cover different Fibonacci divisibility patterns,
    so mixing equal, off-by-one, and off-by-two pairs gives broader coverage.
    """
    if not pairs:
        return [], []
    max_idx = max(m + n + 3 for m, n in pairs)
    if F is None or len(F) <= max_idx:
        F = make_fibonacci(max_idx)

    failures_uni, failures_pal = [], []
    if verbose:
        print(f"\n{'='*70}")
        print(f"  Selected large pairs  ({os.cpu_count()} workers, running in parallel)")
        print(f"{'='*70}")
        for m, n in pairs:
            d = degree_formula(m, n, F)
            print(f"    C({m+n},{n})_F  m={m} n={n}  degree={d:,}")
        print()

    work = [(m, n, F) for m, n in pairs]
    with mp.Pool(os.cpu_count()) as pool:
        results = pool.map(_fib_pair_worker, work)

    for r in results:
        m, n = r["m"], r["n"]
        flag = []
        if not r["uni"]:
            flag.append("FAIL:unimodal")
            failures_uni.append((m, n))
        if not r["pal"]:
            flag.append("FAIL:palindrome")
            failures_pal.append((m, n))
        status = "  ".join(flag) if flag else "ok"
        if verbose:
            print(f"  C({m+n},{n})_F  m={m:2d} n={n:2d}  "
                  f"deg={r['deg']:>9,}  t={r['elapsed']:6.3f}s  {status}")

    if verbose:
        print(f"\n  Unimodal fail  : {len(failures_uni)} {failures_uni}")
        print(f"  Palindrome fail: {len(failures_pal)} {failures_pal}")

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
            assert len(poly) - 1 == degree_formula(m, n, F)
    print("  Degree formula matches for all m,n in 1..5  OK")
    for m in range(1, 6):
        for n in range(1, 6):
            poly = fibonomial_poly(m, n, F)
            assert is_unimodal(poly) and is_palindrome(poly)
    print("  All m,n in 1..5 are unimodal and palindromic OK")
    for m in range(1, 5):
        for n in range(1, 5):
            assert fibonomial_poly(m, n, F) == fibonomial_poly(n, m, F)
    print("  binom(m+n,n)_F == binom(m+n,m)_F for m,n in 1..4  OK")
    print("--- All sanity checks passed ---\n")
    return True


if __name__ == "__main__":
    max_sum = 26
    if len(sys.argv) > 1:
        try:
            max_sum = int(sys.argv[1])
        except ValueError:
            print(f"Usage: {sys.argv[0]} [max_sum]")
            sys.exit(1)

    # Need F indices up to max(m+n+2) across grid and large pairs.
    # Large pairs go up to m=17, n=16 → F[35]; use 40 as safe margin.
    F = make_fibonacci(max(max_sum + 3, 40))
    run_sanity_checks(F)

    results, fail_uni, fail_pal = run_grid(max_sum=max_sum, F=F, verbose=True)
    degs = [r[2] for r in results]
    print(f"\n  Degree range: min={min(degs)}, max={max(degs)}")
    print(f"  All unimodal: {len(fail_uni) == 0}")
    print(f"  All palindromic: {len(fail_pal) == 0}")

    # Large pairs: for each diagonal k=11..16, test (k,k) plus off-by-one (k+1,k)
    # and off-by-two (k+2,k). Off-by-one covers odd m+n sums (different Fibonacci
    # divisibility pattern); off-by-two covers even m+n at asymmetric m,n.
    # (18,16) omitted — degree ~15M, ~15s each.
    large_pairs = [
        (11, 11), (12, 11), (13, 11),
        (12, 12), (13, 12), (14, 12),
        (13, 13), (14, 13), (15, 13),
        (14, 14), (15, 14), (16, 14),
        (15, 15), (16, 15), (17, 15),
        (16, 16), (17, 16),
    ]

    run_selected(large_pairs, F=F, verbose=True)
    print("\nDone.")
