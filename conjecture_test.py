#!/usr/bin/env python3
"""
Computational verification of the following unimodality conjecture:

    Conjecture. Let k, n >= 2. For each 1 <= i <= k, let a_i be a positive
    integer with a_i not divisible by n for i < k. If
        a_k <= 1 + sum_{i=1}^{k-1} floor(a_i / n),
    then P(a_1,...,a_k; n) = [a_1]_q ... [a_{k-1}]_q [a_k]_{q^n} is unimodal.
    Moreover, for k <= 4 or n <= 3, this condition is also necessary.

    [r]_q     = 1 + q + ... + q^{r-1}
    [a]_{q^n} = 1 + q^n + ... + q^{(a-1)n}

    Known necessity counterexample (k=5, n=4):
        P(3,3,3,3,2; 4) = [3]_q^4 [2]_{q^4}  is unimodal,
        but 2 > 1 + 4*floor(3/4) = 1.

Coded with the assistance of Claude (Anthropic).

Optimization notes:
  - Prefix (a_1,...,a_{k-1}) is enumerated as sorted multisets via
    combinations_with_replacement, since the polynomial and condition are both
    symmetric in those variables. This reduces enumeration by up to (k-1)! per
    multiset size compared to itertools.product.
  - The outer (k, n) loop is parallelized across CPU cores via multiprocessing.

Usage:
    python3 conjecture_test.py              # default: k,n<=6, max_a<=20  (~8s on M5 Pro)
    python3 conjecture_test.py --full       # extended: k,n<=8, max_a<=20 (~20-30 min)
    python3 conjecture_test.py 8 8 18       # custom: max_k max_n max_a

    Note: k >= 9 with small n is not feasible for exhaustive search even with
    deduplication (e.g. n=2, k=9 still has C(18,8) ~ 43k multisets × many a_k).
"""

import sys
import os
import time
import itertools
import multiprocessing as mp
from typing import List, Tuple

from fibonomial_test import poly_mul_all_ones, is_unimodal


# ============================================================
# Polynomial arithmetic
# ============================================================

def poly_mul_comb(P: List[int], a: int, n: int) -> List[int]:
    """
    Multiply P by [a]_{q^n} = 1 + q^n + ... + q^{(a-1)n}.

    Decomposes by residue mod n: for each r in {0,...,n-1}, extract P[r::n],
    multiply by all-ones of length a, write back at stride n. O(len(P) + a*n).
    """
    if a == 1:
        return list(P)
    result = [0] * (len(P) + (a - 1) * n)
    for r in range(n):
        sub = P[r::n]
        if sub:
            result[r::n] = poly_mul_all_ones(sub, a)
    return result


def conjecture_poly(a_list: List[int], n: int) -> List[int]:
    """P(a_1,...,a_k; n) = [a_1]_q ... [a_{k-1}]_q * [a_k]_{q^n}."""
    P = [1]
    for a in a_list[:-1]:
        P = poly_mul_all_ones(P, a)
    return poly_mul_comb(P, a_list[-1], n)


def condition_holds(a_list: List[int], n: int) -> bool:
    return a_list[-1] <= 1 + sum(a // n for a in a_list[:-1])


# ============================================================
# Per-(k,n) worker — runs in a subprocess via multiprocessing
# ============================================================

def _worker_sufficiency(args: tuple) -> dict:
    """
    Test all multiset prefixes of length k-1 drawn from valid_pre,
    for a_k in [1, min(bound, max_a)] (condition satisfied).
    Returns dict with tested count, failures, and elapsed time.
    """
    k, n, max_a = args
    valid_pre = [a for a in range(1, max_a + 1) if a % n != 0]
    failures = []
    tested = 0
    t0 = time.perf_counter()

    # combinations_with_replacement gives sorted multisets — exploits symmetry
    # of the polynomial and condition in (a_1,...,a_{k-1}).
    for prefix in itertools.combinations_with_replacement(valid_pre, k - 1):
        bound = 1 + sum(a // n for a in prefix)
        for a_k in range(1, min(bound, max_a) + 1):
            a_list = list(prefix) + [a_k]
            tested += 1
            if not is_unimodal(conjecture_poly(a_list, n)):
                failures.append((a_list, n))

    return {"k": k, "n": n, "tested": tested, "failures": failures,
            "elapsed": time.perf_counter() - t0}


def _worker_necessity(args: tuple) -> dict:
    """
    Test all multiset prefixes of length k-1 drawn from valid_pre,
    for a_k in [bound+1, max_a] (condition violated).
    Only records unimodal cases — a non-empty result for k<=4 or n<=3
    would disprove the conjecture.
    """
    k, n, max_a = args
    valid_pre = [a for a in range(1, max_a + 1) if a % n != 0]
    conj_failures = []   # k<=4 or n<=3 and unimodal despite condition failing
    other_unimodal = []  # k>=5 and n>=4: expected; conjecture says nothing here
    tested = 0
    t0 = time.perf_counter()

    for prefix in itertools.combinations_with_replacement(valid_pre, k - 1):
        bound = 1 + sum(a // n for a in prefix)
        for a_k in range(bound + 1, max_a + 1):
            a_list = list(prefix) + [a_k]
            tested += 1
            if is_unimodal(conjecture_poly(a_list, n)):
                if k <= 4 or n <= 3:
                    conj_failures.append((a_list, n))
                else:
                    other_unimodal.append((a_list, n))

    return {"k": k, "n": n, "tested": tested,
            "conj_failures": conj_failures, "other_unimodal": other_unimodal,
            "elapsed": time.perf_counter() - t0}


# ============================================================
# Sanity checks
# ============================================================

def run_sanity_checks() -> bool:
    print("\n--- Sanity checks ---")

    p = poly_mul_comb([1], 2, 3)
    assert p == [1, 0, 0, 1], f"[2]_{{q^3}} failed: {p}"
    print("  [2]_{q^3} = 1 + q^3                        OK")

    p = poly_mul_comb([1], 3, 2)
    assert p == [1, 0, 1, 0, 1], f"[3]_{{q^2}} failed: {p}"
    print("  [3]_{q^2} = 1 + q^2 + q^4                 OK")

    p = conjecture_poly([3, 2], 2)
    assert p == [1, 1, 2, 1, 1] and is_unimodal(p) and condition_holds([3, 2], 2)
    print("  P(3,2;2) = 1+q+2q^2+q^3+q^4  unimodal  condition OK")

    # Known necessity counterexample from the paper
    p = conjecture_poly([3, 3, 3, 3, 2], 4)
    assert p == [1, 4, 10, 16, 20, 20, 20, 20, 20, 16, 10, 4, 1]
    assert is_unimodal(p) and not condition_holds([3, 3, 3, 3, 2], 4)
    print(f"  P(3,3,3,3,2;4) coefficients = {p}")
    print("    unimodal=True, condition_holds=False      OK (necessity counterexample)")

    print("--- All sanity checks passed ---\n")
    return True


# ============================================================
# Test runners (parallel over (k,n) pairs)
# ============================================================

def run_sufficiency_test(max_k, max_n, max_a, verbose=True) -> List[Tuple]:
    """
    For each valid (multiset) prefix and conditioned a_k, verify unimodality.
    Parallelizes over (k,n) pairs using all CPU cores.
    """
    work = [(k, n, max_a)
            for n in range(2, max_n + 1)
            for k in range(2, max_k + 1)
            if [a for a in range(1, max_a + 1) if a % n != 0]]

    if verbose:
        print(f"\n{'='*70}")
        print(f"  Sufficiency test: k in [2,{max_k}], n in [2,{max_n}], "
              f"max_a={max_a}  ({os.cpu_count()} workers)")
        print(f"  (prefix enumerated as sorted multisets — ~(k-1)! deduplication)")
        print(f"{'='*70}")

    t0 = time.perf_counter()
    with mp.Pool(os.cpu_count()) as pool:
        results = pool.map(_worker_sufficiency, work)

    all_failures = []
    total_tested = 0
    for r in sorted(results, key=lambda x: (x["n"], x["k"])):
        total_tested += r["tested"]
        all_failures.extend(r["failures"])
        if verbose:
            status = "ok" if not r["failures"] else f"FAIL({len(r['failures'])})"
            tups_s = r["tested"] / r["elapsed"] if r["elapsed"] > 0 else 0
            print(f"  n={r['n']:2d} k={r['k']:2d}  "
                  f"multisets={r['tested']:8,d}  "
                  f"t={r['elapsed']:6.3f}s  "
                  f"{tups_s:,.0f}/s  {status}")

    elapsed = time.perf_counter() - t0
    if verbose:
        print(f"\n  Total multisets tested : {total_tested:,}")
        print(f"  Wall time (parallel)   : {elapsed:.2f}s")
        print(f"  Sufficiency failures   : {len(all_failures)}")
        for a_list, n in all_failures:
            print(f"    n={n}  a={a_list}")

    return all_failures


def run_necessity_test(max_k, max_n, max_a, verbose=True) -> List[Tuple]:
    """
    For k<=4 or n<=3, verify unimodal => condition holds.
    Parallelizes over applicable (k,n) pairs.
    """
    # Only run necessity check for (k,n) where conjecture claims necessity
    work_nec = [(k, n, max_a)
                for n in range(2, max_n + 1)
                for k in range(2, max_k + 1)
                if (k <= 4 or n <= 3)
                and [a for a in range(1, max_a + 1) if a % n != 0]]

    # Also run for k>=5,n>=4 to find expected non-necessity cases
    work_other = [(k, n, max_a)
                  for n in range(4, max_n + 1)
                  for k in range(5, max_k + 1)
                  if [a for a in range(1, max_a + 1) if a % n != 0]]

    work = work_nec + work_other

    if verbose:
        print(f"\n{'='*70}")
        print(f"  Necessity test: k in [2,{max_k}], n in [2,{max_n}], max_a={max_a}")
        print(f"  (necessity claimed only for k<=4 or n<=3)")
        print(f"{'='*70}")

    t0 = time.perf_counter()
    with mp.Pool(os.cpu_count()) as pool:
        results = pool.map(_worker_necessity, work)

    conj_failures = []
    other_unimodal = []
    for r in sorted(results, key=lambda x: (x["n"], x["k"])):
        conj_failures.extend(r["conj_failures"])
        other_unimodal.extend(r["other_unimodal"])
        if verbose and (r["k"] <= 4 or r["n"] <= 3):
            status = "ok" if not r["conj_failures"] else f"FAIL({len(r['conj_failures'])})"
            print(f"  n={r['n']:2d} k={r['k']:2d}  "
                  f"tested={r['tested']:8,d}  "
                  f"t={r['elapsed']:6.3f}s  {status}")

    elapsed = time.perf_counter() - t0
    if verbose:
        print(f"\n  Wall time (parallel)   : {elapsed:.2f}s")
        print(f"  Necessity failures (k<=4 or n<=3): {len(conj_failures)}")
        print(f"  Unimodal-despite-condition (k>=5,n>=4): "
              f"{len(other_unimodal)} (expected)")
        if other_unimodal:
            smallest = min(other_unimodal, key=lambda x: (len(x[0]), x[1], x[0]))
            print(f"  Smallest such case: n={smallest[1]}  a={smallest[0]}")

    return conj_failures


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    max_k, max_n, max_a = 6, 6, 20   # default: ~8s on M5 Pro

    args = sys.argv[1:]
    if "--full" in args:
        max_k, max_n, max_a = 8, 8, 20
        args = [a for a in args if a != "--full"]
        print("Full mode: k,n<=8, max_a=20  (warning: 20-30 min, individual pairs up to ~15s)")
        print("k>=9 with small n is not exhaustively feasible even with deduplication.")
    if len(args) >= 3:
        try:
            max_k, max_n, max_a = int(args[0]), int(args[1]), int(args[2])
        except ValueError:
            print(f"Usage: {sys.argv[0]} [--full] [max_k max_n max_a]")
            sys.exit(1)

    print(f"Parameters: max_k={max_k}, max_n={max_n}, max_a={max_a}, "
          f"workers={os.cpu_count()}")
    run_sanity_checks()
    suf = run_sufficiency_test(max_k, max_n, max_a)
    nec = run_necessity_test(max_k, max_n, max_a)

    print(f"\n{'='*70}")
    print("  SUMMARY")
    print(f"{'='*70}")
    print(f"  Sufficiency: {len(suf)} failures  "
          f"{'ALL OK' if not suf else 'CONJECTURE VIOLATED'}")
    print(f"  Necessity (k<=4 or n<=3): {len(nec)} failures  "
          f"{'ALL OK' if not nec else 'CONJECTURE VIOLATED'}")
    print("\nDone.")
