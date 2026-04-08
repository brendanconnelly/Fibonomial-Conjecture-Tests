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

Usage:
    python conjecture_test.py              # quick: k,n<=6, max_a<=15
    python conjecture_test.py --full       # full:  k,n<=10, max_a<=20
    python conjecture_test.py 8 8 18       # custom max_k max_n max_a
"""

import sys
import time
import itertools
from typing import List, Tuple

from fibonomial_test import poly_mul_all_ones, is_unimodal


def poly_mul_comb(P: List[int], a: int, n: int) -> List[int]:
    """
    Multiply P by [a]_{q^n} = 1 + q^n + ... + q^{(a-1)n}.

    Decomposes by residue mod n: for each r in {0,...,n-1}, extract the
    sub-sequence P[r::n], multiply it by [a]_Q (all-ones length a) via
    poly_mul_all_ones, then write back at stride n. O(len(P) + a*n).
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


def run_sufficiency_test(max_k, max_n, max_a, verbose=True) -> List[Tuple]:
    """
    For each valid tuple where the condition holds, verify the polynomial is unimodal.
    Enumerates only a_k values in [1, bound] to avoid generating and filtering all tuples.
    """
    failures = []
    total_t0 = time.perf_counter()
    if verbose:
        print(f"\n{'='*70}")
        print(f"  Sufficiency test: k in [2,{max_k}], n in [2,{max_n}], max_a={max_a}")
        print(f"{'='*70}")
    for n in range(2, max_n + 1):
        valid_pre = [a for a in range(1, max_a + 1) if a % n != 0]
        if not valid_pre:
            continue
        for k in range(2, max_k + 1):
            t0 = time.perf_counter()
            tested = 0
            for prefix in itertools.product(valid_pre, repeat=k - 1):
                bound = 1 + sum(a // n for a in prefix)
                for a_k in range(1, min(bound, max_a) + 1):
                    a_list = list(prefix) + [a_k]
                    tested += 1
                    if not is_unimodal(conjecture_poly(a_list, n)):
                        failures.append((a_list, n))
                        if verbose:
                            print(f"  FAIL  n={n} k={k}  a={a_list}")
            elapsed = time.perf_counter() - t0
            if verbose:
                status = "ok" if not any(f[1]==n and len(f[0])==k for f in failures) else "FAILURES"
                print(f"  n={n:2d} k={k:2d}  tested={tested:8,d}  t={elapsed:6.3f}s  {status}")
    if verbose:
        print(f"\n  Total time    : {time.perf_counter()-total_t0:.2f}s")
        print(f"  Sufficiency failures: {len(failures)}")
        for a_list, n in failures:
            print(f"    n={n}  a={a_list}")
    return failures


def run_necessity_test(max_k, max_n, max_a, verbose=True) -> List[Tuple]:
    """
    For k<=4 or n<=3, verify that unimodal => condition holds.
    Also tracks cases outside this range that are unimodal despite failing the condition
    (these are expected and do not contradict the conjecture).
    """
    counterexamples = []
    other_unimodal = []
    total_t0 = time.perf_counter()
    if verbose:
        print(f"\n{'='*70}")
        print(f"  Necessity test: k in [2,{max_k}], n in [2,{max_n}], max_a={max_a}")
        print(f"  (necessity claimed only for k<=4 or n<=3)")
        print(f"{'='*70}")
    for n in range(2, max_n + 1):
        valid_pre = [a for a in range(1, max_a + 1) if a % n != 0]
        if not valid_pre:
            continue
        for k in range(2, max_k + 1):
            t0 = time.perf_counter()
            tested = 0
            nec_failures = 0
            for prefix in itertools.product(valid_pre, repeat=k - 1):
                bound = 1 + sum(a // n for a in prefix)
                for a_k in range(bound + 1, max_a + 1):
                    a_list = list(prefix) + [a_k]
                    tested += 1
                    if is_unimodal(conjecture_poly(a_list, n)):
                        if k <= 4 or n <= 3:
                            counterexamples.append((a_list, n))
                            nec_failures += 1
                            if verbose:
                                print(f"  NECESSITY FAIL  n={n} k={k}  a={a_list}")
                        else:
                            other_unimodal.append((a_list, n))
            elapsed = time.perf_counter() - t0
            if verbose and (k <= 4 or n <= 3):
                status = "ok" if nec_failures == 0 else f"FAIL({nec_failures})"
                print(f"  n={n:2d} k={k:2d}  tested={tested:8,d}  t={elapsed:6.3f}s  {status}")
    if verbose:
        print(f"\n  Total time    : {time.perf_counter()-total_t0:.2f}s")
        print(f"  Necessity failures (k<=4 or n<=3): {len(counterexamples)}")
        print(f"  Unimodal-despite-condition (k>=5,n>=4): {len(other_unimodal)} (expected)")
        if other_unimodal:
            smallest = min(other_unimodal, key=lambda x: (len(x[0]), x[1], x[0]))
            print(f"  Smallest such case: n={smallest[1]}  a={smallest[0]}")
    return counterexamples


if __name__ == "__main__":
    max_k, max_n, max_a = 6, 6, 15

    args = sys.argv[1:]
    if "--full" in args:
        max_k, max_n, max_a = 10, 10, 20
        args = [a for a in args if a != "--full"]
        print("Full mode: k,n<=10, max_a=20  (slow for large k)")
    if len(args) >= 3:
        try:
            max_k, max_n, max_a = int(args[0]), int(args[1]), int(args[2])
        except ValueError:
            print(f"Usage: {sys.argv[0]} [--full] [max_k max_n max_a]")
            sys.exit(1)

    print(f"Parameters: max_k={max_k}, max_n={max_n}, max_a={max_a}")
    run_sanity_checks()
    suf = run_sufficiency_test(max_k, max_n, max_a)
    nec = run_necessity_test(max_k, max_n, max_a)

    print(f"\n{'='*70}")
    print("  SUMMARY")
    print(f"{'='*70}")
    print(f"  Sufficiency: {len(suf)} failures  {'ALL OK' if not suf else 'CONJECTURE VIOLATED'}")
    print(f"  Necessity (k<=4 or n<=3): {len(nec)} failures  {'ALL OK' if not nec else 'CONJECTURE VIOLATED'}")
    print("\nDone.")
