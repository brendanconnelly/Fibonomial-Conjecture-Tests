#!/usr/bin/env python3
"""
Computational test of the unimodality conjecture:

    Conjecture. Let k, n >= 2. For each 1 <= i <= k, let a_i be a positive
    integer and suppose a_i is not divisible by n for all 1 <= i < k.
    If
        a_k <= 1 + sum_{i=1}^{k-1} floor(a_i / n),
    then the polynomial
        P(a_1,...,a_k; n) = [a_1]_q ... [a_{k-1}]_q [a_k]_{q^n}
    is unimodal. Moreover, if k <= 4 or n <= 3, this condition is also
    necessary.

    Known counterexample to necessity (k=5, n=4):
        P(3,3,3,3,2; 4) = [3]_q^4 [2]_{q^4}
    is unimodal, but 2 > 1 + 4*floor(3/4) = 1.

Notation:
    [r]_q     = 1 + q + q^2 + ... + q^{r-1}      (q-integer)
    [a]_{q^n} = 1 + q^n + q^{2n} + ... + q^{(a-1)n}  (comb, step n)

Usage:
    python conjecture_test.py                   # quick: k,n<=6, max_a<=15
    python conjecture_test.py --full            # full:  k,n<=10, max_a<=20
    python conjecture_test.py 8 8 18            # custom max_k max_n max_a
"""

import sys
import time
import itertools
from typing import List, Tuple

from fibonomial_test import poly_mul_all_ones, poly_mul, is_unimodal


# ============================================================
# New polynomial primitive: multiply by a comb [a]_{q^n}
# ============================================================

def poly_mul_comb(P: List[int], a: int, n: int) -> List[int]:
    """
    Multiply polynomial P by [a]_{q^n} = 1 + q^n + q^{2n} + ... + q^{(a-1)*n}.

    Algorithm — interleaved prefix-sum, O(len(P) + a*n):
        Observe that [a]_{q^n} acts on P by:
            result[j] = sum_{l=0}^{a-1} P[j - l*n]   (P[m] = 0 for m < 0)

        Decompose by residue r = j mod n. For each r in {0,...,n-1}, the
        sub-sequence (P[r], P[r+n], P[r+2n], ...) gets multiplied by
        [a]_Q = 1 + Q + ... + Q^{a-1} where Q = q^n, via poly_mul_all_ones.
        The results are interleaved back into the output at stride n.

    Special case: a=1 returns a copy of P (multiply by 1).
    """
    if a == 1:
        return list(P)

    len_P = len(P)
    result_len = len_P + (a - 1) * n
    result = [0] * result_len

    for r in range(n):
        # Extract sub-sequence at positions r, r+n, r+2n, ...
        sub = P[r::n]                          # length = ceil((len_P - r) / n)
        if not sub:
            continue
        # Multiply sub by all-ones of length a (same as multiplying by [a]_Q)
        sub_out = poly_mul_all_ones(sub, a)    # length = len(sub) + a - 1
        # Write back at positions r, r+n, r+2n, ...
        result[r::n] = sub_out

    return result


# ============================================================
# Conjecture polynomial and condition
# ============================================================

def conjecture_poly(a_list: List[int], n: int) -> List[int]:
    """
    Compute P(a_1,...,a_k; n) = [a_1]_q ... [a_{k-1}]_q * [a_k]_{q^n}.
    """
    P = [1]
    for a in a_list[:-1]:
        P = poly_mul_all_ones(P, a)
    P = poly_mul_comb(P, a_list[-1], n)
    return P


def condition_holds(a_list: List[int], n: int) -> bool:
    """
    Check a_k <= 1 + sum_{i<k} floor(a_i / n).
    """
    a_k = a_list[-1]
    return a_k <= 1 + sum(a // n for a in a_list[:-1])


# ============================================================
# Sanity checks
# ============================================================

def run_sanity_checks() -> bool:
    """
    Verify small known cases. Returns True iff all pass.
    """
    print("\n--- Sanity checks ---")
    ok = True

    # [a]_{q^n} when a=1 should be 1
    p = poly_mul_comb([1], 1, 3)
    assert p == [1], f"[1]_{{q^3}} failed: {p}"
    print("  [1]_{q^3} = 1                              OK")

    # [2]_{q^3} = 1 + q^3, i.e. [1, 0, 0, 1]
    p = poly_mul_comb([1], 2, 3)
    assert p == [1, 0, 0, 1], f"[2]_{{q^3}} failed: {p}"
    print("  [2]_{q^3} = 1 + q^3                        OK")

    # [3]_{q^2} = 1 + q^2 + q^4
    p = poly_mul_comb([1], 3, 2)
    assert p == [1, 0, 1, 0, 1], f"[3]_{{q^2}} failed: {p}"
    print("  [3]_{q^2} = 1 + q^2 + q^4                 OK")

    # P(3; 2) = [3]_{q^2} — no prefix factors
    # (k=1 edge case: the whole product is just the comb)
    p = conjecture_poly([3], 2)
    assert p == [1, 0, 1, 0, 1], f"P(3;2) failed: {p}"
    print("  P(3;2) = [3]_{q^2} = 1+q^2+q^4            OK")

    # P(3,2;2) = [3]_q * [2]_{q^2} = (1+q+q^2)(1+q^2) = 1+q+2q^2+q^3+q^4
    p = conjecture_poly([3, 2], 2)
    assert p == [1, 1, 2, 1, 1], f"P(3,2;2) failed: {p}"
    assert is_unimodal(p), "P(3,2;2) should be unimodal"
    # Condition: a_2=2 <= 1 + floor(3/2)=1 -> 2 <= 2. True.
    assert condition_holds([3, 2], 2), "condition for P(3,2;2) should hold"
    print("  P(3,2;2) = 1+q+2q^2+q^3+q^4  unimodal  condition OK")

    # Known counterexample to necessity: P(3,3,3,3,2; 4)
    # Should be unimodal, but condition fails: 2 > 1 + 4*floor(3/4) = 1
    p_counter = conjecture_poly([3, 3, 3, 3, 2], 4)
    assert is_unimodal(p_counter), "P(3,3,3,3,2;4) should be unimodal"
    assert not condition_holds([3, 3, 3, 3, 2], 4), \
        "condition for P(3,3,3,3,2;4) should fail"
    # Verify the exact coefficients from the paper
    expected = [1, 4, 10, 16, 20, 20, 20, 20, 20, 16, 10, 4, 1]
    assert p_counter == expected, f"P(3,3,3,3,2;4) coefficients wrong: {p_counter}"
    print("  P(3,3,3,3,2;4) = [3]_q^4[2]_{q^4}")
    print(f"    coefficients = {p_counter}")
    print("    unimodal=True, condition_holds=False      OK (necessity counterexample)")

    print("--- All sanity checks passed ---\n")
    return ok


# ============================================================
# Test runners
# ============================================================

def run_sufficiency_test(
    max_k: int, max_n: int, max_a: int, verbose: bool = True
) -> List[Tuple]:
    """
    For every valid (k, n, a_1,...,a_k) where the conjecture condition holds,
    verify the polynomial is unimodal.

    Enumeration strategy (avoids generating all tuples then filtering):
      For each prefix (a_1,...,a_{k-1}), compute the max allowed a_k from
      the condition, then only loop a_k over [1, min(bound, max_a)].

    Returns list of failures: (a_list, n) where condition held but poly not unimodal.
    """
    failures = []
    total_t0 = time.perf_counter()

    if verbose:
        print(f"\n{'='*70}")
        print(f"  Sufficiency test: k in [2,{max_k}], n in [2,{max_n}], "
              f"max_a={max_a}")
        print(f"{'='*70}")

    for n in range(2, max_n + 1):
        for k in range(2, max_k + 1):
            pair_t0 = time.perf_counter()
            tested = 0
            # Values valid for prefix positions: not divisible by n
            valid_pre = [a for a in range(1, max_a + 1) if a % n != 0]
            if not valid_pre:
                continue

            for prefix in itertools.product(valid_pre, repeat=k - 1):
                # Maximum a_k satisfying the condition
                bound = 1 + sum(a // n for a in prefix)
                for a_k in range(1, min(bound, max_a) + 1):
                    a_list = list(prefix) + [a_k]
                    poly = conjecture_poly(a_list, n)
                    tested += 1
                    if not is_unimodal(poly):
                        failures.append((a_list, n))
                        if verbose:
                            print(f"  FAIL  n={n} k={k}  a={a_list}")

            elapsed = time.perf_counter() - pair_t0
            if verbose:
                status = "ok" if not any(
                    f[1] == n and len(f[0]) == k for f in failures
                ) else "FAILURES"
                print(f"  n={n:2d} k={k:2d}  tested={tested:8,d}  "
                      f"t={elapsed:6.3f}s  {status}")

    total_elapsed = time.perf_counter() - total_t0
    if verbose:
        print(f"\n  Total time    : {total_elapsed:.2f}s")
        print(f"  Sufficiency failures: {len(failures)}")
        if failures:
            for a_list, n in failures:
                print(f"    n={n}  a={a_list}")

    return failures


def run_necessity_test(
    max_k: int, max_n: int, max_a: int, verbose: bool = True
) -> List[Tuple]:
    """
    For (k, n) with k<=4 or n<=3 only, check that whenever the condition fails,
    the polynomial is NOT unimodal.

    Returns list of counterexamples to necessity: (a_list, n) where condition
    fails but the polynomial is unimodal (would disprove the conjecture).
    Also collects unimodal-despite-condition-failing cases for k>=5, n>=4
    (these are expected — the conjecture says nothing about necessity there).
    """
    counterexamples_nec = []   # k<=4 or n<=3: these would break the conjecture
    other_unimodal = []        # k>=5 and n>=4: expected counterexamples to necessity

    total_t0 = time.perf_counter()

    if verbose:
        print(f"\n{'='*70}")
        print(f"  Necessity test: k in [2,{max_k}], n in [2,{max_n}], "
              f"max_a={max_a}")
        print(f"  (conjecture claims necessity only for k<=4 or n<=3)")
        print(f"{'='*70}")

    for n in range(2, max_n + 1):
        for k in range(2, max_k + 1):
            pair_t0 = time.perf_counter()
            tested = 0
            nec_failures = 0
            valid_pre = [a for a in range(1, max_a + 1) if a % n != 0]
            if not valid_pre:
                continue

            for prefix in itertools.product(valid_pre, repeat=k - 1):
                bound = 1 + sum(a // n for a in prefix)
                # Loop over a_k values that VIOLATE the condition
                for a_k in range(bound + 1, max_a + 1):
                    a_list = list(prefix) + [a_k]
                    poly = conjecture_poly(a_list, n)
                    tested += 1
                    if is_unimodal(poly):
                        if k <= 4 or n <= 3:
                            counterexamples_nec.append((a_list, n))
                            nec_failures += 1
                            if verbose:
                                print(f"  NECESSITY FAIL  n={n} k={k}  a={a_list}")
                        else:
                            other_unimodal.append((a_list, n))

            elapsed = time.perf_counter() - pair_t0
            if verbose and (k <= 4 or n <= 3):
                status = "ok" if nec_failures == 0 else f"FAIL({nec_failures})"
                print(f"  n={n:2d} k={k:2d}  tested={tested:8,d}  "
                      f"t={elapsed:6.3f}s  {status}")

    total_elapsed = time.perf_counter() - total_t0
    if verbose:
        print(f"\n  Total time    : {total_elapsed:.2f}s")
        print(f"  Necessity failures (k<=4 or n<=3): {len(counterexamples_nec)}")
        print(f"  Unimodal-despite-condition (k>=5,n>=4): "
              f"{len(other_unimodal)} "
              f"(expected; condition not claimed necessary there)")
        if other_unimodal:
            # Show the smallest by a_k to match paper's 'smallest counterexample'
            smallest = min(other_unimodal, key=lambda x: (len(x[0]), x[1], x[0]))
            print(f"  Smallest such case: n={smallest[1]}  a={smallest[0]}")

    return counterexamples_nec


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    # --- Parse arguments ---
    max_k, max_n, max_a = 6, 6, 15    # quick defaults

    args = sys.argv[1:]
    if "--full" in args:
        max_k, max_n, max_a = 10, 10, 20
        args = [a for a in args if a != "--full"]
        print("Full mode: k,n<=10, max_a=20  (this may be slow for large k)")

    if len(args) >= 3:
        try:
            max_k, max_n, max_a = int(args[0]), int(args[1]), int(args[2])
        except ValueError:
            print(f"Usage: {sys.argv[0]} [--full] [max_k max_n max_a]")
            sys.exit(1)
    elif len(args) == 1 and args[0].lstrip('-').isdigit():
        # Legacy: single integer treated as max_k=max_n=that value
        val = int(args[0])
        max_k, max_n, max_a = val, val, max_a

    print(f"Parameters: max_k={max_k}, max_n={max_n}, max_a={max_a}")

    # --- Sanity checks ---
    run_sanity_checks()

    # --- Sufficiency test ---
    suf_failures = run_sufficiency_test(max_k, max_n, max_a)

    # --- Necessity test ---
    nec_failures = run_necessity_test(max_k, max_n, max_a)

    # --- Final summary ---
    print(f"\n{'='*70}")
    print("  SUMMARY")
    print(f"{'='*70}")
    print(f"  Sufficiency (condition => unimodal):")
    print(f"    Failures: {len(suf_failures)}  "
          f"{'ALL OK' if not suf_failures else 'CONJECTURE VIOLATED'}")
    print(f"  Necessity (unimodal => condition, for k<=4 or n<=3):")
    print(f"    Failures: {len(nec_failures)}  "
          f"{'ALL OK' if not nec_failures else 'CONJECTURE VIOLATED'}")
    print("\nDone.")
