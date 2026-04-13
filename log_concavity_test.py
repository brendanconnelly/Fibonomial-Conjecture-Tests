#!/usr/bin/env python3
"""
Check log-concavity of q-Fibonomial polynomial coefficients.

A sequence c_0, ..., c_d is log-concave if c_i^2 >= c_{i-1} * c_{i+1} for all
interior i. Log-concavity is strictly stronger than unimodality.

Coded with the assistance of Claude (Anthropic).

Searches (m,n) pairs in order of increasing m+n. Stops after MAX_FAILURES
failures or when m+n exceeds MAX_SUM (beyond which computation gets slow).
Prints the smallest failure in detail.

Usage:
    python3 log_concavity_test.py [max_sum]   # default max_sum=26
"""

import sys
import time
from fibonomial_test import fibonomial_poly, make_fibonacci, degree_formula

MAX_FAILURES = 5
DEFAULT_MAX_SUM = 26   # degree ~124k at m=n=13; beyond this gets slow


def first_log_concavity_violation(coeffs):
    """
    Returns the index i of the first violation of c_i^2 >= c_{i-1}*c_{i+1},
    or -1 if the sequence is log-concave.
    """
    for i in range(1, len(coeffs) - 1):
        if coeffs[i] * coeffs[i] < coeffs[i - 1] * coeffs[i + 1]:
            return i
    return -1


def format_poly(coeffs, max_terms=12):
    """
    Format polynomial as a readable sum of terms, truncating to max_terms on
    each side of the peak (since the polynomial is palindromic, we only need
    to show the first half and note the symmetry).
    """
    d = len(coeffs) - 1
    terms = []
    for i, c in enumerate(coeffs):
        if c == 0:
            continue
        if i == 0:
            terms.append(str(c))
        elif i == 1:
            terms.append(f"{c}q" if c != 1 else "q")
        else:
            terms.append(f"{c}q^{i}" if c != 1 else f"q^{i}")
    if len(terms) <= 2 * max_terms:
        return " + ".join(terms)
    # Truncate: show first and last max_terms with ellipsis
    shown = terms[:max_terms] + ["..."] + terms[-max_terms:]
    return " + ".join(shown)


def format_violation(m, n, coeffs, idx):
    """Print a detailed report of a log-concavity failure."""
    d = len(coeffs) - 1
    ci = coeffs[idx]
    cl = coeffs[idx - 1]
    cr = coeffs[idx + 1]
    deficit = cl * cr - ci * ci
    print(f"\n  {'='*60}")
    print(f"  Failure: C({m+n},{n})_F   m={m}, n={n}")
    print(f"  Degree : {d:,}")
    print(f"  Poly   : {format_poly(coeffs)}")
    print(f"  Violation at index i={idx}:")
    print(f"    c_{{i-1}} = {cl}")
    print(f"    c_{{i}}   = {ci}")
    print(f"    c_{{i+1}} = {cr}")
    print(f"    c_i^2 - c_{{i-1}}*c_{{i+1}} = {ci*ci} - {cl*cr} = {ci*ci - cl*cr}  (<0)")
    print(f"  {'='*60}")


if __name__ == "__main__":
    max_sum = DEFAULT_MAX_SUM
    if len(sys.argv) > 1:
        try:
            max_sum = int(sys.argv[1])
        except ValueError:
            print(f"Usage: {sys.argv[0]} [max_sum]")
            sys.exit(1)

    F = make_fibonacci(max_sum + 3)
    failures = []   # list of (m, n, coeffs, violation_index)

    print(f"Checking log-concavity of C(m+n,n)_F for m,n>=1, m+n<={max_sum}")
    print(f"Stopping after {MAX_FAILURES} failures.\n")
    print(f"  {'m+n':>4}  {'m':>4}  {'n':>4}  {'deg':>10}  {'t':>7}  result")
    print(f"  {'-'*55}")

    for s in range(2, max_sum + 1):
        for n in range(1, s):
            m = s - n
            t0 = time.perf_counter()
            poly = fibonomial_poly(m, n, F)
            elapsed = time.perf_counter() - t0

            idx = first_log_concavity_violation(poly)
            status = "log-concave" if idx == -1 else f"FAIL at i={idx}"
            print(f"  {s:>4}  {m:>4}  {n:>4}  {len(poly)-1:>10,}  {elapsed:>6.3f}s  {status}")

            if idx != -1:
                failures.append((m, n, poly, idx))
                if len(failures) >= MAX_FAILURES:
                    print(f"\n  Reached {MAX_FAILURES} failures — stopping.")
                    break
        else:
            continue
        break   # propagate inner break

    print(f"\n  Total failures found: {len(failures)}")

    if failures:
        # Smallest = first in search order (smallest m+n, then smallest n)
        m, n, coeffs, idx = failures[0]
        print(f"\nSmallest failure:")
        format_violation(m, n, coeffs, idx)

        if len(failures) > 1:
            print(f"\nAll failures:")
            for m, n, coeffs, idx in failures:
                ci, cl, cr = coeffs[idx], coeffs[idx-1], coeffs[idx+1]
                print(f"    C({m+n},{n})_F  m={m} n={n}  "
                      f"i={idx}  c_i^2-c_{{i-1}}c_{{i+1}}={ci*ci - cl*cr}")
    else:
        print(f"  No log-concavity failures found up to m+n={max_sum}.")
