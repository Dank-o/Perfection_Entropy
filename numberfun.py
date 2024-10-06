from sympy import factorint, proper_divisors, is_perfect
import numpy as np
from itertools import chain, combinations
from numba import jit
import numberdata as nd


def is_primitive(n):
    """Returns True if n < 358,671 is a primitive pseudoperfect number, else
    False."""
    nd.load_primitives()
    if n < nd.PRIMITIVES[-1]:
        return np.isin(n, nd.PRIMITIVES)
    else:
        print(f"The value n = {n} exceeds the maximum data value, \
              {nd.PRIMITIVES[-1]}")


def is_pseudoperfect(n):
    """Returns True if n < 40,656 is a pseudoperfect number, else False."""
    nd.load_pseudos()
    if n < nd.PSEUDOS[-1]:
        return np.isin(n, nd.PSEUDOS)
    else:
        print(f"The value n = {n} exceeds the maximum data value, \
              {nd.PSEUDOS[-1]}")


def subpowerset(n, divisors_iterable):
    """Returns a subset of the powerset of a given set of nonperfect pp numbers
    . (The minimum size of the returned sets is fixed by the floor function of
    the log n. This avoids collecting smaller subsets that can't add up to the
    primitive pseudoperfect n. Also, the full set isn't returned as this is
    only useful for perfect n.)"""
    s = list(divisors_iterable)

    if is_primitive(n):
        floor = int(np.log(n))  # Avoids small subsets that can't add up to n.
    else:
        floor = 2
    return chain.from_iterable(
        combinations(s, r) for r in range(floor, len(s)))


def get_perfections(n, optimise=True, stop=5):
    """Returns tuples of proper divisors of a given pseudoperfect n that add up
    to n."""
    divisor_combos = subpowerset(n, proper_divisors(n))

    if optimise:
        perfection = []
        for divisors in divisor_combos:
            if sum(divisors) == n:
                perfection.append(divisors)
                if len(perfection) >= stop:
                    break
        return perfection

    else:
        return [divisors for divisors in divisor_combos if sum(divisors) == n]


def perfection_entropy(n, optimise=True, stop=5):
    """Calculates the entropy of a pseudoperfect n with parts of the additive
    partition as random variables."""
    if is_perfect(n):
        return [H(n, np.array(proper_divisors(n)))]
    else:
        return [H(n, np.array(i)) for i in get_perfections(n, optimise, stop)]


@jit
def H(total, parts):
    """Calculates Shannon entropy given a partition and its sum."""
    return - np.sum(parts/total * np.log2(parts/total))


@jit
def multiplicity_entropy(n):
    """Calculates the entropy of an integer, with multiplicities of its prime
    divisors  as random variables."""
    factorisation = factorint(n)
    exponents = np.array(list(factorisation.values()))

    if exponents.size == 1:
        return 0
    total = np.sum(exponents)

    return H(total, exponents)
