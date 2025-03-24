from sympy import factorint, proper_divisors, is_perfect, log
import numpy as np
from itertools import chain, combinations
from numba import jit
from src import numberdata as nd

# Constants
MAX_PRIMITIVE = 358671
MAX_PSEUDO = 40656


def perfection_entropy(
    n: int, optimise: bool = True, stop: int = 5, floor: bool = True
) -> list:
    """Calculates the entropy of a pseudoperfect n with parts of the additive
    partition as random variables."""

    try:
        # Check if the input is an integer
        if not isinstance(n, int) or n < 1:
            raise ValueError("Input must be a positive integer.")

        # Handle large integers by using dtype=object
        if n > np.iinfo(np.int64).max:  # If n exceeds int64 range
            # array_type = object
            raise ValueError(f"Input must be less than {np.iinfo(np.uint64).max}.")
        else:
            array_type = np.uint64

        # If n is perfect, calculate its entropy
        if is_perfect(n):
            divisors = tuple(proper_divisors(n))
            return [tuple([H(n, np.array(divisors, dtype=array_type)), divisors])]
        else:
            # For pseudoperfect numbers, handle each partition
            perfections = get_perfections(n, optimise, stop, floor)
            return [(H(n, np.array(i, dtype=array_type)), i) for i in perfections]

    except ValueError as ve:
        print(f"ValueError: {ve}")
        return []


def is_primitive(n: int) -> bool:
    """Check if a number is a primitive pseudoperfect number."""
    nd.load_primitives()
    if n <= MAX_PRIMITIVE:
        return np.isin(n, nd.PRIMITIVES)
    else:
        print(
            f"The value n = {n} exceeds the maximum data value in the\
              primitive_pseudoperfect_list.txt file: {MAX_PRIMITIVE}"
        )


def is_pseudoperfect(n: int) -> bool:
    """Check if a number is a pseudoperfect number."""
    nd.load_pseudos()
    if n <= MAX_PSEUDO:
        return np.isin(n, nd.PSEUDOS)
    else:
        print(
            f"The value n = {n} exceeds the maximum data value in the\
              pseudoperfect_list.txt file: {MAX_PSEUDO}"
        )


def subpowerset(n: int, divisors: list, floor: bool = True) -> chain:
    """Generate subsets of divisors with a minimum size. (The minimum size of
    the returned sets is fixed by
    the floor function of the log n. This avoids collecting smaller subsets
    that can't add up to the primitive pseudoperfect n. Also, the full set
    isn't returned as this is only useful for perfect n.)"""
    if floor:
        if is_primitive(n):
            start = max(int(np.log(n)), 3)  # Avoids small subsets that can't add up to n.
        else:
            start = 3
        return chain.from_iterable(
            combinations(divisors, r) for r in range(start, len(divisors))
        )
    else:
        return chain.from_iterable(
            combinations(divisors, r) for r in range(3, len(divisors))
        )


def get_perfections(
    n: int, optimise: bool = True, stop: int = 5, floor: bool = True
) -> list:
    """Find subsets of proper divisors of n that sum to n."""
    divisor_combos = subpowerset(n, proper_divisors(n), floor)

    perfections = []

    for divisors in divisor_combos:
        if sum(divisors) == n:
            perfections.append(divisors)
            if optimise and len(perfections) >= stop:
                break
    return perfections


@jit
def H(total: int, parts: np.ndarray) -> float:
    """Calculates Shannon entropy given a partition and its sum."""
    return - np.sum(parts/total * np.log2(parts/total))


def H_precise(number: int, partition: list, precision: int = 6) -> float:
    """Function that calculates the entropy of a partition to arbitrary\
        precision. Set precision to 'inf' for closed form expression."""
    if precision == 'inf':
        return sum(
            divisor/number * log(number/divisor, 2) for divisor in partition)
    else:
        return round(sum(
            divisor/number * log(number/divisor, 2).evalf(precision) for divisor in partition), precision)


@jit
def multiplicity_entropy(n: int) -> float:
    """Calculate entropy based on prime factor multiplicities."""
    factorisation = factorint(n)
    exponents = np.array(list(factorisation.values()))
    if exponents.size == 1:
        return 0
    total = np.sum(exponents)
    return H(total, exponents)


def floorless_perfection_entropy(n, optimise=True, stop=5):
    """Calculates the entropy of a pseudoperfect n with parts of the additive
    partition as random variables. Without a lower limit for the size of
    divisor subsets.."""

    def subpowerset(n, divisors_iterable):
        """Returns a subset of the powerset of a given set of nonperfect
        pseudoperfect numbers. (The minimum size of the returned sets is fixed by
        the floor function of the log n. This avoids collecting smaller subsets
        that can't add up to the primitive pseudoperfect n. Also, the full set
        isn't returned as this is only useful for perfect n.)"""
        s = list(divisors_iterable)

        return chain.from_iterable(
            combinations(s, r) for r in range(3, len(s)))

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

    try:
        # Check if the input is an integer
        if not isinstance(n, int) or n < 1:
            raise ValueError("Input must be a positive integer.")

        # Handle large integers by using dtype=object
        if n > np.iinfo(np.int64).max:  # If n exceeds int64 range
            # array_type = object
            raise ValueError(
                f"Input must be less than {np.iinfo(np.uint64).max}.")
        else:
            array_type = np.uint64

        # If n is perfect, calculate its entropy
        if is_perfect(n):
            divisors = tuple(proper_divisors(n))
            return [tuple([H(n, np.array(divisors, dtype=array_type)), divisors])]
        else:
            # For pseudoperfect numbers, handle each partition
            perfections = get_perfections(n, optimise, stop)
            return [
                (H(n, np.array(i, dtype=array_type)), i) for i in perfections]

    except ValueError as ve:
        print(f"ValueError: {ve}")
        return []


@jit
def S(total: int, parts: np.ndarray, q: int = 2) -> float:
    """Calculates logical entropy given a partition and its sum."""
    return (1 - np.sum((parts/total)**q)) / (q - 1)


def Tsallis_PE(n: int, q: int = 2, optimise: bool = True, stop: int = 5) -> list:
    """Calculates the entropy of a pseudoperfect n with parts of the additive
    partition as random variables."""

    try:
        # Check if the input is an integer
        if not isinstance(n, int) or n < 1:
            raise ValueError("Input must be a positive integer.")

        # Handle large integers by using dtype=object
        if n > np.iinfo(np.int64).max:  # If n exceeds int64 range
            # array_type = object
            raise ValueError(
                f"Input must be less than {np.iinfo(np.uint64).max}.")
        else:
            array_type = np.uint64

        # If n is perfect, calculate its entropy
        if is_perfect(n):
            divisors = tuple(proper_divisors(n))
            return [tuple([LH(n, q, np.array(divisors, dtype=array_type)), divisors])]
        else:
            # For pseudoperfect numbers, handle each partition
            perfections = get_perfections(n, optimise, stop)
            return [
                (LH(n, q, np.array(i, dtype=array_type)), i) for i in perfections]

    except ValueError as ve:
        print(f"ValueError: {ve}")
        return []
