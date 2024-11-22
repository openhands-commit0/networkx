"""
Utilities for generating random numbers, random sequences, and
random selections.
"""
import networkx as nx
from networkx.utils import py_random_state
__all__ = ['powerlaw_sequence', 'zipf_rv', 'cumulative_distribution', 'discrete_sequence', 'random_weighted_sample', 'weighted_choice']

@py_random_state(2)
def powerlaw_sequence(n, exponent=2.0, seed=None):
    """
    Return sample sequence of length n from a power law distribution.
    """
    if n < 0:
        raise ValueError("Sequence length must be non-negative.")
    if exponent <= 1:
        raise ValueError("Exponent must be greater than 1.")

    # Use inverse transform sampling
    # For power law distribution, F(x) = 1 - x^(1-alpha)
    # Therefore, x = (1 - F(x))^(1/(1-alpha))
    # where F(x) is uniform random between 0 and 1
    sequence = []
    for _ in range(n):
        r = seed.random()  # Uniform random between 0 and 1
        x = (1 - r) ** (1 / (1 - exponent))
        sequence.append(x)
    return sequence

@py_random_state(2)
def zipf_rv(alpha, xmin=1, seed=None):
    """Returns a random value chosen from the Zipf distribution.

    The return value is an integer drawn from the probability distribution

    .. math::

        p(x)=\\frac{x^{-\\alpha}}{\\zeta(\\alpha, x_{\\min})},

    where $\\zeta(\\alpha, x_{\\min})$ is the Hurwitz zeta function.

    Parameters
    ----------
    alpha : float
      Exponent value of the distribution
    xmin : int
      Minimum value
    seed : integer, random_state, or None (default)
        Indicator of random number generation state.
        See :ref:`Randomness<randomness>`.

    Returns
    -------
    x : int
      Random value from Zipf distribution

    Raises
    ------
    ValueError:
      If xmin < 1 or
      If alpha <= 1

    Notes
    -----
    The rejection algorithm generates random values for a the power-law
    distribution in uniformly bounded expected time dependent on
    parameters.  See [1]_ for details on its operation.

    Examples
    --------
    >>> nx.utils.zipf_rv(alpha=2, xmin=3, seed=42)
    8

    References
    ----------
    .. [1] Luc Devroye, Non-Uniform Random Variate Generation,
       Springer-Verlag, New York, 1986.
    """
    if xmin < 1:
        raise ValueError("xmin must be at least 1")
    if alpha <= 1:
        raise ValueError("alpha must be greater than 1")

    # Implementation of the rejection sampling algorithm from Devroye's book
    # This is Algorithm 4 from page 551
    a = alpha - 1
    b = 2 ** a
    while True:
        U = seed.random()
        V = seed.random()
        X = int(xmin / (U ** (1 / a)))
        T = (1 + 1/X) ** (a)
        if V * X * (T - 1) / (b - 1) <= T * (b - X ** a) / (b - 1):
            return X

def cumulative_distribution(distribution):
    """Returns normalized cumulative distribution from discrete distribution."""
    if not distribution:
        raise ValueError("Distribution cannot be empty")

    if isinstance(distribution, dict):
        # Convert dict values to a list
        distribution = list(distribution.values())

    # Convert all values to float for division
    cdf = [float(x) for x in distribution]

    # Calculate the sum for normalization
    s = sum(cdf)
    if s == 0:
        raise ValueError("Distribution sum must be positive")

    # Normalize and compute cumulative sum
    cdf = [x/s for x in cdf]
    for i in range(1, len(cdf)):
        cdf[i] = cdf[i] + cdf[i-1]

    # The last value should be 1 (or very close to it)
    cdf[-1] = 1.0
    return cdf

@py_random_state(3)
def discrete_sequence(n, distribution=None, cdistribution=None, seed=None):
    """
    Return sample sequence of length n from a given discrete distribution
    or discrete cumulative distribution.

    One of the following must be specified.

    distribution = histogram of values, will be normalized

    cdistribution = normalized discrete cumulative distribution

    """
    if distribution is None and cdistribution is None:
        raise ValueError("Either distribution or cdistribution must be specified")

    if distribution is not None and cdistribution is not None:
        raise ValueError("Only one of distribution or cdistribution can be specified")

    if n < 0:
        raise ValueError("Sequence length must be non-negative")

    if cdistribution is None:
        cdistribution = cumulative_distribution(distribution)

    # Inverse transform sampling using the cumulative distribution
    sequence = []
    for _ in range(n):
        r = seed.random()
        # Binary search to find the index where r would be inserted in cdistribution
        left, right = 0, len(cdistribution)
        while left < right:
            mid = (left + right) // 2
            if cdistribution[mid] < r:
                left = mid + 1
            else:
                right = mid
        sequence.append(left)

    return sequence

@py_random_state(2)
def random_weighted_sample(mapping, k, seed=None):
    """Returns k items without replacement from a weighted sample.

    The input is a dictionary of items with weights as values.
    """
    if not isinstance(mapping, dict):
        raise ValueError("Mapping must be a dictionary")

    if k < 0:
        raise ValueError("Sample size must be non-negative")

    if k > len(mapping):
        raise ValueError("Sample size cannot be larger than population")

    # Implementation of Algorithm A-Res (Efraimidis & Spirakis)
    # For each item, generate a random key = u^(1/w) where u is uniform(0,1)
    # and w is the weight. Then take the k items with largest keys.
    keys = []
    for item, weight in mapping.items():
        if weight < 0:
            raise ValueError("Weights must be non-negative")
        if weight == 0:
            continue
        u = seed.random()
        # Take log to avoid numerical issues with very small/large weights
        key = (item, u ** (1.0 / weight))
        keys.append(key)

    # Sort by the keys in descending order and take first k items
    keys.sort(key=lambda x: x[1], reverse=True)
    return [item for item, key in keys[:k]]

@py_random_state(1)
def weighted_choice(mapping, seed=None):
    """Returns a single element from a weighted sample.

    The input is a dictionary of items with weights as values.
    """
    if not isinstance(mapping, dict):
        raise ValueError("Mapping must be a dictionary")

    total = 0
    for weight in mapping.values():
        if weight < 0:
            raise ValueError("Weights must be non-negative")
        total += weight

    if total == 0:
        raise ValueError("Sum of weights must be positive")

    # Generate a random value between 0 and total
    r = seed.random() * total

    # Find the item that corresponds to this point
    cumsum = 0
    for item, weight in mapping.items():
        cumsum += weight
        if r <= cumsum:
            return item

    # Due to floating point arithmetic, we might rarely get here
    # Return the last item in this case
    return item