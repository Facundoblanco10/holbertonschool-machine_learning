#!/usr/bin/env python3
"""Class Binomial that represents a binomial distribution"""


class Binomial():
    """Binomial distribution"""

    def __init__(self, data=None, n=1, p=0.5):
        if data is None:
            if n <= 0:
                raise ValueError('n must be a positive value')
            self.n = int(n)
            if p >= 1 or p <= 0:
                raise ValueError('p must be greater than 0 and less than 1')
            self.p = float(p)
        else:
            if type(data) is not list:
                raise TypeError('data must be a list')
            if len(data) <= 1:
                raise ValueError('data must contain multiple values')
            mean = float(sum(data) / len(data))
            s = sum(map(lambda x: (x - mean) ** 2, data))
            var = s / len(data)
            self.n = round(mean / (- (var / mean) + 1))
            self.p = mean / self.n

    def pmf(self, k):
        """Calculates the value of the PMF for a given number of successes"""
        def factorial(n):
            if n <= 1:
                return 1
            return n * factorial(n - 1)
        k = int(k)
        if k < 0:
            return 0
        nk = factorial(self.n) / (factorial(k) * factorial((self.n - k)))
        return (nk * (self.p ** k) * ((1 - self.p) ** (self.n - k)))

    def cdf(self, k):
        """Calculates the value of the CDF for a given number of successes"""
        k = int(k)
        if k < 0:
            return (0)
        cdf = 0
        for i in range(k + 1):
            cdf += self.pmf(i)
        return cdf
