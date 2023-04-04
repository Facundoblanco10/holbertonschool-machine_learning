#!/usr/bin/env python3
"""class Poisson that represents a poisson distribution"""


class Poisson():
    """Class Poisson"""

    def __init__(self, data=None, lambtha=1.):
        if data is None:
            if lambtha <= 0:
                raise ValueError('lambtha must be a positive value')
            self.lambtha = float(lambtha)
        else:
            if type(data) is not list:
                raise TypeError('data must be a list')
            if len(data) <= 1:
                raise ValueError('data must contain multiple values')
            self.lambtha = float(sum(data)) / len(data)

    def pmf(self, k):
        """Calculates the value of the PMF"""
        """for a given number of successes"""
        def factorial(n):
            if n == 0:
                return 1
            else:
                return n * factorial(n-1)
        e = 2.7182818285
        k = int(k)
        if k < 0:
            return 0
        else:
            return (self.lambtha ** k) * (e ** -self.lambtha) / factorial(k)

    def cdf(self, k):
        """Calculates the value of the CDF"""
        """for a given number of successes"""
        e = 2.7182818285
        k = int(k)
        if k < 0:
            return 0
        else:
            cdf_value = 0
            for i in range(k+1):
                cdf_value += self.pmf(i)
            return cdf_value
