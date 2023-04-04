#!/usr/bin/env python3
"""class Poisson that represents a poisson distribution"""


class Poisson():
    """Class Poisson"""

    def __init__(self, data=None, lambtha=1.):
        self.lambtha = float(lambtha)
        if data is None:
            if lambtha <= 0:
                raise ValueError('lambtha must be a positive value')
            self.data = lambtha
        else:
            if type(data) is not list:
                raise TypeError('data must be a list')
            if len(data) <= 1:
                raise ValueError('data must contain multiple values')
            self.lambtha = float(sum(data)) / len(data)