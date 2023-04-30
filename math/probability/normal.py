#!/usr/bin/env python3
"""Normal distribution class"""


def erf(x):
    """function that computes the error function"""
    pi = 3.1415926536

    a = 2 / (pi ** 0.5)
    x_3 = (x ** 3) / 3
    x_5 = (x ** 5) / 10
    x_7 = (x ** 7) / 42
    x_9 = (x ** 9) / 216
    return (a * (x - x_3 + x_5 - x_7 + x_9))


class Normal():
    """Normal distribution"""
    def __init__(self, data=None, mean=0., stddev=1.):
        if data is None:
            if stddev <= 0:
                raise ValueError('stddev must be a positive value')
            else:
                self.stddev = float(stddev)
            self.mean = float(mean)
        else:
            if type(data) != list:
                raise TypeError('data must be a list')
            if len(data) <= 1:
                raise ValueError('data must contain multiple values')
            self.mean = float(sum(data) / len(data))
            s = sum(map(lambda x: (x - self.mean) ** 2, data))
            self.stddev = (s / len(data)) ** 0.5

    def z_score(self, x):
        """Calculates the z-score of a given x-value"""
        return (x - self.mean) / self.stddev

    def x_value(self, z):
        """Calculates the x-value of a given z-score"""
        return (z * self.stddev) + self.mean

    def pdf(self, x):
        """Calculates the value of the PDF for a given x-value"""
        pi = 3.1415926536
        e = 2.7182818285
        return (1 / (self.stddev * ((2 * pi) ** 0.5))) *\
               (e ** ((self.z_score(x) ** 2) / -2))

    def cdf(self, x):
        """Calculates the value of the CDF for a given x-value"""
        return (0.5 * (1 + erf((x - self.mean) / (self.stddev * (2 ** 0.5)))))
