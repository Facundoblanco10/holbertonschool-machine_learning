#!/usr/bin/env python3
"""Updates a variable in place using the Adam optimization algorithm"""
import numpy as np


def update_variables_Adam(alpha, beta1, beta2, epsilon, var, grad, v, s, t):
    """
    @alpha is the learning rate
    @beta1 is the weight used for the first moment
    @beta2 is the weight used for the second moment
    @epsilon is a small number to avoid division by zero
    @var is a numpy.ndarray containing the variable to be updated
    @grad is a numpy.ndarray containing the gradient of var
    @v is the previous first moment of var
    @s is the previous second moment of var
    @t is the time step used for bias correction
    Returns: the updated variable, the new first moment, and the new second moment, respectively
    """
    
    # Update biased first moment estimate
    v = beta1 * v + (1 - beta1) * grad
    
    # Update biased second moment estimate
    s = beta2 * s + (1 - beta2) * grad**2
    
    # Correct bias in first moment
    v_corrected = v / (1 - beta1**t)
    
    # Correct bias in second moment
    s_corrected = s / (1 - beta2**t)
    
    # Update variable
    updated_var = var - alpha * v_corrected / (np.sqrt(s_corrected) + epsilon)
    
    return updated_var, v, s
    