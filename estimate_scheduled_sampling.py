#!/usr/bin/python

import numpy as np
from scipy.special import lambertw
import argparse

"""

A script for estimating the parameters of scheduled sampling. Based on the
threshold value and number of step you want to achieve the value, it computes
the coefficient of the inverse sigmoid decay function.

"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Estimates parameter for scheduled sampling.")
    parser.add_argument("--value", type=float, required=True, help="The value the threshold should achieve.")
    parser.add_argument("--step", type=int, required=True, help="Step when you want to achieve the value.")
    args = parser.parse_args()

    x = args.step
    c = args.value

    coeff = c * np.exp(lambertw((1 - c) / c * x)) / (1 - c)

    print coeff.real
