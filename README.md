# Option Pricing Model

This code compares three methods for computing the value of arithmetic asian options.

The fist approximates the asian option using a lognormal distribution and then computes the value using Monte Carlo method. 

The second prices the arithmetic option using the Monte Carlo simulation. 

The third is Kemna Vorst method (control variable). We write E(X) = E(X-Y) + E(Y), X being the arithmetic option and Y being the geometric option. The geometric option is computed thanks to an exact formula and the difference is computed using Monte Carlo. The convergence is much faster using this method.
