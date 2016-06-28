# Option Pricing Model

This code compares two methods for computing the value of arithmetic asian options.

The fist uses the Monte Carlo simulation. 

The second is Kemna Vorst method (control variable). We write E(X) = E(X-Y) + E(Y), X being the arithmetic option and Y being the geometric option. The geometric option is computed thanks to an exact formula and the difference is computed using Monte Carlo. The convergence is much faster using this method.