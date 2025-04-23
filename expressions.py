
import sympy as sp
from dimensions import Constant, Scalar, Dimension

class Expression:
    def __init__(self, terms):
        self.terms = terms
        self.symbol = sp.Mul(*[t.symbol**exp for t, exp in terms])
        self.dimension = Dimension(0, 0, 0, 0)
        for t, exp in terms:
            if isinstance(t, Constant):
                self.dimension += t.dimension * exp

    def is_dimensionless(self):
        return self.dimension.is_dimensionless()

    def evaluate(self, values):
        expr = 1.0
        for term, exp in self.terms:
            key = term.symbol.name
            if key in values:
                expr *= values[key]**exp
        return expr
