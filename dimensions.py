
class Dimension:
    def __init__(self, L=0, T=0, M=0, Q=0):
        self.vector = (L, T, M, Q)

    def __add__(self, other):
        return Dimension(*[a + b for a, b in zip(self.vector, other.vector)])

    def __mul__(self, scalar: int):
        return Dimension(*[a * scalar for a in self.vector])

    def is_dimensionless(self):
        return all(x == 0 for x in self.vector)

class Constant:
    def __init__(self, name, symbol, dimension):
        self.name = name
        self.symbol = symbol
        self.dimension = dimension

class Scalar:
    def __init__(self, name, value):
        import sympy as sp
        self.name = name
        self.symbol = sp.Symbol(name)
        self.value = value
