
from .dimensions import Dimension, Constant, Scalar
import sympy as sp
import math

c = Constant("Speed of Light", sp.Symbol("c"), Dimension(1, -1, 0, 0))
hbar = Constant("Reduced Planck Constant", sp.Symbol("ħ"), Dimension(2, -1, 1, 0))
G = Constant("Gravitational Constant", sp.Symbol("G"), Dimension(3, -2, -1, 0))
k_B = Constant("Boltzmann Constant", sp.Symbol("k_B"), Dimension(2, -2, 1, 0))
eps0 = Constant("Vacuum Permittivity", sp.Symbol("ε₀"), Dimension(-3, 4, -1, 2))

e_dim = eps0.dimension + hbar.dimension + c.dimension
e = Constant("Elementary Charge", sp.Symbol("e"), Dimension(*[v // 2 for v in e_dim.vector]))

four_pi = Scalar("4π", 4 * math.pi)

CONSTANTS = [c, hbar, G, k_B, eps0, e]
SCALARS = [four_pi]

VALUES = {
    "c": 299792458,
    "ħ": 1.054571817e-34,
    "G": 6.67430e-11,
    "k_B": 1.380649e-23,
    "ε₀": 8.8541878128e-12,
    "e": 1.602176634e-19,
    "4π": 4 * math.pi,
}
