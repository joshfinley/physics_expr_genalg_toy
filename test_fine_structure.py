
from constants import e, eps0, hbar, c, four_pi, VALUES
from expressions import Expression

alpha_expr = Expression([
    (e, 2),
    (four_pi, -1),
    (eps0, -1),
    (hbar, -1),
    (c, -1),
])

print("Expression:", alpha_expr.symbol)
print("Dimension:", alpha_expr.dimension.vector)
print("Is Dimensionless?", alpha_expr.is_dimensionless())
if alpha_expr.is_dimensionless():
    print("Numeric Value:", alpha_expr.evaluate(VALUES))
