
import sympy as sp
from dimensions import Constant, Scalar, Dimension
import math
import random

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

class Expression:
    def __init__(self, terms):
        # Merge like terms by symbol
        term_map = {}
        for t, exp in terms:
            key = t.symbol.name
            if key in term_map:
                term_map[key] = (t, term_map[key][1] + exp)
            else:
                term_map[key] = (t, exp)

        # Remove any terms with exponent 0
        self.terms = [(t, e) for t, e in term_map.values() if e != 0]

        # Symbolic and dimension computation
        self.symbol = sp.Mul(*[t.symbol**exp for t, exp in self.terms])
        self.dimension = Dimension(0, 0, 0, 0)
        for t, exp in self.terms:
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

    def mutate(self, constants, rate=0.3, max_delta=2):
        import random
        new_terms = self.terms.copy()

        # Modify existing exponents
        for i in range(len(new_terms)):
            if random.random() < rate:
                t, e = new_terms[i]
                if random.random() < 0.5:
                    # Apply delta
                    e += random.choice([-1, 1]) * random.randint(1, max_delta)
                else:
                    # Reset exponent randomly
                    e = random.choice([-3, -2, -1, 1, 2, 3])
                new_terms[i] = (t, e)

        # Remove a random term
        if len(new_terms) > 1 and random.random() < rate:
            del new_terms[random.randint(0, len(new_terms) - 1)]

        # Add a random new term
        if random.random() < rate:
            new_const = random.choice(constants)
            new_exp = random.choice([-2, -1, 1, 2])
            new_terms.append((new_const, new_exp))

        return Expression(new_terms)


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
alpha_expr = Expression([
    (e, 2),
    (four_pi, -1),
    (eps0, -1),
    (hbar, -1),
    (c, -1),
])
VALUES.update({
    "ℓ_P": 1.616255e-35,         # Planck length (m)
    "m_P": 2.176434e-8,          # Planck mass (kg)
    "t_P": 5.391247e-44,         # Planck time (s)
    "E_P": 1.956e9 * 1.60218e-19, # Planck energy (J)
    "a₀": 5.29177210903e-11,     # Bohr radius (m)
    "N_A": 6.02214076e23,        # Avogadro
    "R": 8.314462618,            # Gas constant
    "Λ": 1e-52,                  # Approx cosmological constant in 1/m² (or tweak)
    "H₀": 2.2e-18,               # Hubble constant (1/s)
})
VALUES.update({
    "ℓ_P": 1.616255e-35,
    "m_P": 2.176434e-8,
    "t_P": 5.391247e-44,
    "E_P": 1.956e9 * 1.60218e-19,
    "a₀": 5.29177210903e-11,
    "N_A": 6.02214076e23,
    "R": 8.314462618,
    "Λ": 1e-52,
    "H₀": 2.2e-18,
    "μ₀": 1.25663706212e-6,
    "Z₀": 376.730313668,
    "m_e": 9.10938356e-31,
    "m_p": 1.67262192369e-27,
    "m_n": 1.67492749804e-27,
})


# Planck scale constants (dimensionally derived)
l_P = Constant("Planck Length", sp.Symbol("ℓ_P"), ((hbar.dimension + G.dimension + c.dimension * -3) * 0.5))
m_P = Constant("Planck Mass", sp.Symbol("m_P"), ((hbar.dimension + c.dimension + G.dimension * -1) * 0.5))
t_P = Constant("Planck Time", sp.Symbol("t_P"), ((hbar.dimension + G.dimension + c.dimension * -5) * 0.5))
E_P = Constant("Planck Energy", sp.Symbol("E_P"), Dimension(2, -2, 1, 0))  # E = m_P * c^2

# Cosmology
Λ = Constant("Cosmological Constant", sp.Symbol("Λ"), Dimension(0, -2, 0, 0))  # 1 / s²
H_0 = Constant("Hubble Constant", sp.Symbol("H₀"), Dimension(0, -1, 0, 0))     # 1 / s

# Thermodynamics
R = Constant("Gas Constant", sp.Symbol("R"), Dimension(2, -2, 1, 0))           # J / mol·K
NA = Constant("Avogadro Number", sp.Symbol("N_A"), Dimension(0, 0, 0, 0))      # dimensionless

# Electromagnetism
mu_0 = Constant("Vacuum Permeability", sp.Symbol("μ₀"), Dimension(1, -2, 1, -2))
Z_0 = Constant("Free Space Impedance", sp.Symbol("Z₀"), Dimension(1, -2, 1, -2))

# Particle Masses
m_e = Constant("Electron Mass", sp.Symbol("m_e"), Dimension(0, 0, 1, 0))
m_p = Constant("Proton Mass", sp.Symbol("m_p"), Dimension(0, 0, 1, 0))
m_n = Constant("Neutron Mass", sp.Symbol("m_n"), Dimension(0, 0, 1, 0))

# Atomic scale
a_0 = Constant("Bohr Radius", sp.Symbol("a₀"), Dimension(1, 0, 0, 0))  # length

CONSTANTS.extend([
    l_P, m_P, t_P, E_P,
    Λ, H_0,
    R, NA,
    mu_0, Z_0,
    m_e, m_p, m_n,
    a_0,
])




print("Expression:", alpha_expr.symbol)
print("Dimension:", alpha_expr.dimension.vector)
print("Is Dimensionless?", alpha_expr.is_dimensionless())
if alpha_expr.is_dimensionless():
    print("Numeric Value:", alpha_expr.evaluate(VALUES))

def generate_random_expression(constants, max_terms=4):
    terms = []
    for _ in range(random.randint(1, max_terms)):
        const = random.choice(constants)
        exponent = random.choice([-2, -1, 1, 2])
        terms.append((const, exponent))
    return Expression(terms)

def crossover(expr1, expr2):
    half1 = expr1.terms[:len(expr1.terms)//2]
    half2 = expr2.terms[len(expr2.terms)//2:]
    return Expression(half1 + half2)


def is_simple(expression):
    # Simplicity = fewer terms and small exponents
    return 1 / (1 + len(expression.terms) + sum(abs(exp) for _, exp in expression.terms))

def trivial_structure_penalty(expression):
    # Penalize all exponents being 1 or 0
    all_one_or_zero = all(exp in (0, 1) for _, exp in expression.terms)
    return 1 if all_one_or_zero else 0

def proximity_to_one(value):
    """
    Returns a score (0 to 1) for how close a value is to 1.
    1.0 → perfect match
    0.0 → very far off (log-scale tolerant)
    """
    if value <= 0:
        return 0
    return 1 / (1 + abs(math.log10(value)))


TABOO_SET = set()

def to_expr_key(expr):
    return str(sp.simplify(expr.symbol))


def known_identity_penalty(expression):
    return to_expr_key(expression) in TABOO_SET


def diversity_penalty(expr, others):
    return sum(1 for o in others if sp.simplify(expr.symbol - o.symbol) == 0)

def term_count_bonus(expr):
    return 1 if 2 <= len(expr.terms) <= 3 else 0

known_exprs = [
    alpha_expr,                                 # fine-structure constant
    Expression([(NA, -1)]),                     # 1 / N_A
    Expression([(Λ, -1)]),                      # 1 / Λ
    Expression([(H_0, -1)]),                    # 1 / H_0
    Expression([(t_P, -2)]),                    # 1 / t_P²
    Expression([(a_0, -1)]),                    # 1 / Bohr radius
    Expression([(l_P, -1)]),                    # 1 / Planck length
    Expression([(k_B, -1)]),                    # 1 / Boltzmann constant
    Expression([(E_P, -1)]),                    # 1 / Planck energy
    Expression([(R, -1)]),                      # 1 / gas constant
    Expression([(m_P, -1)]),                    # 1 / Planck mass
    Expression([(m_e, -1)]),                    # 1 / electron mass
    Expression([(m_n, -1)]),                    # 1 / neutron mass
    Expression([(m_p, -1)]),                    # 1 / proton mass
    Expression([(mu_0, -1)]),                   # 1 / vacuum permeability
    Expression([(Z_0, -1)]),                    # 1 / free space impedance
    Expression([(c, -1)]),                      # 1 / speed of light
    Expression([(hbar, -1)]),                   # 1 / reduced Planck constant
    Expression([(eps0, -1)]),                   # 1 / vacuum permittivity

    # Composite expressions known to evaluate to ~1
    Expression([(NA, 1), (k_B, 1), (R, -1)]),    # N_A * k_B / R
]

def single_term_penalty(expr):
    return 1 if len(expr.terms) == 1 else 0


def fitness(expression, *, known_exprs=None, alpha=10, beta=1, delta=5, epsilon=100, zeta=1, theta=10, eta=5):
    """
    Fitness breakdown:
      + alpha: bonus for dimensionless
      + beta: simplicity score
      + zeta: reward 2–3 term expressions
      - delta: penalize trivial all-1/0 exponents
      - epsilon: penalize known expressions (taboo set)
      - theta: penalize 1-term expressions
    """
    known_exprs = known_exprs or []
    f = 0
    if expression.is_dimensionless():
        f += alpha
    try:
        value = expression.evaluate(VALUES)
        f += eta * proximity_to_one(value)
    except KeyError:
        pass  # skip if some constant is missing from VALUES
    f += beta * is_simple(expression)
    f += zeta * term_count_bonus(expression)
    f -= delta * trivial_structure_penalty(expression)
    f -= theta if len(expression.terms) == 1 else 0
    if known_identity_penalty(expression):
        f -= epsilon
    return f




def evolve(constants, *, known_exprs=None, generations=200, population_size=100, elite_fraction=0.2):
    known_exprs = known_exprs or []
    global TABOO_SET
    TABOO_SET = set(to_expr_key(e) for e in known_exprs)

    population = [generate_random_expression(constants) for _ in range(population_size)]
    best_overall = None
    best_fitness = float('-inf')

    for gen in range(generations):
        scored = [(expr, fitness(expr, known_exprs=known_exprs)) for expr in population]
        scored.sort(key=lambda x: -x[1])

        if not scored:
            print(f"Generation {gen+1} collapsed: no valid expressions.")
            break

        best_expr = scored[0][0]
        best_score = scored[0][1]
        print(f"Generation {gen+1}: Best Fitness = {best_score} Expression = {best_expr.symbol}")

        if best_score > best_fitness:
            best_fitness = best_score
            best_overall = best_expr

        TABOO_SET.add(to_expr_key(best_expr))

        # Elitism
        elite_count = max(1, int(population_size * elite_fraction))
        survivors = [expr for expr, _ in scored[:elite_count]]

        # Breed new population
        new_population = survivors[:]
        while len(new_population) < population_size:
            if random.random() < 0.5:
                parent = random.choice(survivors)
                child = parent.mutate(constants)
            else:
                p1, p2 = random.sample(survivors, 2)
                child = crossover(p1, p2).mutate(constants)

            if to_expr_key(child) in TABOO_SET:
                continue

            new_population.append(child)

        population = new_population

    if best_overall is None:
        raise RuntimeError("No valid expression was found.")

    return best_overall



# Example Usage:
best_expr = evolve(CONSTANTS, known_exprs=[alpha_expr, Expression([(NA, -1)])])
try:
    print("Best Found Expression:", best_expr.symbol)
    print("Dimension:", best_expr.dimension.vector)
    print("Is Dimensionless?", best_expr.is_dimensionless())
    if best_expr.is_dimensionless():
        print("Value:", best_expr.evaluate(VALUES))
except AttributeError:
    print("No valid expression found.")
val = best_expr.evaluate(VALUES)
prox = proximity_to_one(val)
print(f"Value: {val}")
print(f"Proximity to 1.0: {prox:.6f}")
