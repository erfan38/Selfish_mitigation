import numpy as np
from sympy import symbols, diff, solve
from scipy.optimize import minimize

# Define the variables
w, y, z, λ = symbols('w y z λ')
# Define the constants
C, r, rep = symbols('C r rep')

# Define the objective function to maximize w and minimize y
objective_function = w - y

# Define the constraints
constraint1 = z - (r / rep - 1) * λ
constraint2 = y - ((2 * (r / rep) * (rep - C) - (r / rep)**2) / rep**2) * λ
constraint3 = w - ((2 * (r / rep) * (rep - C) - (r / rep)**2) / (rep**2 - C * (r / rep))) * λ
constraint4 = w + y + z - 1

# Solve symbolically
solutions_symbolic = solve((diff(objective_function, w) - λ * diff(constraint4, w),
                           diff(objective_function, y) - λ * diff(constraint4, y),
                           diff(objective_function, z) - λ * diff(constraint4, z),
                           constraint1,
                           constraint2,
                           constraint3,
                           constraint4), (w, y, z, λ))

# Print symbolic solutions
print("Symbolic solutions:")
for solution in solutions_symbolic:
    print("w =", solution[0])
    print("y =", solution[1])
    print("z =", solution[2])
    print()  # Adding an empty line between solutions

# Define the Lagrangian
def lagrangian(wyz, lam):
    w, y, z = wyz[:3]
    return -w + lam * (w + y + z - 1)

# Define the gradient of the Lagrangian
def lagrangian_gradient(wyz, lam):
    w, y, z = wyz[:3]
    grad = np.zeros(4)
    grad[0] = -1 - lam
    grad[1] = -lam
    grad[2] = -lam
    grad[3] = w + y + z - 1
    return grad

# Define the constraint function
def constraint(wyz):
    w, y, z = wyz[:3]
    return w + y + z - 1

# Initial guess
w_init = 0.3  # Adjusted initial guess for w
y_init = 0.3  # Adjusted initial guess for y
z_init = 0.3  # Adjusted initial guess for z
lam_init = 0.0

# Additional constraints
bounds = [(0, None), (0, None), (0, None), (None, None)]  # Bounds for w, y, z, λ

# Solve numerically
result = minimize(lagrangian, [w_init, y_init, z_init, lam_init], args=(lam_init,), jac=lagrangian_gradient, constraints={'type': 'eq', 'fun': constraint}, bounds=bounds)

# Extracting the optimal values
w_opt, y_opt, z_opt = result.x[:3]

# Check if the sum of w, y, z is approximately 1
sum_constraint = abs(w_opt + y_opt + z_opt - 1) < 1e-6

# Print numerical solution
print("\nNumerical solution:")
print("Optimal w:", w_opt)
print("Optimal y:", y_opt)
print("Optimal z:", z_opt)
print("Sum constraint satisfied:", sum_constraint)
