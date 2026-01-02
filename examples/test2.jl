# File: test/test_optimization_debug.jl
"""
Debug version of Test 3 with detailed output.
"""

using LinearAlgebra
using Random

include("../src/basis_functions.jl")
include("../src/data_processing.jl")
include("../src/generator_construction.jl")
include("../src/optimization.jl")

Random.seed!(42)

println("="^70)
println("DEBUG: Gradient Finite Difference Test")
println("="^70)

# Create small test problem
state_space = [[0,0], [1,0], [0,1], [1,1], [2,1]]
stoich_basis = [[1,0], [-1,1], [1,-1]]
dt = 0.5
basis = PolynomialBasis{2, 1}()  # Linear: 3 features

data = build_inverse_problem_data(state_space, stoich_basis, dt, basis=basis)

println("\nProblem setup:")
println("  State space size: $(length(state_space))")
println("  Reactions: $(length(stoich_basis))")
println("  Features: $(get_n_features(basis))")
println("  Parameters: $(length(stoich_basis) * get_n_features(basis))")
println("  dt: $dt")

# Create synthetic window
p_curr = [0.4, 0.3, 0.2, 0.08, 0.02]
p_next = [0.3, 0.35, 0.25, 0.08, 0.02]
window = TimeWindowData(p_curr, p_next, state_space)

println("\nWindow data:")
println("  ||p_curr||₁: $(sum(p_curr))")
println("  ||p_next||₁: $(sum(p_next))")
println("  ||p_next - p_curr||₁: $(sum(abs.(p_next - p_curr)))")

# Random parameters
n_params = length(stoich_basis) * get_n_features(basis)
θ = 0.1 * randn(n_params)

println("\nInitial parameters:")
println("  θ: $(round.(θ, digits=3))")

# Build generator
A = build_generator(θ, data)
println("\nGenerator:")
println("  ||A||_F: $(round(norm(A), digits=4))")
println("  max|A|: $(round(maximum(abs.(A)), digits=4))")

# Compute objective at θ
λ1 = 1e-4
obj_0, grad_0 = compute_objective_gradient_adjoint(θ, data, [window], λ1=λ1)

println("\nObjective at θ:")
println("  J(θ): $(round(obj_0, digits=6))")
println("  ||∇J(θ)||: $(round(norm(grad_0), digits=6))")

# Break down objective
P = exp(Matrix(A * dt))
q = P * p_curr
r = p_next - q
pred_error = sum(abs.(r))
frob_term = λ1 * sum(abs2, A)

println("\nObjective breakdown:")
println("  Prediction term: $(round(pred_error, digits=6))")
println("  Frobenius term:  $(round(frob_term, digits=6))")
println("  Total:           $(round(pred_error + frob_term, digits=6))")
println("  Recorded:        $(round(obj_0, digits=6))")

# Test finite difference on first parameter
println("\n" * "="^70)
println("Finite Difference Test (parameter 1)")
println("="^70)

ε_values = [1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8]

for ε in ε_values
    # Forward difference
    θ_plus = copy(θ)
    θ_plus[1] += ε
    
    obj_plus, _ = compute_objective_gradient_adjoint(θ_plus, data, [window], λ1=λ1)
    
    grad_fd_forward = (obj_plus - obj_0) / ε
    
    # Central difference
    θ_minus = copy(θ)
    θ_minus[1] -= ε
    
    obj_minus, _ = compute_objective_gradient_adjoint(θ_minus, data, [window], λ1=λ1)
    
    grad_fd_central = (obj_plus - obj_minus) / (2ε)
    
    # Analytical
    grad_analytical = grad_0[1]
    
    error_forward = abs(grad_fd_forward - grad_analytical) / (abs(grad_analytical) + 1e-10)
    error_central = abs(grad_fd_central - grad_analytical) / (abs(grad_analytical) + 1e-10)
    
    println("\nε = $ε:")
    println("  J(θ-ε): $(round(obj_minus, digits=8))")
    println("  J(θ):   $(round(obj_0, digits=8))")
    println("  J(θ+ε): $(round(obj_plus, digits=8))")
    println("  ΔJ:     $(round(obj_plus - obj_0, sigdigits=4))")
    println()
    println("  FD (forward):  $(round(grad_fd_forward, sigdigits=6))")
    println("  FD (central):  $(round(grad_fd_central, sigdigits=6))")
    println("  Analytical:    $(round(grad_analytical, sigdigits=6))")
    println()
    println("  Error (forward): $(round(error_forward, sigdigits=3))")
    println("  Error (central): $(round(error_central, sigdigits=3))")
end

# Now test all parameters with optimal ε
println("\n" * "="^70)
println("All Parameters (ε = 1e-5)")
println("="^70)

ε = 1e-5
grad_fd = zeros(n_params)

for i in 1:n_params
    θ_plus = copy(θ)
    θ_plus[i] += ε
    
    θ_minus = copy(θ)
    θ_minus[i] -= ε
    
    obj_plus, _ = compute_objective_gradient_adjoint(θ_plus, data, [window], λ1=λ1)
    obj_minus, _ = compute_objective_gradient_adjoint(θ_minus, data, [window], λ1=λ1)
    
    grad_fd[i] = (obj_plus - obj_minus) / (2ε)
end

println("\nGradient comparison:")
println("  ||grad_analytical||: $(round(norm(grad_0), digits=6))")
println("  ||grad_fd||:         $(round(norm(grad_fd), digits=6))")
println("  Relative error:      $(round(norm(grad_0 - grad_fd) / norm(grad_fd), sigdigits=3))")

println("\nComponent-wise:")
for i in 1:n_params
    error_rel = abs(grad_0[i] - grad_fd[i]) / (abs(grad_fd[i]) + 1e-10)
    println("  θ[$i]: analytical=$(rpad(round(grad_0[i], sigdigits=4), 8)) "*
           "fd=$(rpad(round(grad_fd[i], sigdigits=4), 8)) "*
           "error=$(round(error_rel, sigdigits=2))")
end

# Final diagnosis
println("\n" * "="^70)
println("DIAGNOSIS")
println("="^70)

if norm(grad_fd) < 1e-6
    println("⚠️  ISSUE: Finite difference gradient is essentially zero!")
    println("   This means the objective barely changes when perturbing θ.")
    println("   Possible causes:")
    println("   1. The problem is degenerate (many θ give same J)")
    println("   2. The perturbations E_kf are incorrect")
    println("   3. The generator construction has a bug")
    
    # Check if perturbation actually changes A
    println("\n   Testing if perturbations change A:")
    for i in 1:min(3, n_params)
        θ_pert = copy(θ)
        θ_pert[i] += 0.1
        
        A_pert = build_generator(θ_pert, data)
        
        println("     θ[$i] += 0.1: ||ΔA|| = $(round(norm(A_pert - A), digits=6))")
    end
else
    println("✅ Finite difference gradient is non-zero")
    println("   The issue is likely numerical precision in the original test.")
end
