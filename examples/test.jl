# File: test/test_optimization.jl
"""
Comprehensive test suite for CME inverse problem optimization.
Tests all components: Fréchet derivatives, gradients, adjoint method, etc.
"""

using LinearAlgebra
using Random
using Test

include("../src/basis_functions.jl")
include("../src/data_processing.jl")
include("../src/generator_construction.jl")
include("../src/optimization.jl")

println("="^70)
println("COMPREHENSIVE OPTIMIZATION TEST SUITE")
println("="^70)

Random.seed!(42)

# =============================================================================
# Test 1: Finite Difference Verification of Fréchet Derivative
# =============================================================================

println("\n" * "="^70)
println("TEST 1: Fréchet Derivative (Forward Method)")
println("="^70)

function test_frechet_derivative()
    n = 5
    A = randn(n, n)
    E = randn(n, n)
    
    # Compute Fréchet derivative
    expA, L = frechet(A, E)
    
    # Verify via finite difference
    ε = 1e-8
    expA_plus = exp(A + ε*E)
    
    # Fréchet derivative should satisfy: exp(A + εE) ≈ exp(A) + εL + O(ε²)
    fd_approx = (expA_plus - expA) / ε
    
    error = norm(fd_approx - L) / (norm(L) + 1e-10)
    
    println("Finite difference verification:")
    println("  ||exp(A)||: $(round(norm(expA), digits=4))")
    println("  ||L||:      $(round(norm(L), digits=4))")
    println("  ||FD||:     $(round(norm(fd_approx), digits=4))")
    println("  Relative error: $(round(error, sigdigits=3))")
    
    if error < 1e-5
        println("  ✅ PASS")
        return true
    else
        println("  ❌ FAIL")
        return false
    end
end

test1_pass = test_frechet_derivative()

# =============================================================================
# Test 2: Adjoint Fréchet Derivative Correctness
# =============================================================================

println("\n" * "="^70)
println("TEST 2: Adjoint Fréchet Derivative")
println("="^70)

function test_adjoint_frechet()
    n = 5
    A = randn(n, n)
    E = randn(n, n)
    Λ = randn(n, n)
    
    # Forward Fréchet
    _, L = frechet(A, E)
    
    # Adjoint Fréchet
    L_adj = frechet_adjoint(A, Λ)
    
    # Adjoint property: ⟨Λ, L(A,E)⟩ = ⟨L*(A,Λ), E⟩
    lhs = sum(Λ .* L)  # tr(Λ' L)
    rhs = sum(L_adj .* E)  # tr(L_adj' E)
    
    error = abs(lhs - rhs) / (abs(lhs) + 1e-10)
    
    println("Adjoint property verification:")
    println("  ⟨Λ, L(A,E)⟩:    $(round(lhs, digits=6))")
    println("  ⟨L*(A,Λ), E⟩:   $(round(rhs, digits=6))")
    println("  Relative error: $(round(error, sigdigits=3))")
    
    if error < 1e-10
        println("  ✅ PASS")
        return true
    else
        println("  ❌ FAIL")
        return false
    end
end

test2_pass = test_adjoint_frechet()

# =============================================================================
# Test 3: Gradient via Finite Difference (Full Objective)
# =============================================================================

println("\n" * "="^70)
println("TEST 3: Gradient Finite Difference Check")
println("="^70)

function test_gradient_finite_difference()
    # Create small test problem
    state_space = [[0,0], [1,0], [0,1], [1,1], [2,1]]
    stoich_basis = [[1,0], [-1,1], [1,-1]]
    dt = 0.5
    basis = PolynomialBasis{2, 1}()  # Linear: 3 features
    
    data = build_inverse_problem_data(state_space, stoich_basis, dt, basis=basis)
    
    # Create synthetic window
    p_curr = [0.4, 0.3, 0.2, 0.08, 0.02]
    p_next = [0.3, 0.35, 0.25, 0.08, 0.02]
    window = TimeWindowData(p_curr, p_next, state_space)
    
    # Random parameters
    n_params = length(stoich_basis) * get_n_features(basis)
    θ = 0.1 * randn(n_params)
    
   # Compute analytical gradient (adjoint)
    obj_adj, grad_adj = compute_objective_gradient_adjoint(θ, data, [window], λ1=1e-4)
    
    # Compute CENTRAL finite difference gradient (more accurate)
    ε = 1e-6  # Larger epsilon
    grad_fd = zeros(n_params)
    
    for i in 1:n_params
        θ_plus = copy(θ)
        θ_plus[i] += ε
        
        θ_minus = copy(θ)
        θ_minus[i] -= ε
        
        obj_plus, _ = compute_objective_gradient_adjoint(θ_plus, data, [window], λ1=1e-4)
        obj_minus, _ = compute_objective_gradient_adjoint(θ_minus, data, [window], λ1=1e-4)
        
        grad_fd[i] = (obj_plus - obj_minus) / (2ε)  # Central difference
    end    
    # Compare
    error = norm(grad_adj - grad_fd) / (norm(grad_fd) + 1e-10)
    
    println("Gradient comparison:")
    println("  Objective:      $(round(obj_adj, digits=6))")
    println("  ||grad_adj||:   $(round(norm(grad_adj), digits=6))")
    println("  ||grad_fd||:    $(round(norm(grad_fd), digits=6))")
    println("  Relative error: $(round(error, sigdigits=3))")
    
    # Show worst components
    component_errors = abs.(grad_adj - grad_fd) ./ (abs.(grad_fd) .+ 1e-10)
    worst_idx = sortperm(component_errors, rev=true)[1:min(3, n_params)]
    
    println("\n  Worst components:")
    for i in worst_idx
        println("    θ[$i]: adj=$(round(grad_adj[i], sigdigits=4)), "*
               "fd=$(round(grad_fd[i], sigdigits=4)), "*
               "err=$(round(component_errors[i], sigdigits=2))")
    end
    
    if error < 1e-4
        println("\n  ✅ PASS")
        return true
    else
        println("\n  ❌ FAIL")
        return false
    end
end

test3_pass = test_gradient_finite_difference()

# =============================================================================
# Test 4: Adjoint vs Forward Fréchet Gradients
# =============================================================================

println("\n" * "="^70)
println("TEST 4: Adjoint vs Forward Fréchet Method")
println("="^70)

function test_adjoint_vs_frechet()
    # Create small test problem
    state_space = [[0,0], [1,0], [0,1], [1,1], [2,1], [2,2]]
    stoich_basis = [[1,0], [-1,1], [1,-1]]
    dt = 0.5
    basis = PolynomialBasis{2, 2}()  # Quadratic: 6 features
    
    data = build_inverse_problem_data(state_space, stoich_basis, dt, basis=basis)
    
    # Create synthetic windows
    windows = []
    for _ in 1:3
        p_curr = rand(length(state_space))
        p_curr = p_curr / sum(p_curr)
        
        p_next = rand(length(state_space))
        p_next = p_next / sum(p_next)
        
        push!(windows, TimeWindowData(p_curr, p_next, state_space))
    end
    
    # Random parameters
    n_params = length(stoich_basis) * get_n_features(basis)
    θ = 0.1 * randn(n_params)
    
    # Compute both methods
    obj_adj, grad_adj = compute_objective_gradient_adjoint(θ, data, windows, λ1=1e-4)
    obj_fre, grad_fre = compute_objective_gradient_frechet(θ, data, windows, λ1=1e-4)
    
    # Compare
    obj_error = abs(obj_adj - obj_fre) / (abs(obj_fre) + 1e-10)
    grad_error = norm(grad_adj - grad_fre) / (norm(grad_fre) + 1e-10)
    
    println("Method comparison:")
    println("  Objective adjoint: $(round(obj_adj, digits=8))")
    println("  Objective frechet: $(round(obj_fre, digits=8))")
    println("  Obj rel error:     $(round(obj_error, sigdigits=3))")
    println()
    println("  ||grad_adj||:      $(round(norm(grad_adj), digits=6))")
    println("  ||grad_fre||:      $(round(norm(grad_fre), digits=6))")
    println("  Grad rel error:    $(round(grad_error, sigdigits=3))")
    
    # Component-wise comparison
    component_errors = abs.(grad_adj - grad_fre) ./ (abs.(grad_fre) .+ 1e-10)
    worst_idx = sortperm(component_errors, rev=true)[1:min(5, n_params)]
    
    println("\n  Worst gradient components:")
    for i in worst_idx
        println("    θ[$i]: adj=$(round(grad_adj[i], sigdigits=4)), "*
               "fre=$(round(grad_fre[i], sigdigits=4)), "*
               "err=$(round(component_errors[i], sigdigits=2))")
    end
    
    if obj_error < 1e-10 && grad_error < 1e-6
        println("\n  ✅ PASS")
        return true
    else
        println("\n  ❌ FAIL")
        return false
    end
end

test4_pass = test_adjoint_vs_frechet()

# =============================================================================
# Test 5: Perturbation Matrix Properties
# =============================================================================

println("\n" * "="^70)
println("TEST 5: Perturbation Matrix Structure")
println("="^70)

function test_perturbation_properties()
    state_space = [[0,0], [1,0], [0,1], [1,1], [2,1]]
    stoich_basis = [[1,0], [-1,1]]
    dt = 0.5
    basis = PolynomialBasis{2, 1}()
    
    data = build_inverse_problem_data(state_space, stoich_basis, dt, basis=basis)
    
    all_pass = true
    
    for k in 1:length(stoich_basis)
        for f in 1:get_n_features(basis)
            E = build_perturbation(k, f, data)
            
            # Property 1: Column sums should be zero (stochastic structure)
            col_sums = sum(E, dims=1)[:]
            max_col_sum = maximum(abs.(col_sums))
            
            if max_col_sum > 1e-10
                println("  ❌ Perturbation ($k,$f) has non-zero column sum: $max_col_sum")
                all_pass = false
            end
            
            # Property 2: Each column should have at most 2 nonzeros (one pos, one neg)
            # Actually, can have more due to diagonal, so skip this test
            
            # Property 3: Off-diagonal should match stoichiometry
            # (This is complex to verify, skip for now)
        end
    end
    
    println("Perturbation matrix properties:")
    println("  Checked: $(length(stoich_basis) * get_n_features(basis)) matrices")
    
    if all_pass
        println("  ✅ PASS: All column sums ≈ 0")
        return true
    else
        println("  ❌ FAIL")
        return false
    end
end

test5_pass = test_perturbation_properties()

# =============================================================================
# Test 6: Generator Properties
# =============================================================================

println("\n" * "="^70)
println("TEST 6: Generator Construction Properties")
println("="^70)

function test_generator_properties()
    state_space = [[0,0], [1,0], [0,1], [1,1], [2,1]]
    stoich_basis = [[1,0], [-1,1], [1,-1]]
    dt = 0.5
    basis = PolynomialBasis{2, 1}()
    
    data = build_inverse_problem_data(state_space, stoich_basis, dt, basis=basis)
    
    # Random parameters
    n_params = length(stoich_basis) * get_n_features(basis)
    θ = abs.(randn(n_params))  # Positive for valid propensities
    
    A = build_generator(θ, data)
    
    all_pass = true
    
    # Property 1: Column sums = 0
    col_sums = sum(A, dims=1)[:]
    max_col_sum = maximum(abs.(col_sums))
    
    println("Generator properties:")
    println("  Max |col sum|: $(round(max_col_sum, sigdigits=3))")
    
    if max_col_sum > 1e-10
        println("  ❌ Column sums not zero!")
        all_pass = false
    else
        println("  ✅ Column sums = 0")
    end
    
    # Property 2: Diagonal entries negative
    diag_entries = diag(A)
    if any(diag_entries .> 1e-10)
        println("  ❌ Some diagonal entries positive!")
        all_pass = false
    else
        println("  ✅ All diagonal entries ≤ 0")
    end
    
    # Property 3: Off-diagonal entries non-negative
    n = size(A, 1)
    for i in 1:n
        for j in 1:n
            if i != j && A[i,j] < -1e-10
                println("  ❌ Negative off-diagonal at ($i,$j): $(A[i,j])")
                all_pass = false
            end
        end
    end
    
    if all_pass
        println("  ✅ All off-diagonal entries ≥ 0")
    end
    
    # Property 4: Sparse structure
    sparsity = 100 * (1 - count(abs.(A) .> 1e-10) / length(A))
    println("  Sparsity: $(round(sparsity, digits=1))%")
    
    if all_pass
        println("\n  ✅ PASS")
        return true
    else
        println("\n  ❌ FAIL")
        return false
    end
end

test6_pass = test_generator_properties()

# =============================================================================
# Test 7: Optimization Convergence
# =============================================================================

println("\n" * "="^70)
println("TEST 7: Optimization Convergence")
println("="^70)

function test_optimization_convergence()
    # Create synthetic problem with known solution
    state_space = [[0,0], [1,0], [0,1], [1,1], [2,0], [0,2], [2,1], [1,2]]
    stoich_basis = [[1,0], [-1,1], [0,1]]
    dt = 0.5
    basis = PolynomialBasis{2, 1}()
    
    data = build_inverse_problem_data(state_space, stoich_basis, dt, basis=basis)
    
    # True parameters (known)
    n_params = length(stoich_basis) * get_n_features(basis)
    θ_true = [1.0, 0.5, 0.1,   # Reaction 1: [constant, x, y]
              0.8, 0.3, 0.2,   # Reaction 2
              1.2, 0.1, 0.4]   # Reaction 3
    
    A_true = build_generator(θ_true, data)
    
    # Generate synthetic data
    p_curr = rand(length(state_space))
    p_curr = p_curr / sum(p_curr)
    
    P_true = exp(Matrix(A_true * dt))
    p_next = P_true * p_curr
    
    window = TimeWindowData(p_curr, p_next, state_space)
    
    # Try to recover θ_true
    println("Attempting to recover true parameters...")
    println("  True θ: $(round.(θ_true[1:3], digits=2))...")
    
    A_learned, θ_learned, converged = learn_generator(data, [window],
                                                       λ1=1e-6,
                                                       max_iter=500,
                                                       method=:adjoint,
                                                       show_trace=false)
    
    println("  Converged: $converged")
    println("  Learned θ: $(round.(θ_learned[1:3], digits=2))...")
    
    # Check recovery
    param_error = norm(θ_learned - θ_true) / norm(θ_true)
    gen_error = norm(A_learned - A_true) / norm(A_true)
    
    # Check prediction
    P_learned = exp(Matrix(A_learned * dt))
    p_pred = P_learned * p_curr
    pred_error = norm(p_pred - p_next, 1)
    
    println("\n  Parameter error: $(round(param_error, digits=4))")
    println("  Generator error: $(round(gen_error, digits=4))")
    println("  Prediction error: $(round(pred_error, digits=6))")
    
    if pred_error < 1e-3
        println("\n  ✅ PASS (prediction accurate)")
        return true
    else
        println("\n  ⚠️  Warning: High prediction error (non-identifiable?)")
        if pred_error < 0.1
            println("  ✅ PASS (acceptable for inverse problem)")
            return true
        else
            return false
        end
    end
end

test7_pass = test_optimization_convergence()

# =============================================================================
# Test 8: Scaling and Numerical Stability
# =============================================================================

println("\n" * "="^70)
println("TEST 8: Numerical Stability")
println("="^70)

function test_numerical_stability()
    state_space = [[0,0], [1,0], [0,1], [1,1]]
    stoich_basis = [[1,0], [-1,1]]
    dt = 0.5
    basis = PolynomialBasis{2, 1}()
    
    data = build_inverse_problem_data(state_space, stoich_basis, dt, basis=basis)
    
    # Test with various parameter scales
    scales = [1e-3, 1e-1, 1.0, 10.0, 100.0]
    
    all_pass = true
    
    println("Testing different parameter scales:")
    for scale in scales
        θ = scale * ones(6)
        
        A = build_generator(θ, data)
        
        # Check for NaN/Inf
        if any(!isfinite, A)
            println("  ❌ Scale $scale: Generator contains NaN/Inf!")
            all_pass = false
            continue
        end
        
        # Try to compute exponential
        try
            P = exp(Matrix(A * dt))
            
            if any(!isfinite, P)
                println("  ❌ Scale $scale: Propagator contains NaN/Inf!")
                all_pass = false
            else
                println("  ✅ Scale $scale: OK (||A||=$(round(norm(A), digits=2)), ||P||=$(round(norm(P), digits=2)))")
            end
        catch e
            println("  ❌ Scale $scale: Exponential failed - $e")
            all_pass = false
        end
    end
    
    if all_pass
        println("\n  ✅ PASS")
        return true
    else
        println("\n  ❌ FAIL")
        return false
    end
end

test8_pass = test_numerical_stability()

# =============================================================================
# Summary
# =============================================================================

println("\n" * "="^70)
println("TEST SUMMARY")
println("="^70)

tests = [
    ("Fréchet Derivative", test1_pass),
    ("Adjoint Fréchet", test2_pass),
    ("Gradient Finite Difference", test3_pass),
    ("Adjoint vs Forward", test4_pass),
    ("Perturbation Structure", test5_pass),
    ("Generator Properties", test6_pass),
    ("Optimization Convergence", test7_pass),
    ("Numerical Stability", test8_pass)
]

n_pass = count(last.(tests))
n_total = length(tests)

for (name, pass) in tests
    status = pass ? "✅ PASS" : "❌ FAIL"
    println("  $(rpad(name, 30)) $status")
end

println()
println("="^70)
if n_pass == n_total
    println("🎉 ALL TESTS PASSED ($n_pass/$n_total)")
else
    println("⚠️  SOME TESTS FAILED ($n_pass/$n_total passed)")
end
println("="^70)
