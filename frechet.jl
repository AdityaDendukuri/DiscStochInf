using LinearAlgebra
using Printf
using Test
using Optim
using LineSearches

"""
Compute Fréchet derivative of exp(A*δt) in direction H using integral formula:
D(exp(A*δt))[H] = ∫₀^δt exp(A*s) * H * exp(A*(δt-s)) ds
"""
function frechet_derivative(A::AbstractMatrix, H::AbstractMatrix, δt::Real; n_quad::Int=50)
    n = size(A, 1)
    @assert size(A) == size(H) == (n, n) "Matrices must be square and same size"
    
    # Numerical integration using trapezoidal rule
    s_vals = range(0, δt, length=n_quad)
    ds = δt / (n_quad - 1)
    
    result = zeros(n, n)
    for s in s_vals
        # exp(A*s) * H * exp(A*(δt-s))
        term = exp(A * s) * H * exp(A * (δt - s))
        
        # Trapezoidal weights
        weight = (s == 0 || s == δt) ? 0.5 : 1.0
        result += weight * ds * term
    end
    
    return result
end


"""
Test Fréchet derivative using finite differences:
D(exp(A*δt))[H] ≈ [exp((A+ε*H)*δt) - exp(A*δt)] / ε
"""
function test_frechet_finite_difference()
    n = 3
    δt = 0.1
    
    # Create a simple generator matrix (M-matrix style)
    A = [-2.0 1.0 1.0;
          0.5 -1.5 1.0;
          0.5 0.5 -1.0]
    
    # Direction matrix
    H = randn(n, n)
    
    # Compute Fréchet derivative
    D_analytic = frechet_derivative(A, H, δt, n_quad=100)
    
    # Finite difference approximation
    ε = 1e-7
    exp_A_plus = exp((A + ε * H) * δt)
    exp_A = exp(A * δt)
    D_fd = (exp_A_plus - exp_A) / ε
    
    # Check agreement
    error = norm(D_analytic - D_fd, Inf)
    rel_error = error / norm(D_analytic, Inf)
    
    println("Finite Difference Test:")
    println(" Absolute error: $error")
    println(" Relative error: $rel_error")
    println(" ‖D_analytic‖_∞: $(norm(D_analytic, Inf))")
    println(" ‖D_fd‖_∞: $(norm(D_fd, Inf))")
    
    @test rel_error < 1e-5
    
    return rel_error
end


"""
Test using directional derivative definition directly
"""
function test_frechet_directional()
    n = 3
    δt = 0.1
    
    A = [-2.0 1.0 1.0;
          0.5 -1.5 1.0;
          0.5 0.5 -1.0]
    
    H = randn(n, n)
    
    # Compute Fréchet derivative
    D = frechet_derivative(A, H, δt, n_quad=100)
    
    # Test: should satisfy D(exp(A*δt))[H] * v ≈ d/dε[exp((A+εH)*δt)*v]|_{ε=0}
    v = randn(n)
    Dv = D * v
    
    # Finite difference for directional derivative
    ε = 1e-7
    fd_directional = (exp((A + ε * H) * δt) * v - exp(A * δt) * v) / ε
    
    error = norm(Dv - fd_directional, Inf)
    rel_error = error / norm(Dv, Inf)
    
    println("\nDirectional Derivative Test:")
    println(" Absolute error: $error")
    println(" Relative error: $rel_error")
    
    @test rel_error < 1e-5
    
    return rel_error
end


"""
Test symmetry property: Tr(G^T * D(exp(A*δt))[H]) = Tr(H^T * D(exp(A^T*δt))[G])
"""
function test_frechet_symmetry()
    n = 3
    δt = 0.1

    A = [-2.0  1.0  1.0;
          0.5 -1.5  1.0;
          0.5  0.5 -1.0]

    H = randn(n, n)
    G = randn(n, n)

    # Compute both Fréchet derivatives
    D_A_H = frechet_derivative(A, H, δt, n_quad=100)
    D_AT_G = frechet_derivative(A', G, δt, n_quad=100)  # Changed from G' to G

    # Check symmetry via trace
    lhs = tr(G' * D_A_H)
    rhs = tr(H' * D_AT_G)  # Changed from D_AT_GT * H to H' * D_AT_G

    error = abs(lhs - rhs)
    rel_error = error / max(abs(lhs), abs(rhs))

    println("\nSymmetry Test:")
    println("  LHS (Tr(G^T * D_A[H])): $lhs")
    println("  RHS (Tr(H^T * D_{A^T}[G])): $rhs")
    println("  Absolute error: $error")
    println("  Relative error: $rel_error")

    @test rel_error < 1e-10

    return rel_error
end


"""
Recover generator using Optim.jl with explicit Fréchet derivative gradients.

This integrates your frechet_derivative() function directly into Optim.jl's
optimization framework.

# Arguments
- `p_curr::Vector`: Current probability distribution
- `p_prev::Vector`: Previous probability distribution  
- `δt::Real`: Time step
- `A_init::Matrix`: Initial guess for generator
- `n_quad::Int`: Quadrature points for Fréchet derivative (default: 50)
- `method::Symbol`: Optimization method (:LBFGS, :BFGS, :GradientDescent, :ConjugateGradient)
- `max_iter::Int`: Maximum iterations (default: 1000)
- `g_tol::Float64`: Gradient tolerance (default: 1e-6)
- `show_trace::Bool`: Print optimization progress (default: true)

# Returns
- `A_opt::Matrix`: Recovered generator
- `result`: Optim.jl result object with convergence info

# Example
```julia
A_recovered, result = recover_generator_optim(
    p_curr, p_prev, δt, A_init;
    method = :LBFGS,
    n_quad = 50,
    show_trace = true
)
```
"""
function recover_generator_optim(
    p_curr::Vector{Float64},
    p_prev::Vector{Float64},
    δt::Real,
    A_init::AbstractMatrix;
    n_quad::Int = 50,
    method::Symbol = :LBFGS,
    max_iter::Int = 1000,
    g_tol::Float64 = 1e-6,
    show_trace::Bool = true
)
    n = length(p_curr)
    @assert length(p_prev) == n
    @assert size(A_init) == (n, n)
   
    # Convert to vector for Optim (it works with vectors)
    A_vec_init = vec(Matrix{Float64}(A_init))
   
    """
    Combined objective and gradient function.
    This is the key interface to Optim.jl with explicit gradients.
   
    F = objective value (if requested)
    G = gradient (if requested)
    A_vec = flattened generator matrix as vector
    """
    function objective_and_gradient!(F, G, A_vec)
        # Reshape vector back to matrix
        A = reshape(A_vec, n, n)
       
        # Compute gradient if requested
        if G !== nothing
            # Compute residual
            exp_A = exp(A * δt)
            r = p_curr - exp_A * p_prev
           
            # Compute gradient using Fréchet derivative
            # ∇L = -∫₀^δt exp(A'*s) * r * p_prev' * exp(A*(δt-s)) ds
            s_vals = range(0, δt, length=n_quad)
            ds = δt / (n_quad - 1)
           
            G_mat = zeros(n, n)
            for s in s_vals
                exp_As_T = exp(A' * s)
                exp_A_dt_minus_s = exp(A * (δt - s))
               
                term = exp_As_T * r * p_prev' * exp_A_dt_minus_s
               
                weight = (s == 0 || s == δt) ? 0.5 : 1.0
                G_mat .-= weight * ds * term
            end
           
            # Flatten gradient matrix to vector
            G .= vec(G_mat)
        end
       
        # Compute objective if requested
        if F !== nothing
            exp_A = exp(A * δt)
            r = p_curr - exp_A * p_prev
            return 0.5 * norm(r, 2)^2
        end
    end
   
    # Choose optimization algorithm
    if method == :LBFGS
        optimizer = LBFGS(linesearch=LineSearches.BackTracking())
    elseif method == :BFGS
        optimizer = BFGS(linesearch=LineSearches.BackTracking())
    elseif method == :GradientDescent
        optimizer = GradientDescent(linesearch=LineSearches.BackTracking())
    elseif method == :ConjugateGradient
        optimizer = ConjugateGradient(linesearch=LineSearches.BackTracking())
    else
        error("Unknown method: $method")
    end
   
    # Run optimization
    result = optimize(
        Optim.only_fg!(objective_and_gradient!),
        A_vec_init,
        optimizer,
        Optim.Options(
            iterations = max_iter,
            g_tol = g_tol,
            show_trace = show_trace,
            show_every = 10
        )
    )
   
    # Extract result
    A_opt = reshape(result.minimizer, n, n)
   
    return A_opt, result
end


"""
Recover generator with box constraints (for non-negativity of off-diagonals).

Uses Fminbox algorithm from Optim.jl which handles box constraints.

# Arguments
- Additional `lower` and `upper` bounds for each element of A

# Example
```julia
# Constrain off-diagonals to be non-negative
n = 3
lower = -Inf * ones(n, n)
upper = Inf * ones(n, n)
for i in 1:n, j in 1:n
    if i != j
        lower[i, j] = 0.0  # Off-diagonals must be ≥ 0
    end
end

A_recovered, result = recover_generator_boxconstrained(
    p_curr, p_prev, δt, A_init;
    lower = vec(lower),
    upper = vec(upper)
)
```
"""
function recover_generator_boxconstrained(
    p_curr::Vector{Float64},
    p_prev::Vector{Float64},
    δt::Real,
    A_init::AbstractMatrix;
    lower::Vector{Float64} = -Inf * ones(length(A_init)),
    upper::Vector{Float64} = Inf * ones(length(A_init)),
    n_quad::Int = 50,
    max_iter::Int = 1000,
    g_tol::Float64 = 1e-6,
    show_trace::Bool = true
)
    n = length(p_curr)
    A_vec_init = vec(Matrix{Float64}(A_init))
   
    function objective_and_gradient!(F, G, A_vec)
        A = reshape(A_vec, n, n)
       
        if G !== nothing
            exp_A = exp(A * δt)
            r = p_curr - exp_A * p_prev
           
            s_vals = range(0, δt, length=n_quad)
            ds = δt / (n_quad - 1)
           
            G_mat = zeros(n, n)
            for s in s_vals
                exp_As_T = exp(A' * s)
                exp_A_dt_minus_s = exp(A * (δt - s))
                term = exp_As_T * r * p_prev' * exp_A_dt_minus_s
                weight = (s == 0 || s == δt) ? 0.5 : 1.0
                G_mat .-= weight * ds * term
            end
           
            G .= vec(G_mat)
        end
       
        if F !== nothing
            exp_A = exp(A * δt)
            r = p_curr - exp_A * p_prev
            return 0.5 * norm(r, 2)^2
        end
    end
   
    # Use Fminbox for box-constrained optimization
    result = optimize(
        Optim.only_fg!(objective_and_gradient!),
        lower,
        upper,
        A_vec_init,
        Fminbox(LBFGS()),
        Optim.Options(
            iterations = max_iter,
            g_tol = g_tol,
            show_trace = show_trace,
            show_every = 10
        )
    )
   
    A_opt = reshape(result.minimizer, n, n)
   
    return A_opt, result
end


"""
Test recovery with Optim.jl integration.
"""
function test_recovery_optim()
    println("="^70)
    println("Testing Generator Recovery with Optim.jl")
    println("="^70)
   
    # Setup
    n = 3
    δt = 0.1
   
    # True generator
    A_true = [-2.0  1.0  1.0;
               0.5 -1.5  1.0;
               0.5  0.5 -1.0]
   
    println("\nTrue generator:")
    display(A_true)
   
    # Generate data
    p_prev = [1.0, 0.0, 0.0]
    p_curr = exp(A_true * δt) * p_prev
   
    println("\nGenerated data:")
    println("  p(t-δt) = ", p_prev)
    println("  p(t)    = ", p_curr)
   
    # Initial guess
    A_init = randn(n, n) * 0.1
   
    println("\n" * "="^70)
    println("Method 1: L-BFGS (unconstrained)")
    println("="^70)
   
    A_lbfgs, result_lbfgs = recover_generator_optim(
        p_curr, p_prev, δt, A_init;
        method = :LBFGS,
        n_quad = 50,
        show_trace = true
    )
   
    println("\nRecovered generator (L-BFGS):")
    display(A_lbfgs)
    println("\nError: ||A_true - A_recovered||_F = ", norm(A_true - A_lbfgs))
    println("Optim.jl summary:")
    println("  Converged: ", Optim.converged(result_lbfgs))
    println("  Iterations: ", Optim.iterations(result_lbfgs))
    println("  Final objective: ", Optim.minimum(result_lbfgs))
    println("  Final gradient norm: ", Optim.g_residual(result_lbfgs))
   
    # Forward simulation check
    p_pred = exp(A_lbfgs * δt) * p_prev
    println("\nForward error: ||p_curr - exp(A*δt)*p_prev|| = ", norm(p_curr - p_pred))
   
    println("\n" * "="^70)
    println("Method 2: BFGS (unconstrained)")
    println("="^70)
   
    A_bfgs, result_bfgs = recover_generator_optim(
        p_curr, p_prev, δt, A_init;
        method = :BFGS,
        n_quad = 50,
        show_trace = true
    )
   
    println("\nRecovered generator (BFGS):")
    display(A_bfgs)
    println("\nError: ||A_true - A_recovered||_F = ", norm(A_true - A_bfgs))
   
    println("\n" * "="^70)
    println("Method 3: Box-constrained (off-diagonals ≥ 0)")
    println("="^70)
   
    # Set up box constraints
    lower = -Inf * ones(n, n)
    upper = Inf * ones(n, n)
    for i in 1:n, j in 1:n
        if i != j
            lower[i, j] = 0.0  # Off-diagonals ≥ 0
        end
    end
   
    A_box, result_box = recover_generator_boxconstrained(
        p_curr, p_prev, δt, A_init;
        lower = vec(lower),
        upper = vec(upper),
        n_quad = 50,
        show_trace = true
    )
   
    println("\nRecovered generator (box-constrained):")
    display(A_box)
    println("\nError: ||A_true - A_recovered||_F = ", norm(A_true - A_box))
   
    # Check constraints
    println("\nConstraint verification:")
    println("  All off-diagonals ≥ 0: ",
            all(A_box[i,j] ≥ -1e-10 for i in 1:n for j in 1:n if i != j))
   
    return A_lbfgs, A_bfgs, A_box, result_lbfgs, result_bfgs, result_box
end


"""
Compare different Optim.jl methods.
"""
function compare_methods()
    println("="^70)
    println("Comparing Optimization Methods")
    println("="^70)
   
    n = 4
    δt = 0.1
   
    # Create a slightly larger test problem
    A_true = zeros(n, n)
    for i in 1:n-1
        A_true[i+1, i] = 2.0  # Birth
        A_true[i, i+1] = 1.0  # Death
    end
    for i in 1:n
        A_true[i, i] = -sum(A_true[:, i])
    end
   
    p_prev = zeros(n)
    p_prev[1] = 1.0
    p_curr = exp(A_true * δt) * p_prev
   
    A_init = randn(n, n) * 0.1
   
    methods = [:LBFGS, :BFGS, :ConjugateGradient, :GradientDescent]
    results = Dict()
   
    for method in methods
        println("\n" * "="^70)
        println("Testing method: $method")
        println("="^70)
       
        A_recovered, result = recover_generator_optim(
            p_curr, p_prev, δt, A_init;
            method = method,
            n_quad = 50,
            max_iter = 100,
            show_trace = false
        )
       
        results[method] = (
            A = A_recovered,
            error = norm(A_true - A_recovered),
            iterations = Optim.iterations(result),
            objective = Optim.minimum(result),
            converged = Optim.converged(result),
            time = result.time_run
        )
       
        println("Results:")
        println("  Error: ", results[method].error)
        println("  Iterations: ", results[method].iterations)
        println("  Final objective: ", results[method].objective)
        println("  Converged: ", results[method].converged)
        println("  Time: ", results[method].time, " seconds")
    end
   
    # Summary table
    println("\n" * "="^70)
    println("Summary Comparison")
    println("="^70)
    println("Method                Error        Iters    Time(s)   Converged")
    println("-"^70)
    for method in methods
        r = results[method]
        @printf("%-20s  %.2e  %6d   %7.4f   %s\n",
                method, r.error, r.iterations, r.time, r.converged)
    end
   
    return results
end

# Run all tests
println("="^60)
println("Testing Fréchet Derivative Implementation")
println("="^60)

test_frechet_finite_difference()
test_frechet_directional()
test_frechet_symmetry()

println("\n" * "="^60)
println("All tests passed! ✓")
println("="^60)

