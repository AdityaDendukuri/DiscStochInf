using LinearAlgebra
using Test

"""
Compute Fréchet derivative of exp(A*δt) in direction H using integral formula:
D(exp(A*δt))[H] = ∫₀^δt exp(A*s) * H * exp(A*(δt-s)) ds
"""
function frechet_derivative(A::Matrix, H::Matrix, δt::Real; n_quad::Int=50)
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
    @info A
    @info H
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
Test symmetry property: Tr(G^T * D(exp(A*δt))[H]) = Tr(D(exp(A^T*δt))[G^T] * H)
"""
function test_frechet_symmetry()
    n = 3
    δt = 0.1
    
    A = [-2.0 1.0 1.0;
          0.5 -1.5 1.0;
          0.5 0.5 -1.0]
    
    H = randn(n, n)
    G = randn(n, n)
    
    # Compute both Fréchet derivatives
    D_A_H = frechet_derivative(A, H, δt, n_quad=100)
    D_AT_GT = frechet_derivative(A', G', δt, n_quad=100)
    
    # Check symmetry via trace
    lhs = tr(G' * D_A_H)
    rhs = tr(D_AT_GT * H)
    
    error = abs(lhs - rhs)
    rel_error = error / max(abs(lhs), abs(rhs))
    
    println("\nSymmetry Test:")
    println(" LHS (Tr(G^T * D_A[H])): $lhs")
    println(" RHS (Tr(D_A^T[G^T] * H)): $rhs")
    println(" Absolute error: $error")
    println(" Relative error: $rel_error")
    
    @test rel_error < 1e-10
    
    return rel_error
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

