# File: src/optimization.jl

"""
Optimization routines for learning generators.
"""

using LinearAlgebra
using Optim
using ExponentialUtilities

struct TimeWindowData{T<:Real}
    p_curr::Vector{T}
    p_next::Vector{T}
end

"""
    TimeWindowData(hist_curr, hist_next, state_space)

Build time window data from histograms.
"""
function TimeWindowData(hist_curr, hist_next, state_space)
    n = length(state_space)
    p_curr = [get(hist_curr, s, 0.0) for s in state_space]
    p_next = [get(hist_next, s, 0.0) for s in state_space]
    return TimeWindowData(p_curr, p_next)
end

"""
    frechet(A, E)

Compute Fréchet derivative L(A, E) via block exponential.
"""
function frechet(A::Matrix, E::Matrix)
    n = size(A, 1)
    M = [A E; zeros(n, n) A]
    expM = exponential!(M, ExpMethodHigham2005Base())
    return expM[1:n, n+1:end]
end

"""
    compute_objective_gradient(θ, data, windows; λ1=1e-8, λ2=1e-6)

Compute objective and gradient for all windows.
"""
function compute_objective_gradient(θ, data, windows; λ1=1e-8, λ2=1e-6)
    A = build_generator(θ, data)
    
    obj = 0.0
    grad = zeros(length(θ))
    
    # Loop over windows
    for window in windows
        # Prediction
        expA = exponential!(Matrix(A * data.dt), ExpMethodHigham2005Base())
        p_pred = expA * window.p_curr
        residual = window.p_next - p_pred
        
        # L1 objective
        obj += sum(abs.(residual))
        
        # Gradient via Fréchet derivatives
        for k in 1:length(data.stoich_basis)
            for f in 1:data.n_features
                idx = (k-1)*data.n_features + f
                
                E = build_perturbation(k, f, data)
                L = frechet(A * data.dt, E * data.dt)
                
                grad[idx] -= dot(sign.(residual), L * window.p_curr)
            end
        end
    end
    
    # Regularization
    obj += λ1 * norm(A, 2)^2
    grad .+= 2λ1 * [sum(A .* build_perturbation(k, f, data)) 
                     for k in 1:length(data.stoich_basis) 
                     for f in 1:data.n_features]
    
    # Semigroup penalty
    col_sums = sum(A, dims=1)[:]
    obj += λ2 * sum(col_sums.^2)
    for k in 1:length(data.stoich_basis)
        for f in 1:data.n_features
            idx = (k-1)*data.n_features + f
            E = build_perturbation(k, f, data)
            grad[idx] += 2λ2 * dot(col_sums, sum(E, dims=1)[:])
        end
    end
    
    return obj, grad
end

"""
    learn_generator(data, windows; λ1=1e-8, λ2=1e-6, max_iter=200, show_trace=false)

Learn generator from histogram windows using L-BFGS.
"""
function learn_generator(data, windows; λ1=1e-8, λ2=1e-6, max_iter=200, show_trace=false)
    n_params = length(data.stoich_basis) * data.n_features
    
    # Initialize
    θ0 = 0.1 * randn(n_params)
    
    # Optimization
    function fg!(F, G, θ)
        obj, grad = compute_objective_gradient(θ, data, windows; λ1=λ1, λ2=λ2)
        G !== nothing && copyto!(G, grad)
        return obj
    end
    
    result = optimize(Optim.only_fg!(fg!), θ0, LBFGS(), 
                     Optim.Options(iterations=max_iter, show_trace=show_trace))
    
    θ_opt = Optim.minimizer(result)
    A_opt = build_generator(θ_opt, data)
    
    return A_opt, θ_opt, Optim.converged(result)
end
