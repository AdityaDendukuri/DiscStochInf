# File: src/optimization.jl
"""
Optimization routines for learning generators using adjoint method.
"""

using LinearAlgebra
using Optim
using ExponentialUtilities

struct TimeWindowData{T<:Real}
    p_curr::Vector{T}
    p_next::Vector{T}
    state_space::Vector{Vector{Int}}
end

"""
    TimeWindowData(hist_curr, hist_next, state_space)

Build time window data from histograms.
"""
function TimeWindowData(hist_curr, hist_next, state_space)
    n = length(state_space)
    p_curr = [get(hist_curr, s, 0.0) for s in state_space]
    p_next = [get(hist_next, s, 0.0) for s in state_space]
    return TimeWindowData(p_curr, p_next, state_space)
end

# =============================================================================
# Adjoint Fréchet Derivative
# =============================================================================

"""
    frechet_adjoint(A, Λ)

Compute adjoint Fréchet derivative of matrix exponential.

Mathematical formula:
    L*(A, Λ) = ∫₀¹ exp(A'(1-s)) Λ exp(A's) ds

Computed via block exponential:
    exp([A' Λ; 0 A'])[1:n, n+1:2n]

Returns: ∇_A tr(Λ' exp(A))
"""
function frechet_adjoint(A::Matrix, Λ::Matrix)
    n = size(A, 1)
    
    # Check for numerical issues
    if any(!isfinite, A)
        @error "Generator A contains NaN or Inf!"
        return zeros(n, n)
    end
    
    if any(!isfinite, Λ)
        @error "Sensitivity Λ contains NaN or Inf!"
        return zeros(n, n)
    end
    
    # Build block matrix with A' in BOTH diagonal blocks
    M = [A' Λ; zeros(n, n) A']
    
    # Compute exponential
    try
        expM = exponential!(M, ExpMethodHigham2005Base())
        L_adj = expM[1:n, n+1:end]
        
        # Check output
        if any(!isfinite, L_adj)
            @warn "Adjoint Fréchet produced NaN/Inf, returning zeros"
            return zeros(n, n)
        end
        
        return L_adj
    catch e
        @error "Matrix exponential failed in frechet_adjoint" exception=e
        @error "  ||A||=$(norm(A)), ||Λ||=$(norm(Λ))"
        return zeros(n, n)
    end
end

# =============================================================================
# Objective and Gradient (Adjoint Method)
# =============================================================================

"""
    compute_objective_gradient_adjoint(θ, data, windows; λ1=1e-6, λ3=0.0, verbose=false)

Compute objective and gradient using adjoint method.

Objective:
    J(θ) = Σ_w ||p_{w+1} - exp(A·Δt)p_w||₁ + λ₁||A||²_F + λ₃||θ||₁

Parameters:
    - λ1: Frobenius regularization (L2 on generator)
    - λ3: L1 sparsity regularization on parameters
    - verbose: Print diagnostics

Note: NO semigroup constraint (enforced structurally in generator construction)
"""
function compute_objective_gradient_adjoint(θ, data, windows; 
                                           λ1=1e-6, λ3=0.0, verbose=false)
    A = build_generator(θ, data)
    n = size(A, 1)
    
    # Diagnostic: Check generator
    norm_A = norm(A)
    if verbose
        println("  ||A|| = $(round(norm_A, digits=2))")
    end
    
    if !isfinite(norm_A)
        @error "Generator contains NaN or Inf!"
        return Inf, fill(NaN, length(θ))
    end
    
    # Precompute propagator (same for all windows)
    P = exp(Matrix(A * data.dt))
    
    if any(!isfinite, P)
        @error "Propagator exp(A·dt) contains NaN/Inf!" norm_A=norm_A dt=data.dt
        return Inf, fill(NaN, length(θ))
    end
    
    obj = 0.0
    
    # Accumulate sensitivities ∂J/∂P from all windows
    Λ_P_total = zeros(n, n)
    
    # === FORWARD PASS (all windows) ===
    for window in windows
        # Prediction
        q = P * window.p_curr
        r = window.p_next - q
        
        # L1 prediction error
        obj += sum(abs.(r))
        
        # === BACKWARD ACCUMULATION ===
        # ∂J/∂P from prediction: -sign(r) ⊗ p_w
        Λ_P_pred = -sign.(r) * window.p_curr'
        
        # Accumulate
        Λ_P_total += Λ_P_pred
    end
    
    # === ADJOINT STEP (computed ONCE) ===
    # Compute ∂J/∂B where B = A·dt
    Λ_B = frechet_adjoint(A * data.dt, Λ_P_total)
    
    # Chain rule: ∂J/∂A = ∂J/∂B · ∂B/∂A = Λ_B · dt
    Λ_A = Λ_B * data.dt
    
    # Frobenius regularization
    obj += λ1 * sum(abs2, A)
    Λ_A += 2λ1 * A
    
    # === PROJECT ONTO PARAMETERS ===
    grad = zeros(length(θ))
    
    for k in 1:length(data.stoich_basis)
        for f in 1:data.n_features
            idx = (k-1)*data.n_features + f
            
            # E_kf = ∂A/∂θ_kf
            E = build_perturbation(k, f, data)
            
            # Gradient: ∂J/∂θ_kf = tr(Λ_A' · E) = Λ_A : E
            grad[idx] = sum(Λ_A .* E)
        end
    end
    
    # L1 regularization on parameters (optional)
    if λ3 > 0
        obj += λ3 * sum(abs.(θ))
        grad += λ3 * sign.(θ)
    end
    
    return obj, grad
end

# =============================================================================
# Forward Fréchet Method (for verification)
# =============================================================================

"""
    frechet(A, E)

Compute Fréchet derivative L(A, E) via block exponential.
Returns (exp(A), L(A, E)).
"""
function frechet(A::Matrix, E::Matrix)
    n = size(A, 1)
    M = [A E; zeros(n, n) A]
    expM = exponential!(M, ExpMethodHigham2005Base())
    return expM[1:n, 1:n], expM[1:n, n+1:end]
end

"""
    compute_objective_gradient_frechet(θ, data, windows; λ1=1e-6, λ3=0.0)

Compute objective and gradient using forward Fréchet method.
(For verification against adjoint method)
"""
function compute_objective_gradient_frechet(θ, data, windows; λ1=1e-6, λ3=0.0)
    A = build_generator(θ, data)
    
    obj = 0.0
    grad = zeros(length(θ))
    
    # Loop over windows
    for window in windows
        expA_dt = nothing
        residual = nothing
        
        # Gradient via Fréchet derivatives
        for k in 1:length(data.stoich_basis)
            for f in 1:data.n_features
                idx = (k-1)*data.n_features + f
                
                E = build_perturbation(k, f, data)
                expA, L = frechet(A * data.dt, E * data.dt)
                
                # On first iteration, compute objective
                if expA_dt === nothing
                    expA_dt = expA
                    p_pred = expA_dt * window.p_curr
                    residual = window.p_next - p_pred
                    obj += sum(abs.(residual))
                end
                
                # Prediction gradient
                grad[idx] -= dot(sign.(residual), L * window.p_curr)
            end
        end
    end
    
    # Frobenius regularization
    obj += λ1 * sum(abs2, A)
    for k in 1:length(data.stoich_basis)
        for f in 1:data.n_features
            idx = (k-1)*data.n_features + f
            E = build_perturbation(k, f, data)
            grad[idx] += 2λ1 * sum(A .* E)
        end
    end
    
    # L1 regularization
    if λ3 > 0
        obj += λ3 * sum(abs.(θ))
        grad += λ3 * sign.(θ)
    end
    
    return obj, grad
end

# =============================================================================
# Learning Functions
# =============================================================================

"""
    learn_generator(data, windows; kwargs...)

Learn generator from histogram windows using L-BFGS.

Parameters:
    - λ1: Frobenius regularization (default: 1e-6)
    - λ3: L1 sparsity on parameters (default: 0.0)
    - max_iter: Maximum iterations (default: 300)
    - show_trace: Show optimization progress (default: false)
    - grad_clip: Gradient clipping threshold (default: Inf, no clipping)
    - method: :adjoint or :frechet (default: :adjoint)
"""
function learn_generator(data, windows; 
                        λ1=1e-6, λ3=0.0,
                        max_iter=300, show_trace=true, 
                        grad_clip=Inf, method=:adjoint)
    n_params = length(data.stoich_basis) * data.n_features
    
    # Initialize
    θ0 = 0.01 * randn(n_params)
    
    # Choose method
    if method == :adjoint
        compute_fg = compute_objective_gradient_adjoint
    elseif method == :frechet
        compute_fg = compute_objective_gradient_frechet
    else
        error("Unknown method: $method. Use :adjoint or :frechet")
    end
    
    # Optimization
    function fg!(F, G, θ)
        obj, grad = compute_fg(θ, data, windows; λ1=λ1, λ3=λ3)
        
        if G !== nothing
            # Optional gradient clipping
            if isfinite(grad_clip) && grad_clip > 0
                grad_norm = norm(grad)
                if grad_norm > grad_clip
                    grad = grad * (grad_clip / grad_norm)
                end
            end
            copyto!(G, grad)
        end
        
        return obj
    end
    
    result = optimize(Optim.only_fg!(fg!), θ0, LBFGS(), 
                     Optim.Options(iterations=max_iter, 
                                   show_trace=show_trace,
                                   show_every=10,
                                   g_tol=1e-6))
    
    θ_opt = Optim.minimizer(result)
    A_opt = build_generator(θ_opt, data)
    
    return A_opt, θ_opt, Optim.converged(result)
end

# =============================================================================
# Verification
# =============================================================================

"""
    verify_adjoint_gradient(θ, data, windows; verbose=true)

Verify adjoint gradient against forward Fréchet gradient.
"""
function verify_adjoint_gradient(θ, data, windows; verbose=true)
    println("\n" * "="^70)
    println("GRADIENT VERIFICATION: Adjoint vs Fréchet")
    println("="^70)
    
    # Compute both methods
    obj_adj, grad_adj = compute_objective_gradient_adjoint(θ, data, windows, λ1=1e-6)
    obj_fre, grad_fre = compute_objective_gradient_frechet(θ, data, windows, λ1=1e-6)
    
    # Compare objectives
    obj_diff = abs(obj_adj - obj_fre)
    obj_rel = obj_diff / (abs(obj_fre) + 1e-10)
    
    # Compare gradients
    grad_diff = norm(grad_adj - grad_fre)
    grad_rel = grad_diff / (norm(grad_fre) + 1e-10)
    
    if verbose
        println("\nObjective:")
        println("  Adjoint:  $(round(obj_adj, digits=8))")
        println("  Fréchet:  $(round(obj_fre, digits=8))")
        println("  Abs diff: $(round(obj_diff, sigdigits=3))")
        println("  Rel diff: $(round(obj_rel, sigdigits=3))")
        
        println("\nGradient:")
        println("  ||g_adj||: $(round(norm(grad_adj), digits=6))")
        println("  ||g_fre||: $(round(norm(grad_fre), digits=6))")
        println("  ||diff||:  $(round(grad_diff, sigdigits=3))")
        println("  Rel err:   $(round(grad_rel, sigdigits=3))")
    end
    
    # Pass/fail
    tol = 1e-6
    if obj_rel < tol && grad_rel < tol
        println("\n✅ PASS: Gradients match within tolerance ($tol)")
        return true
    elseif obj_rel < 1e-3 && grad_rel < 1e-3
        println("\n⚠️  WARN: Small discrepancy (< 1e-3), likely numerical")
        return true
    else
        println("\n❌ FAIL: Large discrepancy!")
        return false
    end
end
