"""
CRITICAL FIX: Proper gradient computation for single shooting
"""

using LinearAlgebra
using SparseArrays
using ExponentialUtilities
using NLopt

include("state_space.jl")

function vec_to_matrix(v, nz_off_diag, N_states)
    A = zeros(N_states, N_states)
    for (idx, cart_idx) in enumerate(nz_off_diag)
        A[cart_idx] = v[idx]
    end
    for j in 1:N_states
        A[j, j] = -sum(A[i, j] for i in 1:N_states if i != j)
    end
    return A
end

function matrix_to_vec(A, nz_off_diag)
    return [A[cart_idx] for cart_idx in nz_off_diag]
end

function objective_with_gradient!(
    v::Vector{Float64}, 
    grad::Vector{Float64},
    local_dists::Vector,
    window_times::Vector{Float64},
    N_states::Int,
    nz_off_diag::Vector,
    λ_frobenius::Float64,
    λ_prob_conservation::Float64
)
    # Reconstruct Matrix A
    A = vec_to_matrix(v, nz_off_diag, N_states)
    
    obj_data = 0.0
    obj_prob = 0.0
    
    if length(grad) > 0
        fill!(grad, 0.0)
    end
    
    # Single shooting: propagate forward and accumulate adjoint backward
    n_steps = length(local_dists)
    
    # Forward pass: compute predictions
    predictions = Vector{Vector{Float64}}(undef, n_steps)
    predictions[1] = local_dists[1]
    
    for i in 2:n_steps
        Δt = window_times[i] - window_times[i-1]
        predictions[i] = expv(Δt, A, predictions[i-1])
    end
    
    # Compute objective
    for i in 2:n_steps
        residual = predictions[i] - local_dists[i]
        obj_data += 0.5 * dot(residual, residual)
        
        prob_sum = sum(predictions[i])
        prob_error = prob_sum - 1.0
        obj_prob += 0.5 * λ_prob_conservation * prob_error^2
    end
    
    # Backward pass: compute gradients via adjoint
    if length(grad) > 0
        # Adjoint variable (costate) - accumulates backward
        λ = zeros(N_states)
        
        # Process steps in REVERSE order
        for i in n_steps:-1:2
            Δt = window_times[i] - window_times[i-1]
            
            # Gradient w.r.t. prediction at step i
            residual = predictions[i] - local_dists[i]
            prob_error = sum(predictions[i]) - 1.0
            
            grad_pred = residual + λ_prob_conservation * prob_error * ones(N_states)
            
            # Add to adjoint (from this step's loss)
            λ .+= grad_pred
            
            # Compute gradient contribution from this segment
            # Need: ∂L/∂A from propagating predictions[i-1] → predictions[i]
            
            At = copy(A')
            Q = λ * predictions[i-1]'  # Note: λ not grad_pred!
            
            # Van Loan block matrix
            M_block = zeros(2*N_states, 2*N_states)
            M_block[1:N_states, 1:N_states] = At * Δt
            M_block[1:N_states, (N_states+1):end] = Q * Δt
            M_block[(N_states+1):end, (N_states+1):end] = At * Δt
            
            E_block = exponential!(M_block, ExpMethodHigham2005())
            Grad_A_step = E_block[1:N_states, (N_states+1):end]
            
            # Accumulate gradient
            for (param_idx, cart_idx) in enumerate(nz_off_diag)
                row, col = cart_idx[1], cart_idx[2]
                grad[param_idx] += Grad_A_step[row, col] - Grad_A_step[col, col]
            end
            
            # Propagate adjoint backward through exponential
            # λ_{i-1} = exp(A'*Δt) * λ_i
            λ = expv(Δt, A', λ)
        end
    end
    
    # Regularization
    obj_frob = 0.5 * λ_frobenius * sum(v.^2)
    if length(grad) > 0
        grad .+= λ_frobenius * v
    end
    
    return obj_data + obj_frob + obj_prob
end

function optimize_local_generator(
    window_data::WindowData,
    config::InverseProblemConfig
)
    println("\n" * "="^60)
    println("Window $(window_data.window_idx): t ∈ [$(window_data.times[1]), $(window_data.times[end])]")
    println("="^60)
    
    X_local = build_local_state_space_with_connectivity(
        window_data.distributions, 
        window_data.stoich_vecs, 
        config.mass_threshold
    )
    
    X_local_cart = Set(CartesianIndex(state...) for state in X_local)
    X_local_vec = sort(collect(X_local_cart))
    N_states = length(X_local_vec)
    
    println("  Local state space: $N_states states")
    
    E_matrices = [reaction_direction(ν, X_local_cart) for ν in window_data.stoich_vecs]
    verify_connectivity(E_matrices)
    
    sparsity_pattern, nz_off_diag = build_sparsity_pattern(E_matrices, N_states)
    n_vars = length(nz_off_diag)
    
    println("  Parameters: $n_vars off-diagonal entries")
    
    if n_vars == 0
        @error "No parameters to optimize!"
        return zeros(N_states, N_states), X_local_vec, (success=false,)
    end
    
    local_dists = extract_local_distributions(window_data.distributions, X_local_vec)
    
    # NLopt objective wrapper
    function nlopt_objective!(v::Vector{Float64}, grad::Vector{Float64})
        return objective_with_gradient!(
            v, grad, local_dists, window_data.times,
            N_states, nz_off_diag,
            config.λ_frobenius, config.λ_prob_conservation
        )
    end
    
    # Initial guess
    k_init = ones(length(window_data.stoich_vecs)) * 0.1
    A_init = sum(k_init[j] * E_matrices[j] for j in 1:length(window_data.stoich_vecs))
    v_init = matrix_to_vec(Matrix(A_init), nz_off_diag)
    
    # Test initial objective AND GRADIENT
    println("  Testing initial point...")
    grad_test = zeros(length(v_init))
    obj_init = nlopt_objective!(v_init, grad_test)
    println("  Initial objective: $obj_init")
    println("  Initial gradient norm: $(norm(grad_test))")
    
    if norm(grad_test) < 1e-10
        @error "GRADIENT IS ZERO! Optimization will fail."
        return Matrix(A_init), X_local_vec, (success=false, message="Zero gradient")
    end
    
    if !isfinite(obj_init)
        @error "Initial objective is not finite!"
        return Matrix(A_init), X_local_vec, (success=false,)
    end
    
    # Setup optimizer
    opt = NLopt.Opt(:LD_LBFGS, n_vars)
    NLopt.lower_bounds!(opt, zeros(n_vars))
    NLopt.upper_bounds!(opt, fill(10.0, n_vars))
    NLopt.min_objective!(opt, nlopt_objective!)
    NLopt.xtol_rel!(opt, 1e-6)
    NLopt.ftol_rel!(opt, 1e-8)
    NLopt.maxeval!(opt, 2000)
    
    println("  Starting optimization...")
    (minf, minx, ret) = NLopt.optimize(opt, v_init)
    
    println("  Return: $ret, Objective: $minf")
    println("  Improvement: $(obj_init - minf) ($(round(100*(obj_init - minf)/obj_init, digits=1))%)")
    
    A_learned = vec_to_matrix(minx, nz_off_diag, N_states)
    
    # Validation
    col_sums = vec(sum(A_learned, dims=1))
    max_col_sum_error = maximum(abs.(col_sums))
    
    p_test = expv(0.1, sparse(A_learned), ones(N_states)/N_states)
    prob_conservation = sum(p_test)
    
    println("  Validation:")
    println("    Max column sum error: $max_col_sum_error")
    println("    Probability conservation: $prob_conservation")
    println("    Nonzeros: $(count(abs.(A_learned) .> 1e-10))")
    println("    Max entry: $(maximum(abs.(A_learned)))")
    
    convergence_info = (
        success = (ret == :SUCCESS || ret == :FTOL_REACHED || ret == :XTOL_REACHED),
        return_code = ret,
        final_objective = minf,
        initial_objective = obj_init,
        improvement = obj_init - minf,
        max_col_sum_error = max_col_sum_error,
        prob_conservation = prob_conservation,
        n_nonzeros = count(abs.(A_learned) .> 1e-10)
    )
    
    return A_learned, X_local_vec, convergence_info
end
