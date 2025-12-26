"""
Optimization routines for learning generator matrices.
"""

using LinearAlgebra
using SparseArrays
using ExponentialUtilities
using NLopt

include("state_space.jl")

"""
    vec_to_matrix(v, nz_off_diag, N_states)

Convert parameter vector to generator matrix with zero column sums.

# Arguments
- `v`: Parameter vector (off-diagonal entries)
- `nz_off_diag`: Vector of CartesianIndex for off-diagonal positions
- `N_states`: Size of the matrix

# Returns
Generator matrix A with column sums equal to zero
"""
function vec_to_matrix(v, nz_off_diag, N_states)
    A = zeros(N_states, N_states)
    
    # Set off-diagonal entries
    for (idx, cart_idx) in enumerate(nz_off_diag)
        A[cart_idx] = v[idx]
    end
    
    # Enforce zero column sums (diagonals = negative sum of column)
    for j in 1:N_states
        A[j, j] = -sum(A[i, j] for i in 1:N_states if i != j)
    end
    
    return A
end

"""
    matrix_to_vec(A, nz_off_diag)

Extract parameter vector from generator matrix.
"""
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
    λ_prob_conservation::Float64,
    δ_huber::Float64 = 0.01
)
    A = vec_to_matrix(v, nz_off_diag, N_states)
    
    obj_data = 0.0
    obj_prob = 0.0
    
    if length(grad) > 0
        fill!(grad, 0.0)
    end
    
    # --- Term 1: Data fitting (Multiple Shooting) ---
    p_current = local_dists[1]
    
    for i in 2:length(local_dists)
        p_target = local_dists[i]
        Δt = window_times[i] - window_times[i-1]
        
        # 1. Forward Pass
        p_pred = expv(Δt, A, p_current)
        residual = p_pred - p_target
        
        # 2. Huber Loss & Gradient w.r.t Prediction
        grad_wrt_pred = zeros(N_states)
        for (idx, r) in enumerate(residual)
            if abs(r) <= δ_huber
                obj_data += 0.5 * r^2
                grad_wrt_pred[idx] = r
            else
                obj_data += δ_huber * (abs(r) - 0.5*δ_huber)
                grad_wrt_pred[idx] = δ_huber * sign(r)
            end
        end
        
        # 3. Adjoint Gradient (Fast O(N^3))
        if length(grad) > 0
            # Q = (grad_L / dp)^T * p_start^T
            # Note: In integral form, Q is effectively grad_wrt_pred * p_current'
            Q = grad_wrt_pred * p_current'
            
            # Block Matrix Exponential for Van Loan Integral
            # [ A'  Q ]
            # [ 0   A'] * dt
            At = copy(A')
            M_block = zeros(2*N_states, 2*N_states)
            M_block[1:N_states, 1:N_states] .= At .* Δt
            M_block[1:N_states, (N_states+1):end] .= Q .* Δt
            M_block[(N_states+1):end, (N_states+1):end] .= At .* Δt
            
            expM = exponential!(M_block, ExpMethodHigham2005())
            
            # Integral is top-right block
            Grad_A = expM[1:N_states, (N_states+1):end]
            
            # Map back to parameters v (accounting for column sum constraint)
            for (param_idx, cart_idx) in enumerate(nz_off_diag)
                row, col = cart_idx[1], cart_idx[2]
                # dJ/dv = dJ/dA_ij * 1 + dJ/dA_jj * (-1)
                grad[param_idx] += Grad_A[row, col] - Grad_A[col, col]
            end
        end
        
        # CRITICAL FIX: Reset to Data (Multiple Shooting)
        p_current = p_target 
    end
    
    # --- Term 2: Frobenius ---
    obj_frob = 0.5 * λ_frobenius * sum(v.^2)
    if length(grad) > 0
        grad .+= λ_frobenius * v
    end
    
    # --- Term 3: Probability Conservation ---
    # Kept as Single Shooting to match your original file logic
    p_current = local_dists[1]
    for i in 2:length(local_dists)
        Δt = window_times[i] - window_times[i-1]
        p_pred = expv(Δt, sparse(A), p_current) # Sparse is fine here if just checking sum
        
        prob_sum = sum(p_pred)
        prob_error = prob_sum - 1.0
        obj_prob += 0.5 * λ_prob_conservation * prob_error^2
        
        # Gradient for conservation (Simplified Adjoint)
        if length(grad) > 0 && abs(prob_error) > 1e-8
            # The gradient of the error term w.r.t p_pred is just scalar * ones
            grad_wrt_pred = (λ_prob_conservation * prob_error) .* ones(N_states)
            
            Q = grad_wrt_pred * p_current'
            
            At = copy(A')
            M_block = zeros(2*N_states, 2*N_states)
            M_block[1:N_states, 1:N_states] .= At .* Δt
            M_block[1:N_states, (N_states+1):end] .= Q .* Δt
            M_block[(N_states+1):end, (N_states+1):end] .= At .* Δt
            
            expM = exponential!(M_block, ExpMethodHigham2005())
            Grad_A = expM[1:N_states, (N_states+1):end]
            
            for (param_idx, cart_idx) in enumerate(nz_off_diag)
                row, col = cart_idx[1], cart_idx[2]
                grad[param_idx] += Grad_A[row, col] - Grad_A[col, col]
            end
        end
        p_current = p_pred # Keep propagation for this term
    end
    
    return obj_data + obj_frob + obj_prob
end

"""
    optimize_local_generator(window_data::WindowData, config::InverseProblemConfig)

Learn generator matrix for a single time window.

# Arguments
- `window_data`: WindowData containing distributions, times, and stoichiometry
- `config`: InverseProblemConfig with optimization parameters

# Returns
Tuple (A_learned, X_local_vec, convergence_info) where:
- `A_learned`: Learned generator matrix
- `X_local_vec`: Local state space (CartesianIndex vector)
- `convergence_info`: NamedTuple with optimization details
"""
function optimize_local_generator(
    window_data::WindowData,
    config::InverseProblemConfig
)
    println("\n" * "="^60)
    println("Window $(window_data.window_idx): t ∈ [$(window_data.times[1]), $(window_data.times[end])]")
    println("="^60)
    
    # Build local state space with connectivity
    X_local = build_local_state_space_with_connectivity(
        window_data.distributions, 
        window_data.stoich_vecs, 
        config.mass_threshold
    )
    
    # Convert to CartesianIndex for compatibility
    X_local_cart = Set(CartesianIndex(state...) for state in X_local)
    X_local_vec = sort(collect(X_local_cart))
    N_states = length(X_local_vec)
    
    println("  Local state space: $N_states states")
    println("  Sample states: $(X_local_vec[1:min(3,end)])")
    
    # Build E matrices
    E_matrices = [reaction_direction(ν, X_local_cart) for ν in window_data.stoich_vecs]
    
    # Verify connectivity
    verify_connectivity(E_matrices)
    
    # Build sparsity pattern
    sparsity_pattern, nz_off_diag = build_sparsity_pattern(E_matrices, N_states)
    n_vars = length(nz_off_diag)
    
    println("  Parameters: $n_vars off-diagonal entries")
    
    if n_vars == 0
        @error "No parameters to optimize! Check stoichiometry and state space."
        return zeros(N_states, N_states), X_local_vec, (success=false,)
    end
    
    # Extract local distributions
    local_dists = extract_local_distributions(window_data.distributions, X_local_vec)
    
    # Check distribution differences
    println("  Distribution differences:")
    for i in 2:min(4, length(local_dists))
        diff = norm(local_dists[i] - local_dists[i-1])
        println("    Step $(i-1)->$i: $diff")
    end
    
    # Closure for NLopt
    function nlopt_objective!(v::Vector{Float64}, grad::Vector{Float64})
        return objective_with_gradient!(
            v, grad, local_dists, window_data.times,
            N_states, nz_off_diag,
            config.λ_frobenius, config.λ_prob_conservation
        )
    end
    
    # Initial guess: uniform small positive rates
    k_init = ones(length(window_data.stoich_vecs)) * 0.1
    A_init = sum(k_init[j] * E_matrices[j] for j in 1:length(window_data.stoich_vecs))
    v_init = matrix_to_vec(Matrix(A_init), nz_off_diag)
    
    # Test initial objective
    println("  Testing initial objective...")
    obj_init = nlopt_objective!(v_init, Float64[])
    println("  Initial objective: $obj_init")
    
    if !isfinite(obj_init)
        @error "Initial objective is not finite!"
        return Matrix(A_init), X_local_vec, (success=false, message="Non-finite initial objective")
    end
    
    # Setup NLopt optimizer
    opt = NLopt.Opt(:LD_LBFGS, n_vars)
    NLopt.lower_bounds!(opt, zeros(n_vars))
    NLopt.upper_bounds!(opt, fill(10.0, n_vars))
    NLopt.min_objective!(opt, nlopt_objective!)
    NLopt.xtol_rel!(opt, 1e-4)
    NLopt.ftol_rel!(opt, 1e-6)
    NLopt.maxeval!(opt, 200)
    
    # Optimize
    println("  Starting optimization...")
    (minf, minx, ret) = NLopt.optimize(opt, v_init)
    
    println("  Return: $ret, Objective: $minf")
    
    # Reconstruct learned matrix
    A_learned = vec_to_matrix(minx, nz_off_diag, N_states)
    
    # Validate
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
        max_col_sum_error = max_col_sum_error,
        prob_conservation = prob_conservation,
        n_nonzeros = count(abs.(A_learned) .> 1e-10)
    )
    
    return A_learned, X_local_vec, convergence_info
end
