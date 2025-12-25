using LinearAlgebra
using SparseArrays
using ExponentialUtilities
using NLopt

include("state_space.jl")

# Keep your existing helper functions (vec_to_matrix, etc.)
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

"""
    objective_with_gradient!

COMPUTATIONALLY OPTIMIZED + STABILIZED
1. Uses Adjoint Method (Van Loan Integral) -> O(N^3) speed instead of O(R*N^3)
2. Uses Multiple Shooting (Resets to Data) -> Stable for stiff/chaotic systems
"""
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
    # 1. Reconstruct Matrix A
    A = vec_to_matrix(v, nz_off_diag, N_states)
    
    obj_data = 0.0
    obj_prob = 0.0
    
    if length(grad) > 0
        fill!(grad, 0.0)
    end
    
    # --- MULTIPLE SHOOTING LOOP ---
    # Critical Fix: We iterate through intervals, treating each as independent
    # We start from observed data (local_dists[i-1]), NOT p_pred
    
    for i in 2:length(local_dists)
        p_start = local_dists[i-1] # ALWAYS use observed data as start
        p_target = local_dists[i]
        Δt = window_times[i] - window_times[i-1]
        
        # --- Forward Pass ---
        p_pred = expv(Δt, A, p_start)
        
        # --- Loss Calculation ---
        residual = p_pred - p_target
        
        # Huber Loss
        grad_wrt_pred = zeros(N_states)
        for (idx, r) in enumerate(residual)
            if abs(r) <= δ_huber
                obj_data += 0.5 * r^2
                grad_wrt_pred[idx] += r
            else
                obj_data += δ_huber * (abs(r) - 0.5*δ_huber)
                grad_wrt_pred[idx] += δ_huber * sign(r)
            end
        end
        
        # Probability Conservation Penalty
        prob_sum = sum(p_pred)
        prob_error = prob_sum - 1.0
        obj_prob += 0.5 * λ_prob_conservation * prob_error^2
        if abs(prob_error) > 1e-10
             grad_wrt_pred .+= λ_prob_conservation * prob_error
        end

        # --- Backward Pass (Adjoint Method) ---
        if length(grad) > 0
            # Adjoint logic:
            # We need dJ/dA = Integral_0^t e^{A' s} (grad_wrt_pred * p_start') e^{A' (t-s)} ds
            # This is the top-right block of exp([A' Q; 0 A'] * t)
            
            At = copy(A')
            Q = grad_wrt_pred * p_start'
            
            # Construct 2N x 2N block matrix
            M_block = zeros(2*N_states, 2*N_states)
            M_block[1:N_states, 1:N_states] = At .* Δt
            M_block[1:N_states, (N_states+1):end] = Q .* Δt
            M_block[(N_states+1):end, (N_states+1):end] = At .* Δt
            
            # Compute exponential ONCE per time step
            E_block = exponential!(M_block, ExpMethodHigham2005())
            Grad_A_step = E_block[1:N_states, (N_states+1):end]
            
            # Map back to parameters v
            for (param_idx, cart_idx) in enumerate(nz_off_diag)
                row, col = cart_idx[1], cart_idx[2]
                # dJ/dv_ij = dJ/dA_ij - dJ/dA_jj (due to column sum constraint)
                grad[param_idx] += Grad_A_step[row, col] - Grad_A_step[col, col]
            end
        end
    end
    
    # Regularization
    obj_frob = 0.5 * λ_frobenius * sum(v.^2)
    if length(grad) > 0
        grad .+= λ_frobenius * v
    end
    
    return obj_data + obj_frob + obj_prob
end

function optimize_local_generator(window_data, config)
    println("\n" * "="^60)
    println("Window $(window_data.window_idx): t ∈ [$(window_data.times[1]), $(window_data.times[end])]")
    println("="^60)
    
    # Build local state space with connectivity
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
    sparsity_pattern, nz_off_diag = build_sparsity_pattern(E_matrices, N_states)
    n_vars = length(nz_off_diag)
    println("  Parameters: $n_vars off-diagonal entries")
    
    if n_vars == 0
        return zeros(N_states, N_states), X_local_vec, (success=false,)
    end
    
    local_dists = extract_local_distributions(window_data.distributions, X_local_vec)
    
    # --- INITIALIZATION FIX ---
    # Heuristic: Initialize rates inversely proportional to reaction order
    # Brusselator Reaction 2 is 2X + Y -> 3X (Order 3).
    # 0.1^3 = 0.001. This prevents the "explosion" at start.
    k_init = zeros(length(window_data.stoich_vecs))
    for (r, nu) in enumerate(window_data.stoich_vecs)
        order = sum(c < 0 ? abs(c) : 0 for c in nu)
        k_init[r] = 0.1 ^ order 
    end
    println("  Safe Initialization Rates: $k_init")
    
    A_init = sum(k_init[j] * E_matrices[j] for j in 1:length(k_init))
    v_init = matrix_to_vec(Matrix(A_init), nz_off_diag)

    # Closure for NLopt
    function nlopt_objective!(v::Vector{Float64}, grad::Vector{Float64})
        return objective_with_gradient!(
            v, grad, local_dists, window_data.times,
            N_states, nz_off_diag,
            config.λ_frobenius, config.λ_prob_conservation
        )
    end
    
    # Setup NLopt optimizer
    opt = NLopt.Opt(:LD_LBFGS, n_vars)
    NLopt.lower_bounds!(opt, zeros(n_vars))
    NLopt.upper_bounds!(opt, fill(100.0, n_vars))
    NLopt.min_objective!(opt, nlopt_objective!)
    NLopt.xtol_rel!(opt, 1e-4)
    NLopt.ftol_rel!(opt, 1e-6)
    NLopt.maxeval!(opt, 500)
    
    println("  Starting optimization...")
    t_start = time()
    (minf, minx, ret) = NLopt.optimize(opt, v_init)
    t_end = time()
    
    println("  Return: $ret, Objective: $minf, Time: $(round(t_end-t_start, digits=3))s")
    
    A_learned = vec_to_matrix(minx, nz_off_diag, N_states)
    
    # Validate
    col_sums = vec(sum(A_learned, dims=1))
    max_col_sum_error = maximum(abs.(col_sums))
    p_test = expv(0.1, sparse(A_learned), ones(N_states)/N_states)
    prob_conservation = sum(p_test)
    
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
