"""
CRN-SINDy: Structure Learning for Chemical Reaction Networks.
Infers Topology (Stoichiometry) and Rates from Trajectory Data.
"""

using Catalyst
using Statistics
using LinearAlgebra
using Random
using Combinatorics

# ============================================================================
# 1. GENERATE DATA (Hidden Truth)
# ============================================================================
# We use the same MM system, but the algorithm won't "know" this.
rn_true = @reaction_network begin
    k1, S + E --> SE
    k2, SE --> S + E
    k3, SE --> P + E
end

u0 = [:S => 50.0, :E => 10.0, :SE => 0.0, :P => 0.0]
true_rates = [0.01, 0.1, 0.1]
tspan = (0.0, 50.0)

# Generate noisy trajectory
prob = DiscreteProblem(rn_true, u0, tspan, true_rates)
jump_prob = JumpProblem(rn_true, prob, Direct())
sol = solve(jump_prob, SSAStepper(), saveat=0.5) # Sparse sampling

println("Data Generated: $(length(sol.t)) time points")

# ============================================================================
# 2. BUILD CANDIDATE LIBRARY (The "Unknown Topology" Step)
# ============================================================================
species_names = ["S", "E", "SE", "P"]
n_species = 4

# Generate all possible reactants up to order 2
# 1. Null (0 order) - usually ignored for closed systems
# 2. Uni-molecular (S, E, SE, P)
# 3. Bi-molecular (S+S, S+E, S+SE, ..., P+P)

candidates = [] # Stores (stoich_vector, order_vector, name)

# Uni-molecular candidates (e.g., S -> ?)
for i in 1:n_species
    order_vec = zeros(Int, n_species)
    order_vec[i] = 1
    name = species_names[i]
    push!(candidates, (order_vec, name))
end

# Bi-molecular candidates (e.g., S+E -> ?)
for i in 1:n_species
    for j in i:n_species # Symmetric (S+E = E+S)
        order_vec = zeros(Int, n_species)
        order_vec[i] += 1
        order_vec[j] += 1
        name = "$(species_names[i]) + $(species_names[j])"
        push!(candidates, (order_vec, name))
    end
end

println("Candidate Library Size: $(length(candidates)) potential reaction channels")
for (i, c) in enumerate(candidates)
    println("  $i: $(c[2])")
end

# ============================================================================
# 3. CONSTRUCT FEATURE MATRIX (INTEGRAL WINDOWS)
# ============================================================================
# We solve for the net flow of each species.
# dS/dt = sum(k_r * change_S_r * propensity_r)
# We don't know change_S_r. 
# Wait - SINDy usually assumes we know the LHS (dS/dt).
# But CRN SINDy is harder: we don't know which reaction produces which output.

# SIMPLIFIED APPROACH:
# We assume we are learning the "Net Production Rate" of each species 
# as a linear combination of the candidate propensities.
# dS/dt = θ(x) * Ξ_S
# where Ξ_S is a sparse vector telling us how much S is produced/consumed by each candidate.

n_windows = length(sol.t) - 1
n_candidates = length(candidates)

# A matrix: Integrated Propensities (Windows x Candidates)
A = zeros(n_windows, n_candidates)

# B matrix: Net Change in Species (Windows x Species)
B = zeros(n_windows, n_species)

# Helper to eval basis
function eval_propensity(state, order_vec)
    p = 1.0
    for i in 1:length(state)
        p *= state[i]^order_vec[i]
    end
    return p
end

println("\nBuilding Integral Feature Matrix...")
for i in 1:n_windows
    dt = sol.t[i+1] - sol.t[i]
    x_curr = sol[i]
    x_next = sol[i+1]
    
    # Fill B (LHS: Δx)
    B[i, :] = x_next - x_curr
    
    # Fill A (RHS: ∫ Candidate dt)
    # Use Trapezoidal rule for integration
    for (j, cand) in enumerate(candidates)
        prop_start = eval_propensity(x_curr, cand[1])
        prop_end   = eval_propensity(x_next, cand[1])
        A[i, j] = 0.5 * (prop_start + prop_end) * dt
    end
end

# ============================================================================
# 4. SPARSE REGRESSION (STLSQ)
# ============================================================================
# We solve A * Ξ = B
# Ξ is (Candidates x Species). Ξ[j, s] = Stoichiometry of species s in reaction j * Rate j
# This is tricky: Rate is positive, Stoich is integer.
# But for identification, we just need the non-zero entries.

println("\nPerforming Sparse Regression (STLSQ)...")

function stlsq(A, B; threshold=0.1, max_iter=10)
    Xi = A \ B # Initial Least Squares
    
    for k in 1:max_iter
        small_inds = abs.(Xi) .< threshold
        Xi[small_inds] .= 0.0
        
        # Refit only non-zero elements
        for j in 1:size(B, 2) # For each species
            big_inds = .!small_inds[:, j]
            if sum(big_inds) > 0
                Xi[big_inds, j] = A[:, big_inds] \ B[:, j]
            end
        end
    end
    return Xi
end

# Threshold needs tuning based on noise/scale. 
# Since rates are ~0.1 and dt~0.5, changes are ~0.05. 
# Threshold should be small enough to catch 0.01 rates.
Xi = stlsq(A, B; threshold=0.005) 

# ============================================================================
# 5. INTERPRET RESULTS
# ============================================================================
println("\n" * "="^60)
println("INFERRED TOPOLOGY")
println("="^60)

# Xi[row, col] means: Candidate [row] affects Species [col] with coefficient C.
species_labels = ["S", "E", "SE", "P"]

for (j, cand) in enumerate(candidates)
    coeffs = Xi[j, :]
    if norm(coeffs) > 1e-5
        # This candidate is active!
        
        # Infer Stoichiometry from signs
        # E.g., if S consumes (-0.01) and SE produces (+0.01), 
        # it implies reaction S... -> SE with rate 0.01.
        
        rate_est = maximum(abs.(coeffs))
        
        # Build reaction string
        reactants = cand[2]
        products = []
        
        # Check net change of each species
        for s in 1:n_species
            c = coeffs[s]
            if abs(c) > 1e-5
                # Normalize by rate to get stoichiometry integer
                stoich_coeff = round(Int, c / rate_est)
                
                # If negative, it was consumed (already in reactants)
                # If positive, it is PRODUCED
                if stoich_coeff > 0
                    push!(products, "$stoich_coeff $(species_labels[s])")
                elseif stoich_coeff < 0
                    # Check if consumption matches reactant definition
                    # (This is a simplified check)
                end
            end
        end
        
        # If reactants disappear (-1) and nothing appears, it's -> 0
        prod_str = isempty(products) ? "∅" : join(products, " + ")
        
        println("Reaction Detected:")
        println("  $reactants → $prod_str")
        println("  Rate: $(round(rate_est, sigdigits=3))")
        println("  Coeffs: $(round.(coeffs, sigdigits=2))")
        println("-"^40)
    end
end
