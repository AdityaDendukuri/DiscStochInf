# File: src/generator_construction.jl

"""
Generator construction and parameter mapping.
"""

using LinearAlgebra
using SparseArrays

struct InverseProblemData{T<:Real}
    state_space::Vector{Vector{Int}}
    state_features::Matrix{T}
    stoich_basis::Vector{Vector{Int}}
    n_features::Int
    dt::T
    
    # Cached mappings
    state_to_idx::Dict{Vector{Int}, Int}
    transition_map::Dict{Tuple{Int,Int}, Int}  # (state_idx, reaction_idx) -> target_state_idx
end

"""
    build_inverse_problem_data(state_space, stoich_basis, dt; n_features=6)

Build data structure for inverse problem.
"""
function build_inverse_problem_data(state_space, stoich_basis, dt; n_features=6)
    n_states = length(state_space)
    state_to_idx = Dict(s => i for (i, s) in enumerate(state_space))
    
    # Compute features for each state
    state_features = compute_state_features(state_space, n_features)
    
    # Build transition map
    transition_map = Dict{Tuple{Int,Int}, Int}()
    for (i, state) in enumerate(state_space)
        for (k, nu) in enumerate(stoich_basis)
            target = state .+ nu
            if haskey(state_to_idx, target)
                transition_map[(i, k)] = state_to_idx[target]
            end
        end
    end
    
    return InverseProblemData(
        state_space, state_features, stoich_basis, 
        n_features, dt, state_to_idx, transition_map
    )
end

"""
    compute_state_features(state_space, n_features)

Compute polynomial features for each state.
Features: [1, x, y, xy, x², y²]
"""
function compute_state_features(state_space, n_features)
    n = length(state_space)
    features = zeros(n_features, n)
    
    for (i, state) in enumerate(state_space)
        x, y = state[1], state[2]
        if n_features == 3
            features[:, i] = [1.0, x, y]
        elseif n_features == 6
            features[:, i] = [1.0, x, y, x*y, x^2, y^2]
        else
            error("Only n_features=3 or 6 supported")
        end
    end
    
    return features
end

"""
    build_generator(θ, data::InverseProblemData)

Construct generator matrix from parameters.
"""
function build_generator(θ, data::InverseProblemData)
    n = length(data.state_space)
    n_reactions = length(data.stoich_basis)
    A = zeros(n, n)
    
    # Build off-diagonal entries
    for (i, state) in enumerate(data.state_space)
        for k in 1:n_reactions
            if haskey(data.transition_map, (i, k))
                j = data.transition_map[(i, k)]
                
                # Propensity = θ_k' * features(state_j)
                θ_k = θ[(k-1)*data.n_features + 1 : k*data.n_features]
                prop = max(0.0, dot(θ_k, data.state_features[:, j]))
                
                A[i, j] = prop
            end
        end
    end
    
    # Set diagonal to ensure column sums = 0
    for j in 1:n
        A[j, j] = -sum(A[:, j])
    end
    
    return A
end

"""
    build_perturbation(k, f, data::InverseProblemData)

Build perturbation matrix E_kf for parameter θ_kf.
"""
function build_perturbation(k, f, data::InverseProblemData)
    n = length(data.state_space)
    E = zeros(n, n)
    
    # Off-diagonal: feature value if transition exists
    for (i, state) in enumerate(data.state_space)
        if haskey(data.transition_map, (i, k))
            j = data.transition_map[(i, k)]
            E[i, j] = data.state_features[f, j]
        end
    end
    
    # Diagonal: negative column sum
    for j in 1:n
        E[j, j] = -sum(E[:, j])
    end
    
    return E
end
