"""
Functions for building local state spaces with connectivity preservation.
"""

using SparseArrays

"""
    build_global_state_space(all_distributions, stoich_vecs; connectivity_depth=2)

Build a comprehensive global state space from all distributions across all time windows.
This is built once at the start and used for rate extraction.

# Arguments
- `all_distributions`: Vector of all distribution dictionaries (across all windows)
- `stoich_vecs`: Vector of stoichiometry vectors
- `connectivity_depth`: How many reaction steps to expand for connectivity (default: 2)

# Returns
Set of states (as CartesianIndex) forming the global state space
"""
function build_global_state_space(
    all_distributions,
    stoich_vecs;
    connectivity_depth=2
)
    # Collect all states that appear in any distribution
    all_states = Set{Tuple}()
    
    for dist in all_distributions
        for (state, prob) in dist
            state_tuple = isa(state, Tuple) ? state : tuple(state...)
            push!(all_states, state_tuple)
        end
    end
    
    println("  States observed in data: $(length(all_states))")
    
    # Expand for connectivity - add reachable neighbors
    expanded_states = copy(all_states)
    
    for depth in 1:connectivity_depth
        new_states = Set{Tuple}()
        
        for state in expanded_states
            for ν in stoich_vecs
                state_vec = collect(state)
                
                # Forward neighbor
                neighbor_forward = tuple((state_vec .+ ν)...)
                if all(neighbor_forward .>= 0)
                    push!(new_states, neighbor_forward)
                end
                
                # Backward neighbor
                neighbor_backward = tuple((state_vec .- ν)...)
                if all(neighbor_backward .>= 0)
                    push!(new_states, neighbor_backward)
                end
            end
        end
        
        expanded_states = union(expanded_states, new_states)
        println("  After depth $depth expansion: $(length(expanded_states)) states")
    end
    
    # Convert to CartesianIndex
    global_states = Set(CartesianIndex(state...) for state in expanded_states)
    
    return global_states
end

"""
    reaction_direction(stoich_vec, X)

Build perturbation direction matrix E for a reaction with given stoichiometry.

# Arguments
- `stoich_vec`: Stoichiometry vector (change in state counts)
- `X`: State space (collection of states as CartesianIndex)

# Returns
Sparse matrix E where E[i,j] indicates transition from state j to state i
"""
function reaction_direction(stoich_vec, X)
    n = length(X)
    E = spzeros(n, n)
    
    X_vec = collect(X)  # Convert to vector for indexing
    
    for (idx_from, x_from) in enumerate(X_vec)
        x_to = Tuple(collect(Tuple(x_from)) .+ stoich_vec)
        x_to_cart = CartesianIndex(x_to...)
        
        if x_to_cart in X
            idx_to = findfirst(==(x_to_cart), X_vec)
            E[idx_to, idx_from] = 1.0
            E[idx_from, idx_from] -= 1.0
        end
    end
    
    return E
end

"""
    build_local_state_space_with_connectivity(window_dists, stoich_vecs, mass_threshold)

Build a local state space that includes high-probability states and their neighbors.

This ensures the local state space is connected with respect to the reaction network.

# Arguments
- `window_dists`: Vector of distributions (dicts) in the window
- `stoich_vecs`: Vector of stoichiometry vectors
- `mass_threshold`: Fraction of probability mass to retain (default: 0.95)

# Returns
Set of states (as tuples) forming the connected local state space
"""
function build_local_state_space_with_connectivity(
    window_dists, 
    stoich_vecs, 
    mass_threshold=0.95
)
    # Step 1: Aggregate probabilities across all distributions in window
    state_probs = Dict()
    
    for dist in window_dists
        for (state, prob) in dist
            # Ensure state is a tuple
            state_tuple = isa(state, Tuple) ? state : tuple(state...)
            state_probs[state_tuple] = get(state_probs, state_tuple, 0.0) + prob
        end
    end
    
    # Normalize
    total = sum(values(state_probs))
    for (state, prob) in state_probs
        state_probs[state] = prob / total
    end
    
    # Step 2: Select core states by probability threshold
    sorted_states = sort(collect(state_probs), by=x->x[2], rev=true)
    
    cumulative_mass = 0.0
    core_states = Set{Tuple}()
    
    for (state, prob) in sorted_states
        push!(core_states, state)
        cumulative_mass += prob
        if cumulative_mass >= mass_threshold
            break
        end
    end
    
    println("  Core states (high probability): $(length(core_states))")
    
    # Step 3: Add neighbor states for ALL reactions to ensure connectivity
    expanded_states = copy(core_states)
    
    for state in core_states
        for ν in stoich_vecs
            state_vec = collect(state)
            
            # Forward neighbor: state + ν
            neighbor_forward = tuple((state_vec .+ ν)...)
            if all(neighbor_forward .>= 0)
                push!(expanded_states, neighbor_forward)
            end
            
            # Backward neighbor: state - ν  
            neighbor_backward = tuple((state_vec .- ν)...)
            if all(neighbor_backward .>= 0)
                push!(expanded_states, neighbor_backward)
            end
        end
    end
    
    println("  Expanded states (with neighbors): $(length(expanded_states))")
    
    return expanded_states
end

"""
    build_sparsity_pattern(E_matrices, N_states)

Build sparsity pattern for generator matrix from reaction direction matrices.

# Arguments
- `E_matrices`: Vector of reaction direction matrices
- `N_states`: Number of states

# Returns
- `sparsity_pattern`: Boolean matrix indicating non-zero entries
- `nz_off_diag`: Vector of CartesianIndex for off-diagonal non-zero entries
"""
function build_sparsity_pattern(E_matrices, N_states)
    sparsity_pattern = zeros(Bool, N_states, N_states)
    
    # Add entries from all reaction directions
    for E in E_matrices
        sparsity_pattern .|= (abs.(E) .> 1e-14)
    end
    
    # Diagonal entries are always part of the pattern
    for i in 1:N_states
        sparsity_pattern[i, i] = true
    end
    
    # Extract off-diagonal indices
    nz_off_diag = filter(idx -> idx[1] != idx[2], findall(sparsity_pattern))
    
    return sparsity_pattern, nz_off_diag
end

"""
    extract_local_distributions(window_dists, X_local_vec)

Extract probability vectors for local state space from full distributions.

# Arguments
- `window_dists`: Vector of distribution dictionaries
- `X_local_vec`: Vector of local states (CartesianIndex)

# Returns
Vector of probability vectors (one per time snapshot)
"""
function extract_local_distributions(window_dists, X_local_vec)
    N_states = length(X_local_vec)
    local_dists = []
    
    for dist in window_dists
        local_dist = zeros(N_states)
        
        for (idx, state_cart) in enumerate(X_local_vec)
            state_key = collect(Tuple(state_cart))
            local_dist[idx] = get(dist, state_key, 0.0)
        end
        
        # Normalize
        local_dist ./= (sum(local_dist) + 1e-10)
        push!(local_dists, local_dist)
    end
    
    return local_dists
end

"""
    verify_connectivity(E_matrices)

Check that reaction direction matrices have non-trivial transitions.
"""
function verify_connectivity(E_matrices)
    all_connected = true
    
    for (j, E) in enumerate(E_matrices)
        nz = nnz(sparse(E))
        if nz == 0
            @warn "Reaction $j has no transitions in local state space!"
            all_connected = false
        else
            println("  Reaction $j: $nz non-zero entries")
        end
    end
    
    return all_connected
end
