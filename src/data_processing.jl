# File: src/data_processing.jl
"""
Data processing utilities for CME inverse problems.
"""

using LinearAlgebra
using SparseArrays

"""
    compute_histograms(trajectories, dt; t_max=50.0)

Compute probability histograms from SSA trajectories.

Returns:
- hists: Vector of dictionaries mapping states to probabilities
- transitions: Dictionary tracking state-to-state transitions (aggregated across all time)
"""
function compute_histograms(trajectories, dt; t_max=50.0)
    n_bins = Int(ceil(t_max / dt))
    hists = [Dict{Vector{Int}, Float64}() for _ in 1:n_bins]
    
    # Aggregate transitions across ALL time (for stoichiometry extraction)
    transitions = Dict{Tuple{Vector{Int}, Vector{Int}}, Int}()
    
    for traj in trajectories
        for i in 1:(length(traj.t)-1)
            bin = min(Int(floor(traj.t[i] / dt)) + 1, n_bins)
            s_from, s_to = traj.u[i], traj.u[i+1]
            
            # Accumulate histogram
            hists[bin][s_from] = get(hists[bin], s_from, 0.0) + 1.0
            
            # Track transition (global)
            transitions[(s_from, s_to)] = get(transitions, (s_from, s_to), 0) + 1
        end
    end
    
    # Normalize histograms
    for hist in hists
        total = sum(values(hist))
        if total > 0
            for k in keys(hist)
                hist[k] /= total
            end
        end
    end
    
    return hists, transitions
end

"""
    extract_stoichiometry(transitions; min_count=50)

Extract stoichiometric vectors from observed transitions.

Parameters:
- transitions: Dict mapping (state_from, state_to) → count
- min_count: Minimum number of observations to include a reaction

Returns:
- Vector of stoichiometric vectors (ν = state_to - state_from), sorted by frequency
"""
function extract_stoichiometry(transitions; min_count=50)
    # Aggregate by stoichiometric vector
    stoich_counts = Dict{Vector{Int}, Int}()
    
    for ((s_from, s_to), count) in transitions
        nu = s_to .- s_from
        stoich_counts[nu] = get(stoich_counts, nu, 0) + count
    end
    
    # Filter by minimum count and remove null reaction
    valid = filter(p -> p[2] >= min_count && any(p[1] .!= 0), stoich_counts)
    
    # Sort by frequency (most common first)
    sorted = sort(collect(valid), by=x->x[2], rev=true)
    
    return [nu for (nu, _) in sorted]
end

"""
    build_state_space(hist1, hist2; method=:union, kwargs...)

Build state space using various methods.

Methods:
- :union - All visited states (simple, no truncation)
- :probability - Probability threshold + cumulative mass (basic FSP)
- :adaptive - Hybrid probability + connectivity (recommended)

For FSP methods, additional parameters:
- prob_threshold: Minimum probability to include (default: 1e-4)
- cumulative_mass: Fraction of probability mass to capture (default: 0.99)
- transitions: Transition data for connectivity (optional)
- ensure_connectivity: Add boundary states (default: true)
"""
function build_state_space(hist1, hist2; 
                          method=:union,
                          prob_threshold=1e-4,
                          cumulative_mass=0.99,
                          transitions=nothing,
                          ensure_connectivity=true)
    
    if method == :union
        # Simple union (original method)
        states = union(keys(hist1), keys(hist2))
        return sort(collect(states))
        
    elseif method == :probability
        return build_state_space_fsp(hist1, hist2,
                                     prob_threshold=prob_threshold,
                                     cumulative_mass=cumulative_mass)
        
    elseif method == :adaptive
        return build_state_space_adaptive(hist1, hist2, transitions,
                                          prob_threshold=prob_threshold,
                                          cumulative_mass=cumulative_mass,
                                          ensure_connectivity=ensure_connectivity)
    else
        error("Unknown method: $method. Use :union, :probability, or :adaptive")
    end
end

"""
    build_state_space_fsp(hist1, hist2; prob_threshold=1e-4, cumulative_mass=0.99)

Build state space using Finite State Projection principles.

Keeps states that satisfy BOTH:
1. Probability threshold: p(x) > prob_threshold
2. Cumulative mass: smallest set covering cumulative_mass of probability
"""
function build_state_space_fsp(hist1, hist2; 
                               prob_threshold=1e-4, 
                               cumulative_mass=0.99)
    # Combine histograms
    combined = Dict{Vector{Int}, Float64}()
    
    for (state, prob) in hist1
        combined[state] = get(combined, state, 0.0) + prob
    end
    for (state, prob) in hist2
        combined[state] = get(combined, state, 0.0) + prob
    end
    
    # Normalize
    total = sum(values(combined))
    for state in keys(combined)
        combined[state] /= total
    end
    
    # Method 1: Probability threshold
    states_threshold = Set(state for (state, prob) in combined if prob > prob_threshold)
    
    # Method 2: Cumulative mass
    sorted_states = sort(collect(combined), by=x->x[2], rev=true)
    
    cumsum_prob = 0.0
    states_cumulative = Set{Vector{Int}}()
    
    for (state, prob) in sorted_states
        push!(states_cumulative, state)
        cumsum_prob += prob
        
        if cumsum_prob >= cumulative_mass
            break
        end
    end
    
    # Take intersection (states must pass BOTH criteria)
    state_space = intersect(states_threshold, states_cumulative)
    
    # Fallback if intersection is empty
    if isempty(state_space)
        @warn "Probability threshold too strict, using cumulative mass only"
        state_space = states_cumulative
    end
    
    state_space_vec = sort(collect(state_space))
    
    println("  FSP truncation: $(length(combined)) total → $(length(state_space_vec)) kept")
    println("    Probability threshold: $(length(states_threshold)) states")
    println("    Cumulative mass ($cumulative_mass): $(length(states_cumulative)) states")
    
    return state_space_vec
end

"""
    build_state_space_adaptive(hist1, hist2, transitions; kwargs...)

Adaptive FSP ensuring:
1. High-probability states included
2. Connectivity preserved (no isolated states)
3. Cumulative mass requirement met
"""
function build_state_space_adaptive(hist1, hist2, transitions;
                                    prob_threshold=1e-4,
                                    cumulative_mass=0.99,
                                    ensure_connectivity=true)
    # Combine histograms
    combined = Dict{Vector{Int}, Float64}()
    
    for (state, prob) in hist1
        combined[state] = get(combined, state, 0.0) + prob
    end
    for (state, prob) in hist2
        combined[state] = get(combined, state, 0.0) + prob
    end
    
    # Normalize
    total = sum(values(combined))
    for state in keys(combined)
        combined[state] /= total
    end
    
    # Core states (cumulative mass)
    sorted_states = sort(collect(combined), by=x->x[2], rev=true)
    
    core_states = Set{Vector{Int}}()
    cumsum_prob = 0.0
    
    for (state, prob) in sorted_states
        push!(core_states, state)
        cumsum_prob += prob
        
        if cumsum_prob >= cumulative_mass
            break
        end
    end
    
    println("  Core states ($cumulative_mass mass): $(length(core_states))")
    
    # Add threshold states
    threshold_states = Set(state for (state, prob) in combined if prob > prob_threshold)
    state_space = union(core_states, threshold_states)
    
    println("  + Threshold states: $(length(threshold_states)) → $(length(state_space)) total")
    
    # Ensure connectivity (if transitions available)
    if ensure_connectivity && transitions !== nothing
        boundary_states = Set{Vector{Int}}()
        
        for ((from_state, to_state), count) in transitions
            # If one state is in, check if we should add the other
            if from_state in state_space && !(to_state in state_space)
                # Add to_state if it has non-negligible probability
                if get(combined, to_state, 0.0) > prob_threshold / 10
                    push!(boundary_states, to_state)
                end
            elseif to_state in state_space && !(from_state in state_space)
                if get(combined, from_state, 0.0) > prob_threshold / 10
                    push!(boundary_states, from_state)
                end
            end
        end
        
        state_space = union(state_space, boundary_states)
        println("  + Boundary states (connectivity): $(length(boundary_states)) → $(length(state_space)) total")
    end
    
    state_space_vec = sort(collect(state_space))
    
    # Diagnostic: check captured mass
    captured_mass = sum(get(combined, state, 0.0) for state in state_space_vec)
    println("  Final: $(length(state_space_vec)) states, $(round(100*captured_mass, digits=2))% mass")
    
    return state_space_vec
end
