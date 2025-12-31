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
- transitions: Vector of dictionaries tracking state-to-state transitions
"""
function compute_histograms(trajectories, dt; t_max=50.0)
    n_bins = Int(ceil(t_max / dt))
    hists = [Dict{Vector{Int}, Float64}() for _ in 1:n_bins]
    transitions = [Dict{Tuple{Vector{Int}, Vector{Int}}, Int}() for _ in 1:n_bins]
    
    for traj in trajectories
        for i in 1:(length(traj.t)-1)
            bin = min(Int(floor(traj.t[i] / dt)) + 1, n_bins)
            s_from, s_to = traj.u[i], traj.u[i+1]
            
            # Accumulate histogram
            hists[bin][s_from] = get(hists[bin], s_from, 0.0) + 1.0
            
            # Track transition
            transitions[bin][(s_from, s_to)] = get(transitions[bin], (s_from, s_to), 0) + 1
        end
    end
    
    # Normalize histograms
    for hist in hists
        total = sum(values(hist))
        total > 0 && foreach(k -> hist[k] /= total, keys(hist))
    end
    
    return hists, transitions
end

"""
    extract_stoichiometry(transitions; min_count=50)

Extract stoichiometric vectors from observed transitions.
Returns vectors sorted by frequency (most common first).
"""
function extract_stoichiometry(transitions; min_count=50)
    # Aggregate transitions across all windows
    all_trans = Dict{Vector{Int}, Int}()
    for trans_dict in transitions
        for ((s_from, s_to), count) in trans_dict
            nu = s_to .- s_from
            all_trans[nu] = get(all_trans, nu, 0) + count
        end
    end
    
    # Filter and sort by frequency
    valid = filter(p -> p[2] >= min_count && p[1] != [0,0], all_trans)
    sorted = sort(collect(valid), by=x->x[2], rev=true)
    
    return [nu for (nu, _) in sorted]
end

"""
    build_state_space(hist1, hist2)

Build joint state space from two consecutive histograms.
"""
function build_state_space(hist1, hist2)
    states = union(keys(hist1), keys(hist2))
    return sort(collect(states))
end
