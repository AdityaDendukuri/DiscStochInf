"""
Functions for generating SSA trajectory data and converting to distributions.
"""

using JumpProcesses
using StatsBase
using ProgressLogging

"""
    merge_hist(A1::Dict, A2::Dict) -> Dict

Merge two histogram dictionaries by summing counts for common keys.
"""
function merge_hist(A1::Dict, A2::Dict)
    if isempty(A1)
        return A2
    end
    if isempty(A2)
        return A1
    end
    
    ks = keys(merge(A1, A2))
    dict = Dict{AbstractVector, Float64}()
    
    for k in ks
        dict[k] = 0.0
        if k in keys(A1)
            dict[k] += A1[k]
        end
        if k in keys(A2)
            dict[k] += A2[k]
        end
    end
    
    return dict
end

"""
    gen_dist(trajs, tspan) -> Dict

Generate empirical distribution from trajectories within a time window.

# Arguments
- `trajs`: Vector of trajectory solutions
- `tspan`: Tuple (t_start, t_end) defining the time window

# Returns
Dictionary mapping states (as vectors) to their empirical probabilities
"""
function gen_dist(trajs, tspan)
    A = Dict{AbstractVector, Float64}()
    
    for traj in trajs
        idxs = findall(tspan[1] .< traj.t .< tspan[2])
        A = merge_hist(A, countmap(traj.u[idxs]))
    end
    
    # Normalize by number of trajectories
    for a in keys(A)
        A[a] = A[a] / length(trajs)
    end
    
    return A
end

"""
    convert_to_distributions(trajs, tspan, dt) -> (times, distributions)

Convert trajectories to a sequence of distributions at regular time intervals.

# Arguments
- `trajs`: Vector of trajectory solutions
- `tspan`: Tuple (t_start, t_end) for the overall time range
- `dt`: Time interval between snapshots

# Returns
- `times`: Vector of time points
- `distributions`: Vector of distribution dictionaries at each time
"""
function convert_to_distributions(trajs, tspan, dt)
    T = tspan[1]:dt:tspan[end]
    distrbs = []
    
    for i in 1:length(T)-1
        push!(distrbs, gen_dist(trajs, (T[i], T[i+1])))
    end
    
    return T, distrbs
end

"""
    pad_distributions(distributions, state_space) -> Vector{Dict}

Pad distributions to include all states in state_space with zero probability for missing states.

# Arguments
- `distributions`: Vector of distribution dictionaries
- `state_space`: Iterable of states (typically CartesianIndex)

# Returns
Vector of padded distribution dictionaries where all states have entries
"""
function pad_distributions(distributions, state_space)
    # Create template dictionary with all states initialized to zero
    template = Dict()
    for x in state_space
        template[collect(Tuple(x))] = 0.0
    end
    
    padded_dists = []
    for dist in distributions
        d = copy(template)
        for (state, prob) in dist
            d[state] = prob
        end
        push!(padded_dists, d)
    end
    
    return padded_dists
end

"""
    generate_ssa_data(rn, u0, tspan, ps, n_trajs; seed=nothing)

Generate SSA trajectory data for a reaction network.

# Arguments
- `rn`: Reaction network (from Catalyst)
- `u0`: Initial condition (vector of species counts or symbol pairs)
- `tspan`: Time span tuple (t_start, t_end)
- `ps`: Parameters (vector or symbol pairs)
- `n_trajs`: Number of trajectories to generate
- `seed`: Random seed (optional)

# Returns
Vector of trajectory solutions
"""
function generate_ssa_data(rn, u0, tspan, ps, n_trajs; seed=nothing)
    
    # Setup jump problem
    jinput = JumpInputs(rn, u0, tspan, ps)
    jprob = JumpProblem(jinput)
    
    # Test solve
    if !isnothing(seed)
        test_sol = solve(jprob, SSAStepper(); seed=seed)
        println("Test trajectory: $(length(test_sol.t)) time points")
    end
    
    # Generate trajectories
    ssa_trajs = []
    @progress for i in 1:n_trajs
        push!(ssa_trajs, solve(jprob, SSAStepper()))
    end
    
    return ssa_trajs
end

"""
    infer_reactions_from_trajectories(trajs) -> Vector{Vector{Int}}

Infer stoichiometry vectors by observing state changes in trajectories.

# Arguments
- `trajs`: Vector of trajectory solutions

# Returns
Vector of stoichiometry vectors (sorted by frequency of occurrence)
"""
function infer_reactions_from_trajectories(trajs)
    # Collect all observed state changes
    stoich_counts = Dict{Vector{Int}, Int}()
    
    for traj in trajs
        for i in 1:length(traj.t)-1
            Δ = collect(traj.u[i+1]) - collect(traj.u[i])
            stoich_counts[Δ] = get(stoich_counts, Δ, 0) + 1
        end
    end
    
    # Filter out zero changes and sort by frequency
    reactions = filter(p -> p[1] != zeros(Int, length(p[1])), stoich_counts)
    sorted_reactions = sort(collect(reactions), by=x->x[2], rev=true)
    
    return [r[1] for r in sorted_reactions]
end

"""
    create_windows(times, distributions, config::InverseProblemConfig, stoich_vecs)

Create WindowData objects for sliding window analysis.

# Returns
Vector of WindowData objects
"""
function create_windows(times, distributions, config::InverseProblemConfig, stoich_vecs)
    n_windows = Int(floor((times[end] - times[1]) / config.dt_window))
    n_windows = min(n_windows, config.max_windows)
    
    windows = WindowData[]
    
    for window_idx in 1:n_windows
        start_idx = (window_idx - 1) * config.snapshots_per_window + 1
        end_idx = min(window_idx * config.snapshots_per_window, length(distributions))
        
        window_dists = distributions[start_idx:end_idx]
        window_times = times[start_idx:end_idx]
        
        push!(windows, WindowData(window_idx, window_dists, window_times, stoich_vecs))
    end
    
    return windows
end
