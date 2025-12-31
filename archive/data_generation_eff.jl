"""
Memory-efficient data generation for inverse CME problems.

Key changes:
1. Stream trajectories instead of storing all in memory
2. Build distributions incrementally
3. Downsample trajectories to fixed time grid
"""

using Catalyst
using JumpProcesses
using Random

"""
    generate_ssa_data_streaming(rn, u0, tspan, ps, n_trajectories; 
                                time_grid, seed=1234)

Generate SSA trajectories but only keep downsampled points on time_grid.
This avoids storing full high-resolution trajectories.

Returns: Vector of downsampled trajectories
"""
function generate_ssa_data_streaming(
    rn::ReactionSystem,
    u0::Vector{Pair{Symbol, Int}},
    tspan::Tuple{Float64, Float64},
    ps::Vector{Pair{Symbol, Float64}},
    n_trajectories::Int;
    time_grid::Vector{Float64},
    seed::Int = 1234
)
    Random.seed!(seed)
    
    # Preallocate output
    downsampled_trajs = Vector{Any}(undef, n_trajectories)
    
    # Generate trajectories one at a time
    println("Generating $n_trajectories trajectories (streaming mode)...")
    prog_interval = max(1, n_trajectories ÷ 10)
    
    for i in 1:n_trajectories
        # Generate one trajectory
        dprob = DiscreteProblem(rn, u0, tspan, ps)
        jprob = JumpProblem(rn, dprob, Direct())
        sol = solve(jprob, SSAStepper(), saveat=time_grid)
        
        # Store only downsampled version
        downsampled_trajs[i] = sol
        
        # Progress
        if i % prog_interval == 0
            println("  Progress: $i/$n_trajectories ($(round(i/n_trajectories*100, digits=1))%)")
        end
        
        # Explicit garbage collection every 1000 trajectories
        if i % 1000 == 0
            GC.gc()
        end
    end
    
    println("✓ Generated $n_trajectories trajectories")
    return downsampled_trajs
end

"""
    generate_distributions_directly(rn, u0, tspan, ps, n_trajectories;
                                    dt_snapshot, seed)

Generate distributions directly without storing full trajectories.
Most memory-efficient option.
"""
function generate_distributions_directly(
    rn::ReactionSystem,
    u0::Vector{Pair{Symbol, Int}},
    tspan::Tuple{Float64, Float64},
    ps::Vector{Pair{Symbol, Float64}},
    n_trajectories::Int;
    dt_snapshot::Float64 = 0.1,
    seed::Int = 1234
)
    Random.seed!(seed)
    
    # Time grid
    time_grid = collect(tspan[1]:dt_snapshot:tspan[2])
    n_times = length(time_grid)
    
    # Initialize distribution storage
    # distributions[t] = Dict(state => count)
    distributions = [Dict{Vector{Int}, Int}() for _ in 1:n_times]
    
    println("Generating $n_trajectories trajectories (direct to distributions)...")
    prog_interval = max(1, n_trajectories ÷ 10)
    
    for i in 1:n_trajectories
        # Generate trajectory on exact time grid
        dprob = DiscreteProblem(rn, u0, tspan, ps)
        jprob = JumpProblem(rn, dprob, Direct())
        sol = solve(jprob, SSAStepper(), saveat=time_grid)
        
        # Add to distributions
        for (t_idx, t) in enumerate(time_grid)
            if t_idx <= length(sol.t)
                state = Int.(sol.u[t_idx])
                distributions[t_idx][state] = get(distributions[t_idx], state, 0) + 1
            end
        end
        
        # Progress
        if i % prog_interval == 0
            println("  Progress: $i/$n_trajectories ($(round(i/n_trajectories*100, digits=1))%)")
            GC.gc()  # Force garbage collection
        end
    end
    
    # Convert counts to probabilities
    prob_distributions = Vector{Dict{Vector{Int}, Float64}}()
    for counts in distributions
        prob_dist = Dict{Vector{Int}, Float64}()
        total = sum(values(counts))
        if total > 0
            for (state, count) in counts
                prob_dist[state] = count / total
            end
        end
        push!(prob_distributions, prob_dist)
    end
    
    println("✓ Generated distributions from $n_trajectories trajectories")
    return time_grid, prob_distributions
end

"""
    generate_in_batches(rn, u0, tspan, ps, n_trajectories;
                       batch_size, time_grid, seed)

Generate trajectories in batches to control memory usage.
"""
function generate_in_batches(
    rn::ReactionSystem,
    u0::Vector{Pair{Symbol, Int}},
    tspan::Tuple{Float64, Float64},
    ps::Vector{Pair{Symbol, Float64}},
    n_trajectories::Int;
    batch_size::Int = 1000,
    time_grid::Vector{Float64},
    seed::Int = 1234
)
    n_batches = ceil(Int, n_trajectories / batch_size)
    all_trajs = Vector{Any}()
    
    println("Generating $n_trajectories trajectories in $n_batches batches...")
    
    for batch in 1:n_batches
        batch_start = (batch - 1) * batch_size + 1
        batch_end = min(batch * batch_size, n_trajectories)
        batch_n = batch_end - batch_start + 1
        
        println("  Batch $batch/$n_batches: trajectories $batch_start-$batch_end")
        
        # Generate this batch
        batch_trajs = generate_ssa_data_streaming(
            rn, u0, tspan, ps, batch_n;
            time_grid = time_grid,
            seed = seed + batch
        )
        
        append!(all_trajs, batch_trajs)
        
        # Force garbage collection between batches
        GC.gc()
    end
    
    return all_trajs
end

"""
    estimate_memory_usage(n_trajectories, tspan, dt_snapshot)

Estimate memory requirements before generation.
"""
function estimate_memory_usage(n_trajectories::Int, tspan::Tuple{Float64,Float64}, 
                               dt_snapshot::Float64)
    n_timepoints = length(collect(tspan[1]:dt_snapshot:tspan[2]))
    n_species = 4  # Assume typical
    
    # Bytes per trajectory (rough estimate)
    bytes_per_point = 8 * n_species  # Float64 for each species
    bytes_per_traj = bytes_per_point * n_timepoints
    total_bytes = bytes_per_traj * n_trajectories
    
    # Convert to human-readable
    total_mb = total_bytes / 1e6
    total_gb = total_bytes / 1e9
    
    println("Memory estimate:")
    println("  Trajectories: $n_trajectories")
    println("  Timepoints per trajectory: $n_timepoints")
    println("  Total data points: $(n_trajectories * n_timepoints)")
    println("  Estimated memory: $(round(total_mb, digits=1)) MB ($(round(total_gb, digits=2)) GB)")
    
    if total_gb > 4.0
        println("  ⚠️  WARNING: May exceed available RAM!")
        println("  ➡️  Recommendation: Use batch_size=1000 or generate_distributions_directly()")
    end
    
    return total_gb
end

# Export functions
export generate_ssa_data_streaming
export generate_distributions_directly
export generate_in_batches
export estimate_memory_usage
