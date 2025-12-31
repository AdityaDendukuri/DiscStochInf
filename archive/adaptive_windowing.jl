"""
Adaptive window sizing based on theoretical identifiability bounds.

Implements theoretically-guided window selection that maintains
uniqueness criterion Δt·w_eff < 0.3 by estimating effective flux
from trajectory data.
"""

push!(LOAD_PATH, @__DIR__)

using Catalyst
using JumpProcesses
using LinearAlgebra
using Printf
using JSON3
using Dates
using Statistics

# Load modules
include("types.jl")
include("data_generation.jl")
include("state_space.jl")
include("optimization.jl")
include("analysis.jl")
include("src/DiscStochSim.jl")
using .DiscStochSim

# Load base experiment infrastructure
include("experiments.jl")
include("state_space.jl")  # If not already included
# ============================================================================
# EFFECTIVE FLUX ESTIMATION
# ============================================================================

"""
    estimate_w_eff_from_trajectories(trajectories, t_start, t_end)

Estimate effective flux w_eff from empirical transition rate in time window.

The effective flux is defined as w_eff = Σᵢ p(i)·w(i) where w(i) is the
exit rate from state i. We approximate this by counting the average
number of transitions per unit time across all trajectories.

# Arguments
- `trajectories`: Vector of SSA trajectory solutions
- `t_start`: Window start time
- `t_end`: Window end time

# Returns
- Estimated w_eff (average transition rate)
"""
function estimate_w_eff_from_trajectories(
    trajectories::Vector,
    t_start::Float64,
    t_end::Float64
)
    total_transitions = 0
    total_time = 0.0
    
    for traj in trajectories
        # Find portion of trajectory in [t_start, t_end]
        idx_start = findfirst(t -> t >= t_start, traj.t)
        idx_end = findlast(t -> t <= t_end, traj.t)
        
        if idx_start !== nothing && idx_end !== nothing && idx_end > idx_start
            # Count state changes (jumps)
            n_jumps = idx_end - idx_start
            duration = traj.t[idx_end] - traj.t[idx_start]
            
            total_transitions += n_jumps
            total_time += duration
        end
    end
    
    # Average transition rate ≈ w_eff
    if total_time > 0
        return total_transitions / (total_time * length(trajectories))
    else
        return NaN
    end
end

"""
    estimate_w_eff_profile(trajectories, T, window_duration)

Estimate w_eff evolution over time using sliding probe windows.

# Arguments
- `trajectories`: Vector of SSA trajectories
- `T`: Time vector for distributions
- `window_duration`: Duration for probe windows (default: 1.0)

# Returns
- `times`: Time points where w_eff was estimated
- `w_eff_values`: Estimated w_eff at each time point
"""
function estimate_w_eff_profile(
    trajectories::Vector,
    T::Vector{Float64},
    window_duration::Float64 = 1.0
)
    times = Float64[]
    w_eff_values = Float64[]
    
    # Probe at regular intervals
    t_current = T[1]
    
    while t_current + window_duration <= T[end]
        w_est = estimate_w_eff_from_trajectories(
            trajectories,
            t_current,
            t_current + window_duration
        )
        
        if !isnan(w_est) && w_est > 0
            push!(times, t_current + window_duration/2)  # Midpoint
            push!(w_eff_values, w_est)
        end
        
        t_current += window_duration * 0.5  # 50% overlap
    end
    
    return times, w_eff_values
end

# ============================================================================
# ADAPTIVE WINDOW CREATION
# ============================================================================

"""
    create_adaptive_windows(T, distributions, base_config, stoich_vecs, trajectories)

Create windows with adaptive sizing based on estimated effective flux.

Maintains uniqueness criterion Δt·w_eff < threshold by computing
dt = target_product / w_eff(t) at each time point.

# Arguments
- `T`: Time points
- `distributions`: Probability distributions at each time
- `base_config`: Base configuration (provides other parameters)
- `stoich_vecs`: Inferred stoichiometry
- `trajectories`: SSA trajectories for w_eff estimation

# Keyword Arguments
- `target_product::Float64 = 0.25`: Target for Δt·w_eff (< 0.3)
- `dt_min::Float64 = 0.5`: Minimum window size
- `dt_max::Float64 = 10.0`: Maximum window size
- `probe_duration::Float64 = 1.0`: Duration for w_eff probes
- `verbose::Bool = true`: Print window sizing decisions
"""
function create_adaptive_windows(
    T::Vector{Float64},
    distributions::Vector,
    base_config::InverseProblemConfig,
    stoich_vecs::Vector{Vector{Int}},
    trajectories::Vector;
    target_product::Float64 = 0.25,
    dt_min::Float64 = 0.5,
    dt_max::Float64 = 10.0,
    probe_duration::Float64 = 1.0,
    verbose::Bool = true
)
    
    if verbose
        println("\n" * "="^80)
        println("ADAPTIVE WINDOW SIZING")
        println("="^80)
        println("Target: Δt·w_eff ≈ $target_product (threshold: 0.3)")
        println("Bounds: dt ∈ [$dt_min, $dt_max]")
        println()
    end
    
    windows = WindowData[]
    
    t_current = T[1]
    window_idx = 1
    
    # First, estimate w_eff profile
    if verbose
        println("Estimating w_eff profile...")
    end
    
    profile_times, profile_w_eff = estimate_w_eff_profile(
        trajectories, T, probe_duration
    )
    
  if verbose
    println("✓ Estimated w_eff at $(length(profile_times)) time points")
    println("\nw_eff evolution:")
    n_show = min(5, length(profile_times))
    for i in 1:n_show
        println("  t=$(round(profile_times[i], digits=2)): w_eff=$(round(profile_w_eff[i], sigdigits=3))")
    end
    if length(profile_times) > 5
        println("  ...")
    end
    println()
end
    
    while t_current < T[end] && window_idx <= base_config.max_windows
        # Interpolate w_eff at current time
        if isempty(profile_times)
            # Fallback to base config
            w_eff_est = NaN
        else
            # Find nearest w_eff estimate
            idx = argmin(abs.(profile_times .- t_current))
            w_eff_est = profile_w_eff[idx]
        end
        
        if isnan(w_eff_est) || w_eff_est <= 0
            # Fallback to base config
            dt_window = base_config.dt_window
            if verbose
                @warn "Window $window_idx: Could not estimate w_eff at t=$(round(t_current, digits=2)), using default dt=$dt_window"
            end
        else
            # Adaptive sizing: dt = target / w_eff
            dt_window = target_product / w_eff_est
            
            # Clamp to safety bounds
            dt_window_clamped = clamp(dt_window, dt_min, dt_max)
            
            if verbose
                clamped_str = dt_window != dt_window_clamped ? " (clamped from $(round(dt_window, digits=2)))" : ""
                uniqueness_product = dt_window_clamped * w_eff_est
                status = uniqueness_product < 0.3 ? "✓" : "⚠"
                
                println("Window $window_idx: t=$(round(t_current, digits=2)), w_eff=$(round(w_eff_est, sigdigits=3)) → dt=$(round(dt_window_clamped, digits=2))$clamped_str  [Δt·w_eff=$(round(uniqueness_product, sigdigits=3)) $status]")
            end
            
            dt_window = dt_window_clamped
        end
        
    # Find distributions in this window
t_end = min(t_current + dt_window, T[end])  # Don't go past the end
indices_in_window = findall(t -> t_current <= t <= t_end, T)

# Make sure indices are within bounds
indices_in_window = filter(i -> i <= length(distributions), indices_in_window)

if length(indices_in_window) < 2
    if verbose
        @warn "Window $window_idx: Too few snapshots (only $(length(indices_in_window))), breaking"
    end
    break
end

# Extract distributions for this window
window_times = T[indices_in_window]
window_dists = distributions[indices_in_window]
        
        # Build local state space
    # Build local state space (inline to avoid import issues)
# First, determine dimensionality from distributions
first_state = first(first(window_dists))[1]
n_species = length(first_state isa CartesianIndex ? Tuple(first_state) : first_state)
X_local = Set{CartesianIndex{n_species}}()

for dist in window_dists
    for (state, prob) in dist
        if prob > 1e-10  # Only include states with non-negligible probability
            # Convert state to CartesianIndex if it isn't already
            if state isa CartesianIndex
                push!(X_local, state)
            else
                push!(X_local, CartesianIndex(Tuple(state)))
            end
        end
    end
end

# Add reachable neighbors (connectivity)
for state in collect(X_local)
    for stoich_vec in stoich_vecs
        # Convert stoich_vec to tuple for CartesianIndex
        stoich_tuple = Tuple(stoich_vec)
        
        # Forward reaction
        new_coords_forward = Tuple(state) .+ stoich_tuple
        if all(x -> x >= 0, new_coords_forward)
            push!(X_local, CartesianIndex(new_coords_forward))
        end
        
        # Backward reaction
        new_coords_backward = Tuple(state) .- stoich_tuple
        if all(x -> x >= 0, new_coords_backward)
            push!(X_local, CartesianIndex(new_coords_backward))
        end
    end
end
        
        # Create window data (correct argument order: idx, distributions, times, stoich_vecs)
window = WindowData(
    window_idx,
    window_dists,
    window_times,
    stoich_vecs
)
        
        push!(windows, window)
        
        # Advance to next window (with overlap for continuity)
        t_current += dt_window * 0.8  # 20% overlap
        window_idx += 1
    end
    
    if verbose
        println("\n✓ Created $(length(windows)) adaptive windows")
        
        # Summary statistics
        dt_sizes = [w.times[end] - w.times[1] for w in windows]
        println("\nWindow size statistics:")
        println("  Mean: $(round(mean(dt_sizes), digits=2))")
        println("  Std:  $(round(std(dt_sizes), digits=2))")
        println("  Min:  $(round(minimum(dt_sizes), digits=2))")
        println("  Max:  $(round(maximum(dt_sizes), digits=2))")
    end
    
    return windows, profile_times, profile_w_eff
end

# ============================================================================
# ADAPTIVE EXPERIMENT RUNNER
# ============================================================================

"""
    run_adaptive_experiment(rn, u0, true_rates; kwargs...)

Run inverse CME experiment with theoretically-guided adaptive windowing.

This extends run_comprehensive_experiment by using adaptive window sizing
that maintains the uniqueness criterion Δt·w_eff < 0.3.

# Additional Keyword Arguments
- `adaptive_windowing::Bool = true`: Enable adaptive window sizing
- `target_uniqueness::Float64 = 0.25`: Target for Δt·w_eff
- `dt_bounds::Tuple{Float64,Float64} = (0.5, 10.0)`: Min/max window sizes
"""
function run_adaptive_experiment(
    rn::ReactionSystem,
    u0::Vector{Pair{Symbol, Int}},
    true_rates::Vector{Float64};
    n_trajectories::Int = 5000,
    tspan::Tuple{Float64, Float64} = (0.0, 200.0),
    tspan_learning::Tuple{Float64, Float64} = (0.0, 150.0),
    config::InverseProblemConfig = default_config(),
    experiment_name::String = "cme_adaptive",
    seed::Int = 1234,
    compute_theoretical_bounds::Bool = true,
    adaptive_windowing::Bool = true,
    target_uniqueness::Float64 = 0.25,
    dt_bounds::Tuple{Float64, Float64} = (0.5, 10.0),
    verbose::Bool = true
)
    
    # Initialize logger
    logger = ExperimentLogger(experiment_name)
    
    # Log metadata (including adaptive settings)
    log_metadata!(logger, "experiment_type", "inverse_cme_adaptive")
    log_metadata!(logger, "adaptive_windowing", adaptive_windowing)
    log_metadata!(logger, "target_uniqueness", target_uniqueness)
    log_metadata!(logger, "dt_bounds", dt_bounds)
    log_metadata!(logger, "n_species", length(species(rn)))
    log_metadata!(logger, "n_reactions", length(reactions(rn)))
    log_metadata!(logger, "true_rates", true_rates)
    log_metadata!(logger, "n_trajectories", n_trajectories)
    log_metadata!(logger, "tspan", tspan)
    log_metadata!(logger, "tspan_learning", tspan_learning)
    log_metadata!(logger, "config", Dict(
        "mass_threshold" => config.mass_threshold,
        "λ_frobenius" => config.λ_frobenius,
        "λ_prob_conservation" => config.λ_prob_conservation,
        "dt_snapshot" => config.dt_snapshot,
        "dt_window" => config.dt_window,
        "snapshots_per_window" => config.snapshots_per_window,
        "max_windows" => config.max_windows
    ))
    log_metadata!(logger, "seed", seed)
    
    if verbose
        println("="^80)
        println("ADAPTIVE INVERSE CME EXPERIMENT")
        println("="^80)
        println("\nSystem: $(length(reactions(rn))) reactions, $(length(species(rn))) species")
        println("True rates: $true_rates")
        println("Adaptive windowing: $adaptive_windowing")
        if adaptive_windowing
            println("  Target: Δt·w_eff ≈ $target_uniqueness")
            println("  Bounds: dt ∈ $dt_bounds")
        end
    end
    
    # ========================================================================
    # STEPS 1-3: Same as base experiment (data generation, state space)
    # ========================================================================
    
    if verbose
        println("\n" * "="^80)
        println("STEP 1-3: DATA GENERATION AND STATE SPACE")
        println("="^80)
    end
    
    ps = build_parameter_mapping(rn, true_rates)
    ssa_trajs = generate_ssa_data(rn, u0, tspan, ps, n_trajectories; seed=seed)
    
    traj_lengths = [length(t.t) for t in ssa_trajs]
    log_diagnostic!(logger, "trajectory_statistics", Dict(
        "n_trajectories" => n_trajectories,
        "mean_length" => mean(traj_lengths),
        "std_length" => std(traj_lengths),
        "min_length" => minimum(traj_lengths),
        "max_length" => maximum(traj_lengths),
        "median_length" => median(traj_lengths)
    ))
    
    T, distrs = convert_to_distributions(ssa_trajs, tspan_learning, config.dt_snapshot)
    
    support_sizes = [length(d) for d in distrs]
    log_diagnostic!(logger, "distribution_statistics", Dict(
        "n_distributions" => length(distrs),
        "mean_support_size" => mean(support_sizes),
        "std_support_size" => std(support_sizes),
        "min_support_size" => minimum(support_sizes),
        "max_support_size" => maximum(support_sizes),
        "time_range" => [T[1], T[end]]
    ))
    
    X = build_fsp_state_space(rn, u0)
    padded_dists = pad_distributions(distrs, X)
    
    inferred_stoich = infer_reactions_from_trajectories(ssa_trajs)
    X_global = build_global_state_space(distrs, inferred_stoich, connectivity_depth=2)
    
    log_diagnostic!(logger, "state_space_statistics", Dict(
        "fsp_size" => length(X),
        "global_size" => length(X_global),
        "compression_ratio" => length(X_global) / length(X),
        "inferred_reactions" => length(inferred_stoich),
        "stoichiometry" => inferred_stoich
    ))
    
    if verbose
        println("✓ Generated $n_trajectories trajectories")
        println("✓ Created $(length(distrs)) distributions")
        println("✓ Inferred $(length(inferred_stoich)) reactions")
    end
    
    # ========================================================================
    # STEP 4: ADAPTIVE OR FIXED WINDOWING
    # ========================================================================
    
if adaptive_windowing
    windows, profile_times, profile_w_eff = create_adaptive_windows(
        collect(T),  # Convert to Vector{Float64}
        padded_dists, 
        config, 
        inferred_stoich, 
        ssa_trajs;
        target_product = target_uniqueness,
        dt_min = dt_bounds[1],
        dt_max = dt_bounds[2],
        verbose = verbose
    )
    
    # Log w_eff profile
    log_diagnostic!(logger, "w_eff_profile", Dict(
        "times" => profile_times,
        "w_eff_values" => profile_w_eff,
        "mean_w_eff" => mean(profile_w_eff),
        "std_w_eff" => std(profile_w_eff),
        "min_w_eff" => minimum(profile_w_eff),
        "max_w_eff" => maximum(profile_w_eff)
    ))
else
    if verbose
        println("\n" * "="^80)
        println("STEP 4: FIXED WINDOWING (adaptive disabled)")
        println("="^80)
    end
    
    windows = create_windows(collect(T), padded_dists, config, inferred_stoich)  # Also convert here
    profile_times = Float64[]
    profile_w_eff = Float64[]
end
    
    # ========================================================================
    # STEP 5-7: Same as base experiment (optimization, analysis)
    # ========================================================================
    
    if verbose
        println("\n" * "="^80)
        println("STEP 5: OPTIMIZATION")
        println("="^80)
    end
    
    learned_generators = LearnedGenerator[]
    propensity_fn = auto_detect_propensity_function(rn, inferred_stoich)
    
    for (w_idx, window_data) in enumerate(windows)
        A_learned, X_local, conv_info = optimize_local_generator(window_data, config)
        
        learned_gen = LearnedGenerator(
            window_data.times[1],
            A_learned,
            X_local,
            conv_info
        )
        push!(learned_generators, learned_gen)
        
        # Log window data (same as base experiment)
        rate_stats = extract_rates(learned_gen, propensity_fn, inferred_stoich)
        
        window_log_data = Dict(
            "window_idx" => w_idx,
            "time_start" => window_data.times[1],
            "time_end" => window_data.times[end],
            "window_duration" => window_data.times[end] - window_data.times[1],
            "state_space_size" => length(X_local),
            "convergence" => Dict(
                "success" => conv_info.success,
                "return_code" => String(conv_info.return_code),
                "initial_objective" => conv_info.initial_objective,
                "final_objective" => conv_info.final_objective,
                "improvement" => conv_info.improvement,
                "improvement_pct" => 100 * conv_info.improvement / conv_info.initial_objective
            ),
            "generator_properties" => Dict(
                "max_col_sum_error" => conv_info.max_col_sum_error,
                "prob_conservation" => conv_info.prob_conservation,
                "n_nonzeros" => conv_info.n_nonzeros,
                "frobenius_norm" => norm(A_learned)
            ),
            "rate_estimates" => Dict()
        )
        
        # Add w_eff estimate for this window if available
        if adaptive_windowing && !isempty(profile_times)
            t_mid = (window_data.times[1] + window_data.times[end]) / 2
            idx = argmin(abs.(profile_times .- t_mid))
            w_est = profile_w_eff[idx]
            dt_win = window_data.times[end] - window_data.times[1]
            
            window_log_data["w_eff_estimate"] = w_est
            window_log_data["uniqueness_product"] = dt_win * w_est
            window_log_data["uniqueness_satisfied"] = dt_win * w_est < 0.3
        end
        
        for (j, stats) in rate_stats
            median_val = isnan(stats.median) ? nothing : stats.median
            mean_val = isnan(stats.mean) ? nothing : stats.mean
            std_val = isnan(stats.std) ? nothing : stats.std
            
            rel_error = if median_val === nothing
                nothing
            else
                abs(stats.median - true_rates[j]) / true_rates[j]
            end
            
            window_log_data["rate_estimates"]["reaction_$j"] = Dict(
                "median" => median_val,
                "mean" => mean_val,
                "std" => std_val,
                "n_transitions" => stats.n_transitions,
                "true_rate" => true_rates[j],
                "relative_error" => rel_error
            )
            
            if median_val !== nothing
                log_rate_evolution!(logger, j, window_data.times[1], Dict(
                    "window_idx" => w_idx,
                    "median" => median_val,
                    "mean" => mean_val,
                    "std" => std_val,
                    "n_transitions" => stats.n_transitions,
                    "relative_error" => rel_error
                ))
            end
        end
        
        log_window_data!(logger, w_idx, window_log_data)
    end
    
    # Aggregated analysis
    aggregated_stats = extract_rates_aggregated(learned_generators, propensity_fn, inferred_stoich)
    
    aggregated_log = Dict()
    for j in 1:length(inferred_stoich)
        stats = aggregated_stats[j]
        true_k = true_rates[j]
        
        median_val = isnan(stats.median) ? nothing : stats.median
        mean_val = isnan(stats.mean) ? nothing : stats.mean
        std_val = isnan(stats.std) ? nothing : stats.std
        
        rel_error = if median_val === nothing || isnan(stats.median)
            nothing
        else
            abs(stats.median - true_k) / true_k
        end
        
        aggregated_log["reaction_$j"] = Dict(
            "median" => median_val,
            "mean" => mean_val,
            "std" => std_val,
            "n_transitions" => stats.n_transitions,
            "true_rate" => true_k,
            "relative_error" => rel_error,
            "absolute_error" => median_val === nothing ? nothing : abs(stats.median - true_k)
        )
        
        if verbose && stats.n_transitions > 0 && median_val !== nothing
            err_pct = abs(stats.median - true_k) / true_k * 100
            println("  R$j: $(round(stats.median, sigdigits=4)) (true: $true_k, error: $(round(err_pct, digits=1))%, n=$(stats.n_transitions))")
        end
    end
    
    logger.data["statistics"]["aggregated_rates"] = aggregated_log
    
    # Error decomposition
    error_decomposition = analyze_error_sources(
        learned_generators, aggregated_stats, true_rates, inferred_stoich, propensity_fn
    )
    logger.data["statistics"]["error_decomposition"] = error_decomposition
    
    # Summary
    errors = [aggregated_log["reaction_$j"]["relative_error"] 
              for j in 1:length(inferred_stoich) 
              if aggregated_log["reaction_$j"]["relative_error"] !== nothing]
    
    logger.data["statistics"]["summary"] = Dict(
        "mean_relative_error" => isempty(errors) ? nothing : mean(errors),
        "median_relative_error" => isempty(errors) ? nothing : median(errors),
        "max_relative_error" => isempty(errors) ? nothing : maximum(errors),
        "min_relative_error" => isempty(errors) ? nothing : minimum(errors),
        "n_reactions_recovered" => length(errors),
        "total_windows" => length(windows),
        "total_transitions" => sum(aggregated_log["reaction_$j"]["n_transitions"] for j in 1:length(inferred_stoich))
    )
    
    save_log!(logger)
    
    if verbose
        println("\n" * "="^80)
        println("ADAPTIVE EXPERIMENT COMPLETE")
        println("="^80)
        if !isempty(errors)
            println("\nResults:")
            println("  Mean error: $(round(mean(errors)*100, digits=1))%")
            println("  Windows created: $(length(windows))")
            if adaptive_windowing
                dt_sizes = [w.times[end] - w.times[1] for w in windows]
                println("  Window size range: $(round(minimum(dt_sizes), digits=2)) - $(round(maximum(dt_sizes), digits=2))")
            end
        end
    end
    
    return ExperimentResult(
        rn, true_rates, learned_generators, inferred_stoich,
        X_global, config, ssa_trajs, (T, distrs), windows
    ), logger, (profile_times, profile_w_eff)
end

# Export
export estimate_w_eff_from_trajectories, estimate_w_eff_profile
export create_adaptive_windows, run_adaptive_experiment
