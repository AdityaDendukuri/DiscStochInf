"""
Enhanced experiment runner with comprehensive logging and analysis for publication.

Generates detailed experimental data including:
- Time evolution plots of rate estimates
- Convergence diagnostics
- State space coverage analysis
- Transition sampling statistics
- Error decomposition
- Comparative studies across parameter regimes
"""

push!(LOAD_PATH, @__DIR__)

using Catalyst
using JumpProcesses
using LinearAlgebra
using Printf
using JSON3
using Dates
using Statistics, Distributions

# Load modules
include("types.jl")
include("data_generation.jl")
include("state_space.jl")
include("optimization.jl")
include("analysis.jl")
include("src/DiscStochSim.jl")
using .DiscStochSim

# ============================================================================
# LOGGING INFRASTRUCTURE
# ============================================================================

"""
    ExperimentLogger

Comprehensive logger for experimental analysis.
"""
mutable struct ExperimentLogger
    log_dir::String
    experiment_id::String
    data::Dict{String, Any}
    start_time::DateTime
end

function ExperimentLogger(experiment_name::String)
    timestamp = Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")
    experiment_id = "$(experiment_name)_$(timestamp)"
    log_dir = joinpath("experiments", experiment_id)
    mkpath(log_dir)
    mkpath(joinpath(log_dir, "data"))
    mkpath(joinpath(log_dir, "figures"))
    
    logger = ExperimentLogger(
        log_dir,
        experiment_id,
        Dict{String, Any}(
            "metadata" => Dict{String, Any}(),
            "windows" => Dict{String, Any}(),
            "rates" => Dict{String, Any}(),
            "diagnostics" => Dict{String, Any}(),
            "statistics" => Dict{String, Any}()
        ),
        now()
    )
    
    println("📊 Experiment logging initialized: $log_dir")
    return logger
end

function log_metadata!(logger::ExperimentLogger, key::String, value)
    logger.data["metadata"][key] = value
end

function log_window_data!(logger::ExperimentLogger, window_idx::Int, data::Dict)
    logger.data["windows"]["window_$window_idx"] = data
end

function log_rate_evolution!(logger::ExperimentLogger, reaction_idx::Int, time::Float64, data::Dict)
    if !haskey(logger.data["rates"], "reaction_$reaction_idx")
        logger.data["rates"]["reaction_$reaction_idx"] = []
    end
    push!(logger.data["rates"]["reaction_$reaction_idx"], merge(data, Dict("time" => time)))
end

function log_diagnostic!(logger::ExperimentLogger, category::String, data::Dict)
    if !haskey(logger.data["diagnostics"], category)
        logger.data["diagnostics"][category] = []
    end
    push!(logger.data["diagnostics"][category], data)
end

# Add this helper function at the top of experiments.jl (after the using statements)

"""
    sanitize_for_json(obj)

Replace NaN and Inf values with null for JSON serialization.
"""
function sanitize_for_json(obj)
    if obj isa Dict
        return Dict(k => sanitize_for_json(v) for (k, v) in obj)
    elseif obj isa Array || obj isa Vector
        return [sanitize_for_json(x) for x in obj]
    elseif obj isa Float64
        if isnan(obj) || isinf(obj)
            return nothing  # Will be serialized as null in JSON
        else
            return obj
        end
    else
        return obj
    end
end

function build_parameter_mapping(rn::ReactionSystem, rates::Vector{Float64})
    params = parameters(rn)
    @assert length(params) == length(rates) "Number of rates must match number of parameters"
    return [params[i] => rates[i] for i in 1:length(rates)]
end

function save_log!(logger::ExperimentLogger)
    log_file = joinpath(logger.log_dir, "experiment_log.json")
    
    # Add final metadata
    logger.data["metadata"]["end_time"] = Dates.format(now(), "yyyy-mm-dd HH:MM:SS")
    logger.data["metadata"]["duration_seconds"] = (now() - logger.start_time).value / 1000
    
    # Sanitize data to remove NaN/Inf values
    sanitized_data = sanitize_for_json(logger.data)
    
    open(log_file, "w") do io
        JSON3.pretty(io, sanitized_data)
    end
    
    println("💾 Experiment log saved: $log_file")
    return log_file
end

"""
Enhanced experiment runner with comprehensive logging and analysis for publication.

Generates detailed experimental data including:
- Time evolution plots of rate estimates
- Convergence diagnostics
- State space coverage analysis
- Transition sampling statistics
- Error decomposition
- Comparative studies across parameter regimes
"""

push!(LOAD_PATH, @__DIR__)

using Catalyst
using JumpProcesses
using LinearAlgebra
using Printf
using JSON3
using Dates
using Statistics, Distributions

# Load modules
include("types.jl")
include("data_generation.jl")
include("state_space.jl")
include("optimization.jl")
include("analysis.jl")
include("src/DiscStochSim.jl")
using .DiscStochSim

# ============================================================================
# LOGGING INFRASTRUCTURE
# ============================================================================

"""
    ExperimentLogger

Comprehensive logger for experimental analysis.
"""
mutable struct ExperimentLogger
    log_dir::String
    experiment_id::String
    data::Dict{String, Any}
    start_time::DateTime
end

function ExperimentLogger(experiment_name::String)
    timestamp = Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")
    experiment_id = "$(experiment_name)_$(timestamp)"
    log_dir = joinpath("experiments", experiment_id)
    mkpath(log_dir)
    mkpath(joinpath(log_dir, "data"))
    mkpath(joinpath(log_dir, "figures"))
    
    logger = ExperimentLogger(
        log_dir,
        experiment_id,
        Dict{String, Any}(
            "metadata" => Dict{String, Any}(),
            "windows" => Dict{String, Any}(),
            "rates" => Dict{String, Any}(),
            "diagnostics" => Dict{String, Any}(),
            "statistics" => Dict{String, Any}()
        ),
        now()
    )
    
    println("📊 Experiment logging initialized: $log_dir")
    return logger
end

function log_metadata!(logger::ExperimentLogger, key::String, value)
    logger.data["metadata"][key] = value
end

function log_window_data!(logger::ExperimentLogger, window_idx::Int, data::Dict)
    logger.data["windows"]["window_$window_idx"] = data
end

function log_rate_evolution!(logger::ExperimentLogger, reaction_idx::Int, time::Float64, data::Dict)
    if !haskey(logger.data["rates"], "reaction_$reaction_idx")
        logger.data["rates"]["reaction_$reaction_idx"] = []
    end
    push!(logger.data["rates"]["reaction_$reaction_idx"], merge(data, Dict("time" => time)))
end

function log_diagnostic!(logger::ExperimentLogger, category::String, data::Dict)
    if !haskey(logger.data["diagnostics"], category)
        logger.data["diagnostics"][category] = []
    end
    push!(logger.data["diagnostics"][category], data)
end

# Add this helper function at the top of experiments.jl (after the using statements)

"""
    sanitize_for_json(obj)

Replace NaN and Inf values with null for JSON serialization.
"""
function sanitize_for_json(obj)
    if obj isa Dict
        return Dict(k => sanitize_for_json(v) for (k, v) in obj)
    elseif obj isa Array || obj isa Vector
        return [sanitize_for_json(x) for x in obj]
    elseif obj isa Float64
        if isnan(obj) || isinf(obj)
            return nothing  # Will be serialized as null in JSON
        else
            return obj
        end
    else
        return obj
    end
end

# Then replace the save_log! function with this version:

function save_log!(logger::ExperimentLogger)
    log_file = joinpath(logger.log_dir, "experiment_log.json")
    
    # Add final metadata
    logger.data["metadata"]["end_time"] = Dates.format(now(), "yyyy-mm-dd HH:MM:SS")
    logger.data["metadata"]["duration_seconds"] = (now() - logger.start_time).value / 1000
    
    # Sanitize data to remove NaN/Inf values
    sanitized_data = sanitize_for_json(logger.data)
    
    open(log_file, "w") do io
        JSON3.pretty(io, sanitized_data)
    end
    
    println("💾 Experiment log saved: $log_file")
    return log_file
end

# ============================================================================
# ENHANCED EXPERIMENT RUNNER
# ============================================================================

"""
    run_comprehensive_experiment(rn, u0, true_rates; kwargs...)

Run inverse CME experiment with comprehensive logging and analysis.

# Additional Keyword Arguments
- `experiment_name::String = "cme_inverse"`: Name for this experiment
- `compute_theoretical_bounds::Bool = true`: Compute theoretical identifiability bounds
- `analyze_convergence::Bool = true`: Detailed convergence analysis
- `compare_methods::Bool = false`: Compare with baseline methods
"""
function run_comprehensive_experiment(
    rn::ReactionSystem,
    u0::Vector{Pair{Symbol, Int}},
    true_rates::Vector{Float64};
    n_trajectories::Int = 300,
    tspan::Tuple{Float64, Float64} = (0.0, 200.0),
    tspan_learning::Tuple{Float64, Float64} = (0.0, 150.0),
    config::InverseProblemConfig = default_config(),
    experiment_name::String = "cme_inverse",
    seed::Int = 1234,
    compute_theoretical_bounds::Bool = true,
    analyze_convergence::Bool = true,
    verbose::Bool = true
)
    
    # Initialize logger
    logger = ExperimentLogger(experiment_name)
    
    # Log metadata
    log_metadata!(logger, "experiment_type", "inverse_cme")
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
        println("COMPREHENSIVE INVERSE CME EXPERIMENT")
        println("="^80)
        println("\nSystem: $(length(reactions(rn))) reactions, $(length(species(rn))) species")
        println("True rates: $true_rates")
        println("Experiment ID: $(logger.experiment_id)")
    end
    
    # ========================================================================
    # STEP 1: DATA GENERATION
    # ========================================================================
    
    if verbose
        println("\n" * "="^80)
        println("STEP 1: DATA GENERATION")
        println("="^80)
    end
    
    ps = build_parameter_mapping(rn, true_rates)
    ssa_trajs = generate_ssa_data(rn, u0, tspan, ps, n_trajectories; seed=seed)
    
    # Log trajectory statistics
    traj_lengths = [length(t.t) for t in ssa_trajs]
    log_diagnostic!(logger, "trajectory_statistics", Dict(
        "n_trajectories" => n_trajectories,
        "mean_length" => mean(traj_lengths),
        "std_length" => std(traj_lengths),
        "min_length" => minimum(traj_lengths),
        "max_length" => maximum(traj_lengths),
        "median_length" => median(traj_lengths)
    ))
    
    if verbose
        println("✓ Generated $n_trajectories trajectories")
        println("  Mean length: $(round(mean(traj_lengths), digits=1)) ± $(round(std(traj_lengths), digits=1))")
    end
    
    # ========================================================================
    # STEP 2: DISTRIBUTION CONVERSION
    # ========================================================================
    
    if verbose
        println("\n" * "="^80)
        println("STEP 2: DISTRIBUTION CONVERSION")
        println("="^80)
    end
    
    T, distrs = convert_to_distributions(ssa_trajs, tspan_learning, config.dt_snapshot)
    
    # Analyze distribution support evolution
    support_sizes = [length(d) for d in distrs]
    log_diagnostic!(logger, "distribution_statistics", Dict(
        "n_distributions" => length(distrs),
        "mean_support_size" => mean(support_sizes),
        "std_support_size" => std(support_sizes),
        "min_support_size" => minimum(support_sizes),
        "max_support_size" => maximum(support_sizes),
        "time_range" => [T[1], T[end]]
    ))
    
    if verbose
        println("✓ Created $(length(distrs)) distributions")
        println("  Support size: $(round(mean(support_sizes), digits=1)) ± $(round(std(support_sizes), digits=1))")
    end
    
    # ========================================================================
    # STEP 3: STATE SPACE CONSTRUCTION
    # ========================================================================
    
    if verbose
        println("\n" * "="^80)
        println("STEP 3: STATE SPACE CONSTRUCTION")
        println("="^80)
    end
    
    X = build_fsp_state_space(rn, u0)
    padded_dists = pad_distributions(distrs, X)
    
    inferred_stoich = infer_reactions_from_trajectories(ssa_trajs)
    X_global = build_global_state_space(distrs, inferred_stoich, connectivity_depth=2)
    
    # Log state space statistics
    log_diagnostic!(logger, "state_space_statistics", Dict(
        "fsp_size" => length(X),
        "global_size" => length(X_global),
        "compression_ratio" => length(X_global) / length(X),
        "inferred_reactions" => length(inferred_stoich),
        "stoichiometry" => inferred_stoich
    ))
    
    if verbose
        println("✓ FSP state space: $(length(X)) states")
        println("✓ Global state space: $(length(X_global)) states")
        println("✓ Inferred $(length(inferred_stoich)) reactions")
    end
    
    # ========================================================================
    # STEP 4: THEORETICAL BOUNDS (if requested)
    # ========================================================================
    
    if compute_theoretical_bounds && verbose
        println("\n" * "="^80)
        println("STEP 4: THEORETICAL IDENTIFIABILITY ANALYSIS")
        println("="^80)
    end
    
    if compute_theoretical_bounds
        theoretical_bounds = compute_identifiability_bounds(
            distrs, T, config, X_global, inferred_stoich
        )
        
        log_diagnostic!(logger, "theoretical_bounds", theoretical_bounds)
        
        if verbose
            println("✓ Effective flux: $(round(theoretical_bounds["w_eff_mean"], sigdigits=3))")
            println("✓ Spectral gap (estimated): $(round(theoretical_bounds["lambda_gap_estimate"], sigdigits=3))")
            println("✓ Conditioning ratio: $(round(theoretical_bounds["conditioning_ratio"], sigdigits=3))")
            println("✓ Uniqueness criterion: Δt·w_eff = $(round(theoretical_bounds["uniqueness_product"], sigdigits=3)) $(theoretical_bounds["uniqueness_satisfied"] ? "✓" : "✗")")
        end
    end
    
    # ========================================================================
    # STEP 5: SLIDING WINDOW OPTIMIZATION
    # ========================================================================
    
    if verbose
        println("\n" * "="^80)
        println("STEP 5: SLIDING WINDOW OPTIMIZATION")
        println("="^80)
    end
    
    windows = create_windows(T, padded_dists, config, inferred_stoich)
    learned_generators = LearnedGenerator[]
    
    propensity_fn = auto_detect_propensity_function(rn, inferred_stoich)
    
    for (w_idx, window_data) in enumerate(windows)
        if verbose
            println("\n--- Window $w_idx: t ∈ [$(window_data.times[1]), $(window_data.times[end])] ---")
        end
        
        A_learned, X_local, conv_info = optimize_local_generator(window_data, config)
        
        learned_gen = LearnedGenerator(
            window_data.times[1],
            A_learned,
            X_local,
            conv_info
        )
        push!(learned_generators, learned_gen)
        
        # Extract rates for this window
        rate_stats = extract_rates(learned_gen, propensity_fn, inferred_stoich)
       # *** ADD TRANSITION DIAGNOSTICS HERE ***
    log_transition_diagnostics!(
        logger, w_idx, learned_gen, propensity_fn, 
        inferred_stoich, true_rates
    )
     
        # Log comprehensive window data
        window_log_data = Dict(
            "window_idx" => w_idx,
            "time_start" => window_data.times[1],
            "time_end" => window_data.times[end],
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
            )
        )
        
        # Add rate estimates
        window_log_data["rate_estimates"] = Dict()
        for (j, stats) in rate_stats
            window_log_data["rate_estimates"]["reaction_$j"] = Dict(
                "median" => stats.median,
                "mean" => stats.mean,
                "std" => stats.std,
                "min" => stats.min,
                "max" => stats.max,
                "n_transitions" => stats.n_transitions,
                "true_rate" => true_rates[j],
                "relative_error" => isnan(stats.median) ? NaN : abs(stats.median - true_rates[j]) / true_rates[j]
            )
            
            # Log rate evolution
            if !isnan(stats.median)
                log_rate_evolution!(logger, j, window_data.times[1], Dict(
                    "window_idx" => w_idx,
                    "median" => stats.median,
                    "mean" => stats.mean,
                    "std" => stats.std,
                    "n_transitions" => stats.n_transitions,
                    "relative_error" => abs(stats.median - true_rates[j]) / true_rates[j]
                ))
            end
        end
        
        log_window_data!(logger, w_idx, window_log_data)
        
        if verbose && conv_info.success
            println("  ✓ Converged: $(round(conv_info.improvement/conv_info.initial_objective*100, digits=1))% improvement")
        end
    end
    
    # ========================================================================
    # STEP 6: AGGREGATED ANALYSIS
    # ========================================================================
    
    if verbose
        println("\n" * "="^80)
        println("STEP 6: AGGREGATED RATE ANALYSIS")
        println("="^80)
    end
    
    aggregated_stats = extract_rates_aggregated(learned_generators, propensity_fn, inferred_stoich)
    
    # Log aggregated results
    aggregated_log = Dict()
    for j in 1:length(inferred_stoich)
        stats = aggregated_stats[j]
        true_k = true_rates[j]
        
        aggregated_log["reaction_$j"] = Dict(
            "median" => stats.median,
            "mean" => stats.mean,
            "std" => stats.std,
            "n_transitions" => stats.n_transitions,
            "true_rate" => true_k,
            "relative_error" => isnan(stats.median) ? NaN : abs(stats.median - true_k) / true_k,
            "absolute_error" => isnan(stats.median) ? NaN : abs(stats.median - true_k)
        )
        
        if verbose && stats.n_transitions > 0
            err_pct = abs(stats.median - true_k) / true_k * 100
            println("  R$j: $(round(stats.median, sigdigits=4)) (true: $true_k, error: $(round(err_pct, digits=1))%, n=$(stats.n_transitions))")
        end
    end
    
    logger.data["statistics"]["aggregated_rates"] = aggregated_log
    
    # ========================================================================
    # STEP 7: ERROR DECOMPOSITION ANALYSIS
    # ========================================================================
    
    if analyze_convergence && verbose
        println("\n" * "="^80)
        println("STEP 7: ERROR DECOMPOSITION")
        println("="^80)
    end
    
    if analyze_convergence
    error_decomposition = analyze_error_sources(
    learned_generators, aggregated_stats, true_rates, inferred_stoich, propensity_fn
)
        
        logger.data["statistics"]["error_decomposition"] = error_decomposition
        
        if verbose
            println("  Sampling variance contribution: $(round(error_decomposition["sampling_variance_contribution"]*100, digits=1))%")
            println("  Bias contribution: $(round(error_decomposition["bias_contribution"]*100, digits=1))%")
            println("  Window inconsistency: $(round(error_decomposition["window_inconsistency"], sigdigits=3))")
        end
    end
    
    # ========================================================================
    # STEP 8: FINAL SUMMARY
    # ========================================================================
    
    # Compute summary statistics
    errors = [aggregated_log["reaction_$j"]["relative_error"] 
              for j in 1:length(inferred_stoich) 
              if !isnan(aggregated_log["reaction_$j"]["relative_error"])]
    
    logger.data["statistics"]["summary"] = Dict(
        "mean_relative_error" => isempty(errors) ? NaN : mean(errors),
        "median_relative_error" => isempty(errors) ? NaN : median(errors),
        "max_relative_error" => isempty(errors) ? NaN : maximum(errors),
        "min_relative_error" => isempty(errors) ? NaN : minimum(errors),
        "n_reactions_recovered" => length(errors),
        "total_windows" => length(windows),
        "total_transitions" => sum(aggregated_log["reaction_$j"]["n_transitions"] for j in 1:length(inferred_stoich))
    )
    
    # Save log
    save_log!(logger)
    
    if verbose
        println("\n" * "="^80)
        println("EXPERIMENT COMPLETE")
        println("="^80)
        println("\n📊 Results logged to: $(logger.log_dir)")
        println("\nSummary:")
        println("  Mean error: $(round(mean(errors)*100, digits=1))%")
        println("  Median error: $(round(median(errors)*100, digits=1))%")
        println("  Best reaction: $(round(minimum(errors)*100, digits=1))%")
        println("  Worst reaction: $(round(maximum(errors)*100, digits=1))%")
    end
    
    # Return result object
    return ExperimentResult(
        rn, true_rates, learned_generators, inferred_stoich,
        X_global, config, ssa_trajs, (T, distrs), windows
    ), logger
end

# ============================================================================
# THEORETICAL BOUNDS COMPUTATION
# ============================================================================

"""
    compute_identifiability_bounds(distrs, T, config, X, stoich_vecs)

Compute theoretical bounds from Theorems 2.7, 2.12, 2.13.
"""
function compute_identifiability_bounds(distrs, T, config, X, stoich_vecs)
    # Estimate effective flux
    w_eff_samples = Float64[]
    
    for (i, dist) in enumerate(distrs[1:min(100, length(distrs))])
        # Approximate exit rates from local transitions
        w_local = 0.0
        for (state, prob) in dist
            # Simple heuristic: exit rate ~ number of possible reactions * typical rate
            n_possible_reactions = length(stoich_vecs)
            w_local += prob * n_possible_reactions * 0.1  # Assume typical rate ~ 0.1
        end
        push!(w_eff_samples, w_local)
    end
    
    w_eff_mean = mean(w_eff_samples)
    
    # Estimate spectral gap from distribution decay
    # Fit exponential decay ||p(t) - p_ss|| ~ exp(-λ_gap * t)
    if length(distrs) > 10
        # Simple heuristic: measure time to reach near-equilibrium
        support_sizes = [length(d) for d in distrs]
        # When support stabilizes, we're near equilibrium
        stabilization_idx = findfirst(i -> std(support_sizes[i:min(i+10, end)]) < 0.1 * mean(support_sizes[i:min(i+10, end)]), 1:length(support_sizes)-10)
        t_relax = stabilization_idx !== nothing ? T[stabilization_idx] : T[end]
        lambda_gap_estimate = 1.0 / t_relax
    else
        lambda_gap_estimate = 0.01  # Default fallback
    end
    
    # Compute identifiability metrics
    uniqueness_product = config.dt_window * w_eff_mean
    uniqueness_satisfied = uniqueness_product < 0.3
    
    conditioning_ratio = w_eff_mean / lambda_gap_estimate
    
    return Dict(
        "w_eff_mean" => w_eff_mean,
        "w_eff_std" => std(w_eff_samples),
        "lambda_gap_estimate" => lambda_gap_estimate,
        "relaxation_time_estimate" => 1.0 / lambda_gap_estimate,
        "uniqueness_product" => uniqueness_product,
        "uniqueness_satisfied" => uniqueness_satisfied,
        "uniqueness_threshold" => 0.3,
        "conditioning_ratio" => conditioning_ratio,
        "conditioning_threshold" => 5.0,
        "conditioning_satisfied" => conditioning_ratio >= 5.0
    )
end

# ============================================================================
# ERROR ANALYSIS
# ============================================================================

"""
    analyze_error_sources(learned_generators, aggregated_stats, true_rates, stoich_vecs, propensity_fn)

Decompose error into sampling variance, bias, and window inconsistency.
"""
function analyze_error_sources(learned_generators, aggregated_stats, true_rates, stoich_vecs, propensity_fn)
    # For each reaction, collect estimates across windows
    window_estimates = Dict{Int, Vector{Float64}}()
    
    for gen in learned_generators
        rate_stats = extract_rates(gen, propensity_fn, stoich_vecs)
        for (j, stats) in rate_stats
            if stats.n_transitions > 0 && !isnan(stats.median)
                if !haskey(window_estimates, j)
                    window_estimates[j] = Float64[]
                end
                push!(window_estimates[j], stats.median)
            end
        end
    end
    
    # Compute variance across windows
    window_variances = Dict(j => length(estimates) > 1 ? var(estimates) : 0.0 
                           for (j, estimates) in window_estimates)
    
    # Compute bias
    biases = Dict{Int, Float64}()
    for j in 1:length(stoich_vecs)
        stats = aggregated_stats[j]
        if !isnan(stats.median)
            biases[j] = abs(stats.median - true_rates[j])
        end
    end
    
    # Estimate sampling variance contribution
    sampling_variances = Dict{Int, Float64}()
    for j in 1:length(stoich_vecs)
        stats = aggregated_stats[j]
        if stats.n_transitions > 0 && !isnan(stats.std)
            sampling_variances[j] = (stats.std)^2 / max(stats.n_transitions, 1)
        end
    end
    
    # Compute contributions
    total_variance = sum(values(window_variances))
    total_sampling_var = sum(values(sampling_variances))
    total_bias = sum(values(biases))
    
    window_means = [mean(estimates) for estimates in values(window_estimates) if length(estimates) > 1]
    window_inconsistency = isempty(window_means) ? 0.0 : std(window_means)
    
    return Dict(
        "window_variances" => window_variances,
        "sampling_variances" => sampling_variances,
        "biases" => biases,
        "sampling_variance_contribution" => total_sampling_var / (total_sampling_var + total_variance + 1e-10),
        "bias_contribution" => total_bias / (total_bias + total_variance + 1e-10),
        "window_inconsistency" => window_inconsistency
    )
end

"""
Memory-efficient wrapper for running experiments.
Bypasses the full trajectory storage.
"""

function run_experiment_efficient(
    rn, u0, true_rates;
    n_trajectories = 5000,
    tspan = (0.0, 200.0),
    tspan_learning = (0.0, 150.0),
    config = default_config(),
    experiment_name = "experiment",
    seed = 1234
)
    
    println("="^80)
    println("MEMORY-EFFICIENT EXPERIMENT: $experiment_name")
    println("="^80)
    
    # Initialize logger
    logger = ExperimentLogger(experiment_name)
    log_metadata!(logger, "n_trajectories", n_trajectories)
    log_metadata!(logger, "true_rates", true_rates)
    
    # Build parameter mapping
    ps = build_parameter_mapping(rn, true_rates)
    
    # STEP 1: Generate distributions DIRECTLY (no trajectory storage)
    println("\nGenerating distributions directly from SSA...")
    include("data_generation_eff.jl")
    
    T, distrs = generate_distributions_directly(
        rn, u0, tspan_learning, ps, n_trajectories;
        dt_snapshot = config.dt_snapshot,
        seed = seed
    )
    
    println("✓ Generated $(length(distrs)) distributions")
    
    # STEP 2: Build state space
    println("\nBuilding state space...")
    X = build_fsp_state_space(rn, u0)
    padded_dists = pad_distributions(distrs, X)
    
    # For stoichiometry, we need at least one trajectory
    # Generate just ONE for inference
    println("Generating single trajectory for stoichiometry inference...")
    dprob = DiscreteProblem(rn, u0, (0.0, 20.0), ps)
    jprob = JumpProblem(rn, dprob, Direct())
    test_traj = solve(jprob, SSAStepper())
    
    inferred_stoich = infer_reactions_from_trajectories([test_traj])
    X_global = build_global_state_space(distrs, inferred_stoich, connectivity_depth=2)
    
    println("✓ Global state space: $(length(X_global)) states")
    println("✓ Inferred $(length(inferred_stoich)) reactions")
    
    # STEP 3: Sliding window optimization
    println("\nRunning sliding window optimization...")
    windows = create_windows(T, padded_dists, config, inferred_stoich)
    learned_generators = LearnedGenerator[]
    
    propensity_fn = auto_detect_propensity_function(rn, inferred_stoich)
    
    for (w_idx, window_data) in enumerate(windows)
        println("\n--- Window $w_idx: t ∈ [$(window_data.times[1]), $(window_data.times[end])] ---")
        
        A_learned, X_local, conv_info = optimize_local_generator(window_data, config)
        
        learned_gen = LearnedGenerator(
            window_data.times[1],
            A_learned,
            X_local,
            conv_info
        )
        push!(learned_generators, learned_gen)
        
        if conv_info.success
            improvement_pct = 100 * conv_info.improvement / conv_info.initial_objective
            println("  ✓ Converged: $(round(improvement_pct, digits=1))% improvement")
        end
    end
    
    # STEP 4: Extract rates
    println("\n" * "="^80)
    println("AGGREGATED RATE ANALYSIS")
    println("="^80)
    
    aggregated_stats = extract_rates_aggregated(learned_generators, propensity_fn, inferred_stoich)
    
    for j in 1:length(inferred_stoich)
        stats = aggregated_stats[j]
        true_k = true_rates[j]
        
        if stats.n_transitions > 0
            err_pct = abs(stats.median - true_k) / true_k * 100
            println("  R$j: $(round(stats.median, sigdigits=4)) (true: $true_k, error: $(round(err_pct, digits=1))%, n=$(stats.n_transitions))")
        end
    end
    
    # Save minimal log
    logger.data["statistics"] = Dict(
        "aggregated_rates" => Dict(
            "reaction_$j" => Dict(
                "median" => aggregated_stats[j].median,
                "true_rate" => true_rates[j],
                "relative_error" => abs(aggregated_stats[j].median - true_rates[j]) / true_rates[j],
                "n_transitions" => aggregated_stats[j].n_transitions
            ) for j in 1:length(inferred_stoich)
        ),
        "summary" => Dict(
            "mean_relative_error" => mean([abs(aggregated_stats[j].median - true_rates[j]) / true_rates[j] 
                                          for j in 1:length(inferred_stoich) 
                                          if aggregated_stats[j].n_transitions > 0])
        )
    )
    
    save_log!(logger)
    
    println("\n✅ Experiment complete!")
    println("📊 Log saved: $(logger.log_dir)")
    
    return learned_generators, logger
end


# ============================================================================
# COMPARATIVE EXPERIMENTS
# ============================================================================

# ============================================================================
# COMPARATIVE EXPERIMENTS
# ============================================================================

"""
    create_modified_config(base_config, param_name, param_value)

Create a new InverseProblemConfig with one modified parameter.
"""
function create_modified_config(base_config::InverseProblemConfig, param_name::Symbol, param_value)
    # Get all field names
    fields = fieldnames(InverseProblemConfig)
    
    # Create dictionary of current values
    kwargs = Dict{Symbol, Any}()
    for f in fields
        kwargs[f] = getfield(base_config, f)
    end
    
    # Modify the specified field
    kwargs[param_name] = param_value
    
    # Create and return new config using keyword arguments
    return InverseProblemConfig(;kwargs...)
end

"""
    run_parameter_sweep(rn, u0, true_rates, param_name, param_values; kwargs...)

Run experiments across parameter sweep for sensitivity analysis.
"""
function run_parameter_sweep(
    rn::ReactionSystem,
    u0::Vector{Pair{Symbol, Int}},
    true_rates::Vector{Float64},
    param_name::Symbol,
    param_values::Vector;
    base_config = default_config(),
    base_n_trajectories::Int = 5000,
    experiment_name::String = "parameter_sweep"
)
    
    results = []
    
    for (i, param_value) in enumerate(param_values)
        println("\n" * "="^80)
        println("PARAMETER SWEEP: $param_name = $param_value ($i/$(length(param_values)))")
        println("="^80)
        
        # Create modified config or n_trajectories
        if param_name == :n_trajectories
            n_traj = param_value
            config = base_config
        else
            n_traj = base_n_trajectories
            # Create new config with modified parameter
            config = create_modified_config(base_config, param_name, param_value)
        end
        
        # Run experiment
        exp_name = "$(experiment_name)_$(param_name)_$(param_value)"
        result, logger = run_comprehensive_experiment(
            rn, u0, true_rates;
            n_trajectories = n_traj,
            config = config,
            experiment_name = exp_name,
            verbose = false
        )
        
        push!(results, (param_value = param_value, result = result, logger = logger))
    end
    
    return results
end

"""
Create adaptive windows based on system dynamics.
Shorter windows for fast-mixing regions, longer for slow regions.
"""
function create_adaptive_windows(
    T::Vector{Float64},
    distrs::Vector{Distribution},
    config::InverseProblemConfig,
    stoich_vecs::Vector{Vector{Int}}
)
    windows = WindowData[]
    
    # Estimate local mixing rate
    mixing_rates = estimate_local_mixing_rates(T, distrs)
    
    i = 1
    while i <= length(T) - config.snapshots_per_window
        # Adapt window size based on local mixing rate
        mixing_rate = mixing_rates[i]
        
        # Faster mixing → shorter windows (more frequent updates)
        # Slower mixing → longer windows (more data per window)
        adaptive_dt = min(
            config.dt_window,
            max(2.0, 5.0 / mixing_rate)  # At least 5 characteristic times
        )
        
        # Find endpoint
        t_start = T[i]
        t_end = t_start + adaptive_dt
        end_idx = findlast(t -> t <= t_end, T)
        
        if end_idx - i + 1 >= config.snapshots_per_window
            window_times = T[i:end_idx]
            window_dists = distrs[i:end_idx]
            
            push!(windows, WindowData(
                window_idx = length(windows) + 1,
                times = window_times,
                distributions = window_dists,
                stoich_vecs = stoich_vecs
            ))
            
            i = end_idx + 1
        else
            i += 1
        end
    end
    
    return windows
end

function estimate_local_mixing_rates(T::Vector{Float64}, distrs::Vector{Distribution})
    mixing_rates = zeros(length(T))
    
    for i in 1:length(T)-1
        # Measure distribution change
        overlap = distribution_overlap(distrs[i], distrs[i+1])
        dt = T[i+1] - T[i]
        
        # Mixing rate ~ -log(overlap) / dt
        mixing_rates[i] = -log(max(overlap, 1e-10)) / dt
    end
    
    # Smooth
    for i in 2:length(T)-1
        mixing_rates[i] = mean(mixing_rates[max(1,i-5):min(end,i+5)])
    end
    
    return mixing_rates
end

function distribution_overlap(d1::Distribution, d2::Distribution)
    states1 = Set(keys(d1))
    states2 = Set(keys(d2))
    common_states = intersect(states1, states2)
    
    overlap = sum(min(d1[s], d2[s]) for s in common_states)
    return overlap
end

"""
Hierarchical Bayesian aggregation across windows.
Accounts for both within-window and between-window variance.
"""
function extract_rates_hierarchical(
    learned_generators::Vector{LearnedGenerator},
    propensity_fn::Function,
    stoich_vecs::Vector{Vector{Int}}
)
    # Collect window-level estimates
    window_estimates = Dict{Int, Vector{Tuple{Float64, Float64}}}()  # (mean, variance)
    
    for gen in learned_generators
        rate_stats = extract_rates_flux_filtered(gen, propensity_fn, stoich_vecs)
        
        for (j, stats) in rate_stats
            if stats.n_transitions > 5  # Require minimum sample size
                if !haskey(window_estimates, j)
                    window_estimates[j] = Tuple{Float64, Float64}[]
                end
                
                # Collect mean and variance from this window
                push!(window_estimates[j], (stats.mean, stats.std^2))
            end
        end
    end
    
    # Hierarchical estimate: James-Stein shrinkage
    hierarchical_stats = Dict{Int, RateStatistics}()
    
    for (j, estimates) in window_estimates
        means = [e[1] for e in estimates]
        variances = [e[2] for e in estimates]
        
        # Grand mean
        grand_mean = mean(means)
        
        # Between-window variance
        tau2 = var(means)
        
        # Within-window variance (average)
        sigma2_mean = mean(variances)
        
        # Shrinkage weights (James-Stein)
        weights = tau2 ./ (tau2 .+ variances)
        
        # Shrink toward grand mean
        shrunk_estimates = weights .* grand_mean .+ (1 .- weights) .* means
        
        # Final estimate: weighted average
        final_mean = mean(shrunk_estimates)
        final_std = sqrt(tau2 + sigma2_mean / length(means))
        
        hierarchical_stats[j] = RateStatistics(
            median = median(shrunk_estimates),
            mean = final_mean,
            std = final_std,
            min = minimum(means),
            max = maximum(means),
            n_transitions = sum(length(gen.X_local) for gen in learned_generators)
        )
    end
    
    return hierarchical_stats
end

"""
Extract rates with flux-based filtering.
Only use transitions that carry significant probability flux.
"""
function extract_rates_flux_filtered(
    gen::LearnedGenerator,
    propensity_fn::Function,
    stoich_vecs::Vector{Vector{Int}};
    flux_threshold::Float64 = 1e-6  # Minimum flux to consider
)
    rate_samples = Dict{Int, Vector{Float64}}()
    flux_weights = Dict{Int, Vector{Float64}}()  # NEW
    
    for (i, state_i) in enumerate(gen.X_local)
        state_vec = [state_i[k] for k in 1:length(state_vec)]
        
        for j in 1:length(stoich_vecs)
            target_vec = state_vec + stoich_vecs[j]
            target_ci = CartesianIndex(target_vec...)
            
            target_idx = findfirst(==(target_ci), gen.X_local)
            if !isnothing(target_idx)
                propensity = propensity_fn(state_vec, j)
                
                if propensity > 0 && gen.A_learned[target_idx, i] > 0
                    rate = gen.A_learned[target_idx, i] / propensity
                    
                    # Compute flux through this transition
                    # Flux = rate × propensity × probability
                    # We don't have p(state) here, so approximate with uniform
                    flux = rate * propensity / length(gen.X_local)
                    
                    if flux > flux_threshold  # FILTER
                        if !haskey(rate_samples, j)
                            rate_samples[j] = Float64[]
                            flux_weights[j] = Float64[]
                        end
                        push!(rate_samples[j], rate)
                        push!(flux_weights[j], flux)
                    end
                end
            end
        end
    end
    
    # Compute flux-weighted statistics
    rate_stats = Dict{Int, RateStatistics}()
    for j in 1:length(stoich_vecs)
        if haskey(rate_samples, j) && !isempty(rate_samples[j])
            samples = rate_samples[j]
            weights = flux_weights[j]
            weights_norm = weights ./ sum(weights)
            
            # Weighted statistics
            weighted_mean = sum(samples .* weights_norm)
            weighted_median = median(samples)  # Could implement weighted median
            
            rate_stats[j] = RateStatistics(
                median = weighted_median,
                mean = weighted_mean,
                std = sqrt(sum(weights_norm .* (samples .- weighted_mean).^2)),
                min = minimum(samples),
                max = maximum(samples),
                n_transitions = length(samples)
            )
        end
    end
    
    return rate_stats
end

# Export
export run_comprehensive_experiment, run_parameter_sweep, ExperimentLogger
export compute_identifiability_bounds, analyze_error_sources1
# ============================================================================
# ENHANCED EXPERIMENT RUNNER
# ============================================================================

"""
    run_comprehensive_experiment(rn, u0, true_rates; kwargs...)

Run inverse CME experiment with comprehensive logging and analysis.

# Additional Keyword Arguments
- `experiment_name::String = "cme_inverse"`: Name for this experiment
- `compute_theoretical_bounds::Bool = true`: Compute theoretical identifiability bounds
- `analyze_convergence::Bool = true`: Detailed convergence analysis
- `compare_methods::Bool = false`: Compare with baseline methods
"""
function run_comprehensive_experiment(
    rn::ReactionSystem,
    u0::Vector{Pair{Symbol, Int}},
    true_rates::Vector{Float64};
    n_trajectories::Int = 5000,
    tspan::Tuple{Float64, Float64} = (0.0, 200.0),
    tspan_learning::Tuple{Float64, Float64} = (0.0, 150.0),
    config::InverseProblemConfig = default_config(),
    experiment_name::String = "cme_inverse",
    seed::Int = 1234,
    compute_theoretical_bounds::Bool = true,
    analyze_convergence::Bool = true,
    verbose::Bool = true
)
    
    # Initialize logger
    logger = ExperimentLogger(experiment_name)
    
    # Log metadata
    log_metadata!(logger, "experiment_type", "inverse_cme")
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
        println("COMPREHENSIVE INVERSE CME EXPERIMENT")
        println("="^80)
        println("\nSystem: $(length(reactions(rn))) reactions, $(length(species(rn))) species")
        println("True rates: $true_rates")
        println("Experiment ID: $(logger.experiment_id)")
    end
    
    # ========================================================================
    # STEP 1: DATA GENERATION
    # ========================================================================
    
    if verbose
        println("\n" * "="^80)
        println("STEP 1: DATA GENERATION")
        println("="^80)
    end
    
    ps = build_parameter_mapping(rn, true_rates)
    ssa_trajs = generate_ssa_data(rn, u0, tspan, ps, n_trajectories; seed=seed)
    
    # Log trajectory statistics
    traj_lengths = [length(t.t) for t in ssa_trajs]
    log_diagnostic!(logger, "trajectory_statistics", Dict(
        "n_trajectories" => n_trajectories,
        "mean_length" => mean(traj_lengths),
        "std_length" => std(traj_lengths),
        "min_length" => minimum(traj_lengths),
        "max_length" => maximum(traj_lengths),
        "median_length" => median(traj_lengths)
    ))
    
    if verbose
        println("✓ Generated $n_trajectories trajectories")
        println("  Mean length: $(round(mean(traj_lengths), digits=1)) ± $(round(std(traj_lengths), digits=1))")
    end
    
    # ========================================================================
    # STEP 2: DISTRIBUTION CONVERSION
    # ========================================================================
    
    if verbose
        println("\n" * "="^80)
        println("STEP 2: DISTRIBUTION CONVERSION")
        println("="^80)
    end
    
    T, distrs = convert_to_distributions(ssa_trajs, tspan_learning, config.dt_snapshot)
    
    # Analyze distribution support evolution
    support_sizes = [length(d) for d in distrs]
    log_diagnostic!(logger, "distribution_statistics", Dict(
        "n_distributions" => length(distrs),
        "mean_support_size" => mean(support_sizes),
        "std_support_size" => std(support_sizes),
        "min_support_size" => minimum(support_sizes),
        "max_support_size" => maximum(support_sizes),
        "time_range" => [T[1], T[end]]
    ))
    
    if verbose
        println("✓ Created $(length(distrs)) distributions")
        println("  Support size: $(round(mean(support_sizes), digits=1)) ± $(round(std(support_sizes), digits=1))")
    end
    
    # ========================================================================
    # STEP 3: STATE SPACE CONSTRUCTION
    # ========================================================================
    
    if verbose
        println("\n" * "="^80)
        println("STEP 3: STATE SPACE CONSTRUCTION")
        println("="^80)
    end
    
    X = build_fsp_state_space(rn, u0)
    padded_dists = pad_distributions(distrs, X)
    
    inferred_stoich = infer_reactions_from_trajectories(ssa_trajs)
    X_global = build_global_state_space(distrs, inferred_stoich, connectivity_depth=2)
    
    # Log state space statistics
    log_diagnostic!(logger, "state_space_statistics", Dict(
        "fsp_size" => length(X),
        "global_size" => length(X_global),
        "compression_ratio" => length(X_global) / length(X),
        "inferred_reactions" => length(inferred_stoich),
        "stoichiometry" => inferred_stoich
    ))
    
    if verbose
        println("✓ FSP state space: $(length(X)) states")
        println("✓ Global state space: $(length(X_global)) states")
        println("✓ Inferred $(length(inferred_stoich)) reactions")
    end
    
    # ========================================================================
    # STEP 4: THEORETICAL BOUNDS (if requested)
    # ========================================================================
    
    if compute_theoretical_bounds && verbose
        println("\n" * "="^80)
        println("STEP 4: THEORETICAL IDENTIFIABILITY ANALYSIS")
        println("="^80)
    end
    
    if compute_theoretical_bounds
        theoretical_bounds = compute_identifiability_bounds(
            distrs, T, config, X_global, inferred_stoich
        )
        
        log_diagnostic!(logger, "theoretical_bounds", theoretical_bounds)
        
        if verbose
            println("✓ Effective flux: $(round(theoretical_bounds["w_eff_mean"], sigdigits=3))")
            println("✓ Spectral gap (estimated): $(round(theoretical_bounds["lambda_gap_estimate"], sigdigits=3))")
            println("✓ Conditioning ratio: $(round(theoretical_bounds["conditioning_ratio"], sigdigits=3))")
            println("✓ Uniqueness criterion: Δt·w_eff = $(round(theoretical_bounds["uniqueness_product"], sigdigits=3)) $(theoretical_bounds["uniqueness_satisfied"] ? "✓" : "✗")")
        end
    end
    
    # ========================================================================
    # STEP 5: SLIDING WINDOW OPTIMIZATION
    # ========================================================================
    
    if verbose
        println("\n" * "="^80)
        println("STEP 5: SLIDING WINDOW OPTIMIZATION")
        println("="^80)
    end
    
    windows = create_windows(T, padded_dists, config, inferred_stoich)
    learned_generators = LearnedGenerator[]
    
    propensity_fn = auto_detect_propensity_function(rn, inferred_stoich)
    
    for (w_idx, window_data) in enumerate(windows)
        if verbose
            println("\n--- Window $w_idx: t ∈ [$(window_data.times[1]), $(window_data.times[end])] ---")
        end
        
        A_learned, X_local, conv_info = optimize_local_generator(window_data, config)
        
        learned_gen = LearnedGenerator(
            window_data.times[1],
            A_learned,
            X_local,
            conv_info
        )
        push!(learned_generators, learned_gen)
        
        # Extract rates for this window
        rate_stats = extract_rates(learned_gen, propensity_fn, inferred_stoich)
        
        # Log comprehensive window data
        window_log_data = Dict(
            "window_idx" => w_idx,
            "time_start" => window_data.times[1],
            "time_end" => window_data.times[end],
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
            )
        )
        
        # Add rate estimates
        window_log_data["rate_estimates"] = Dict()
        for (j, stats) in rate_stats
            window_log_data["rate_estimates"]["reaction_$j"] = Dict(
                "median" => stats.median,
                "mean" => stats.mean,
                "std" => stats.std,
                "min" => stats.min,
                "max" => stats.max,
                "n_transitions" => stats.n_transitions,
                "true_rate" => true_rates[j],
                "relative_error" => isnan(stats.median) ? NaN : abs(stats.median - true_rates[j]) / true_rates[j]
            )
            
            # Log rate evolution
            if !isnan(stats.median)
                log_rate_evolution!(logger, j, window_data.times[1], Dict(
                    "window_idx" => w_idx,
                    "median" => stats.median,
                    "mean" => stats.mean,
                    "std" => stats.std,
                    "n_transitions" => stats.n_transitions,
                    "relative_error" => abs(stats.median - true_rates[j]) / true_rates[j]
                ))
            end
        end
        
        log_window_data!(logger, w_idx, window_log_data)
        
        if verbose && conv_info.success
            println("  ✓ Converged: $(round(conv_info.improvement/conv_info.initial_objective*100, digits=1))% improvement")
        end
    end
    
    # ========================================================================
    # STEP 6: AGGREGATED ANALYSIS
    # ========================================================================
    
    if verbose
        println("\n" * "="^80)
        println("STEP 6: AGGREGATED RATE ANALYSIS")
        println("="^80)
    end
    
    aggregated_stats = extract_rates_aggregated(learned_generators, propensity_fn, inferred_stoich)
    
    # Log aggregated results
    aggregated_log = Dict()
    for j in 1:length(inferred_stoich)
        stats = aggregated_stats[j]
        true_k = true_rates[j]
        
        aggregated_log["reaction_$j"] = Dict(
            "median" => stats.median,
            "mean" => stats.mean,
            "std" => stats.std,
            "n_transitions" => stats.n_transitions,
            "true_rate" => true_k,
            "relative_error" => isnan(stats.median) ? NaN : abs(stats.median - true_k) / true_k,
            "absolute_error" => isnan(stats.median) ? NaN : abs(stats.median - true_k)
        )
        
        if verbose && stats.n_transitions > 0
            err_pct = abs(stats.median - true_k) / true_k * 100
            println("  R$j: $(round(stats.median, sigdigits=4)) (true: $true_k, error: $(round(err_pct, digits=1))%, n=$(stats.n_transitions))")
        end
    end
    
    logger.data["statistics"]["aggregated_rates"] = aggregated_log
    
    # ========================================================================
    # STEP 7: ERROR DECOMPOSITION ANALYSIS
    # ========================================================================
    
    if analyze_convergence && verbose
        println("\n" * "="^80)
        println("STEP 7: ERROR DECOMPOSITION")
        println("="^80)
    end
    
    if analyze_convergence
    error_decomposition = analyze_error_sources(
    learned_generators, aggregated_stats, true_rates, inferred_stoich, propensity_fn
)
        
        logger.data["statistics"]["error_decomposition"] = error_decomposition
        
        if verbose
            println("  Sampling variance contribution: $(round(error_decomposition["sampling_variance_contribution"]*100, digits=1))%")
            println("  Bias contribution: $(round(error_decomposition["bias_contribution"]*100, digits=1))%")
            println("  Window inconsistency: $(round(error_decomposition["window_inconsistency"], sigdigits=3))")
        end
    end
    
    # ========================================================================
    # STEP 8: FINAL SUMMARY
    # ========================================================================
    
    # Compute summary statistics
    errors = [aggregated_log["reaction_$j"]["relative_error"] 
              for j in 1:length(inferred_stoich) 
              if !isnan(aggregated_log["reaction_$j"]["relative_error"])]
    
    logger.data["statistics"]["summary"] = Dict(
        "mean_relative_error" => isempty(errors) ? NaN : mean(errors),
        "median_relative_error" => isempty(errors) ? NaN : median(errors),
        "max_relative_error" => isempty(errors) ? NaN : maximum(errors),
        "min_relative_error" => isempty(errors) ? NaN : minimum(errors),
        "n_reactions_recovered" => length(errors),
        "total_windows" => length(windows),
        "total_transitions" => sum(aggregated_log["reaction_$j"]["n_transitions"] for j in 1:length(inferred_stoich))
    )
    
    # Save log
    save_log!(logger)
    
    if verbose
        println("\n" * "="^80)
        println("EXPERIMENT COMPLETE")
        println("="^80)
        println("\n📊 Results logged to: $(logger.log_dir)")
        println("\nSummary:")
        println("  Mean error: $(round(mean(errors)*100, digits=1))%")
        println("  Median error: $(round(median(errors)*100, digits=1))%")
        println("  Best reaction: $(round(minimum(errors)*100, digits=1))%")
        println("  Worst reaction: $(round(maximum(errors)*100, digits=1))%")
    end
    
    # Return result object
    return ExperimentResult(
        rn, true_rates, learned_generators, inferred_stoich,
        X_global, config, ssa_trajs, (T, distrs), windows
    ), logger
end

# ============================================================================
# THEORETICAL BOUNDS COMPUTATION
# ============================================================================

"""
    compute_identifiability_bounds(distrs, T, config, X, stoich_vecs)

Compute theoretical bounds from Theorems 2.7, 2.12, 2.13.
"""
function compute_identifiability_bounds(distrs, T, config, X, stoich_vecs)
    # Estimate effective flux
    w_eff_samples = Float64[]
    
    for (i, dist) in enumerate(distrs[1:min(100, length(distrs))])
        # Approximate exit rates from local transitions
        w_local = 0.0
        for (state, prob) in dist
            # Simple heuristic: exit rate ~ number of possible reactions * typical rate
            n_possible_reactions = length(stoich_vecs)
            w_local += prob * n_possible_reactions * 0.1  # Assume typical rate ~ 0.1
        end
        push!(w_eff_samples, w_local)
    end
    
    w_eff_mean = mean(w_eff_samples)
    
    # Estimate spectral gap from distribution decay
    # Fit exponential decay ||p(t) - p_ss|| ~ exp(-λ_gap * t)
    if length(distrs) > 10
        # Simple heuristic: measure time to reach near-equilibrium
        support_sizes = [length(d) for d in distrs]
        # When support stabilizes, we're near equilibrium
        stabilization_idx = findfirst(i -> std(support_sizes[i:min(i+10, end)]) < 0.1 * mean(support_sizes[i:min(i+10, end)]), 1:length(support_sizes)-10)
        t_relax = stabilization_idx !== nothing ? T[stabilization_idx] : T[end]
        lambda_gap_estimate = 1.0 / t_relax
    else
        lambda_gap_estimate = 0.01  # Default fallback
    end
    
    # Compute identifiability metrics
    uniqueness_product = config.dt_window * w_eff_mean
    uniqueness_satisfied = uniqueness_product < 0.3
    
    conditioning_ratio = w_eff_mean / lambda_gap_estimate
    
    return Dict(
        "w_eff_mean" => w_eff_mean,
        "w_eff_std" => std(w_eff_samples),
        "lambda_gap_estimate" => lambda_gap_estimate,
        "relaxation_time_estimate" => 1.0 / lambda_gap_estimate,
        "uniqueness_product" => uniqueness_product,
        "uniqueness_satisfied" => uniqueness_satisfied,
        "uniqueness_threshold" => 0.3,
        "conditioning_ratio" => conditioning_ratio,
        "conditioning_threshold" => 5.0,
        "conditioning_satisfied" => conditioning_ratio >= 5.0
    )
end

# ============================================================================
# ERROR ANALYSIS
# ============================================================================

"""
    analyze_error_sources(learned_generators, aggregated_stats, true_rates, stoich_vecs, propensity_fn)

Decompose error into sampling variance, bias, and window inconsistency.
"""
function analyze_error_sources(learned_generators, aggregated_stats, true_rates, stoich_vecs, propensity_fn)
    # For each reaction, collect estimates across windows
    window_estimates = Dict{Int, Vector{Float64}}()
    
    for gen in learned_generators
        rate_stats = extract_rates(gen, propensity_fn, stoich_vecs)
        for (j, stats) in rate_stats
            if stats.n_transitions > 0 && !isnan(stats.median)
                if !haskey(window_estimates, j)
                    window_estimates[j] = Float64[]
                end
                push!(window_estimates[j], stats.median)
            end
        end
    end
    
    # Compute variance across windows
    window_variances = Dict(j => length(estimates) > 1 ? var(estimates) : 0.0 
                           for (j, estimates) in window_estimates)
    
    # Compute bias
    biases = Dict{Int, Float64}()
    for j in 1:length(stoich_vecs)
        stats = aggregated_stats[j]
        if !isnan(stats.median)
            biases[j] = abs(stats.median - true_rates[j])
        end
    end
    
    # Estimate sampling variance contribution
    sampling_variances = Dict{Int, Float64}()
    for j in 1:length(stoich_vecs)
        stats = aggregated_stats[j]
        if stats.n_transitions > 0 && !isnan(stats.std)
            sampling_variances[j] = (stats.std)^2 / max(stats.n_transitions, 1)
        end
    end
    
    # Compute contributions
    total_variance = sum(values(window_variances))
    total_sampling_var = sum(values(sampling_variances))
    total_bias = sum(values(biases))
    
    window_means = [mean(estimates) for estimates in values(window_estimates) if length(estimates) > 1]
    window_inconsistency = isempty(window_means) ? 0.0 : std(window_means)
    
    return Dict(
        "window_variances" => window_variances,
        "sampling_variances" => sampling_variances,
        "biases" => biases,
        "sampling_variance_contribution" => total_sampling_var / (total_sampling_var + total_variance + 1e-10),
        "bias_contribution" => total_bias / (total_bias + total_variance + 1e-10),
        "window_inconsistency" => window_inconsistency
    )
end
# ============================================================================
# COMPARATIVE EXPERIMENTS
# ============================================================================

# ============================================================================
# COMPARATIVE EXPERIMENTS
# ============================================================================

"""
    create_modified_config(base_config, param_name, param_value)

Create a new InverseProblemConfig with one modified parameter.
"""
function create_modified_config(base_config::InverseProblemConfig, param_name::Symbol, param_value)
    # Get all field names
    fields = fieldnames(InverseProblemConfig)
    
    # Create dictionary of current values
    kwargs = Dict{Symbol, Any}()
    for f in fields
        kwargs[f] = getfield(base_config, f)
    end
    
    # Modify the specified field
    kwargs[param_name] = param_value
    
    # Create and return new config using keyword arguments
    return InverseProblemConfig(;kwargs...)
end

"""
    run_parameter_sweep(rn, u0, true_rates, param_name, param_values; kwargs...)

Run experiments across parameter sweep for sensitivity analysis.
"""
function run_parameter_sweep(
    rn::ReactionSystem,
    u0::Vector{Pair{Symbol, Int}},
    true_rates::Vector{Float64},
    param_name::Symbol,
    param_values::Vector;
    base_config = default_config(),
    base_n_trajectories::Int = 5000,
    experiment_name::String = "parameter_sweep"
)
    
    results = []
    
    for (i, param_value) in enumerate(param_values)
        println("\n" * "="^80)
        println("PARAMETER SWEEP: $param_name = $param_value ($i/$(length(param_values)))")
        println("="^80)
        
        # Create modified config or n_trajectories
        if param_name == :n_trajectories
            n_traj = param_value
            config = base_config
        else
            n_traj = base_n_trajectories
            # Create new config with modified parameter
            config = create_modified_config(base_config, param_name, param_value)
        end
        
        # Run experiment
        exp_name = "$(experiment_name)_$(param_name)_$(param_value)"
        result, logger = run_comprehensive_experiment(
            rn, u0, true_rates;
            n_trajectories = n_traj,
            config = config,
            experiment_name = exp_name,
            verbose = false
        )
        
        push!(results, (param_value = param_value, result = result, logger = logger))
    end
    
    return results
end

"""
Create adaptive windows based on system dynamics.
Shorter windows for fast-mixing regions, longer for slow regions.
"""
function create_adaptive_windows(
    T::Vector{Float64},
    distrs::Vector{Distribution},
    config::InverseProblemConfig,
    stoich_vecs::Vector{Vector{Int}}
)
    windows = WindowData[]
    
    # Estimate local mixing rate
    mixing_rates = estimate_local_mixing_rates(T, distrs)
    
    i = 1
    while i <= length(T) - config.snapshots_per_window
        # Adapt window size based on local mixing rate
        mixing_rate = mixing_rates[i]
        
        # Faster mixing → shorter windows (more frequent updates)
        # Slower mixing → longer windows (more data per window)
        adaptive_dt = min(
            config.dt_window,
            max(2.0, 5.0 / mixing_rate)  # At least 5 characteristic times
        )
        
        # Find endpoint
        t_start = T[i]
        t_end = t_start + adaptive_dt
        end_idx = findlast(t -> t <= t_end, T)
        
        if end_idx - i + 1 >= config.snapshots_per_window
            window_times = T[i:end_idx]
            window_dists = distrs[i:end_idx]
            
            push!(windows, WindowData(
                window_idx = length(windows) + 1,
                times = window_times,
                distributions = window_dists,
                stoich_vecs = stoich_vecs
            ))
            
            i = end_idx + 1
        else
            i += 1
        end
    end
    
    return windows
end

function estimate_local_mixing_rates(T::Vector{Float64}, distrs::Vector{Distribution})
    mixing_rates = zeros(length(T))
    
    for i in 1:length(T)-1
        # Measure distribution change
        overlap = distribution_overlap(distrs[i], distrs[i+1])
        dt = T[i+1] - T[i]
        
        # Mixing rate ~ -log(overlap) / dt
        mixing_rates[i] = -log(max(overlap, 1e-10)) / dt
    end
    
    # Smooth
    for i in 2:length(T)-1
        mixing_rates[i] = mean(mixing_rates[max(1,i-5):min(end,i+5)])
    end
    
    return mixing_rates
end

function distribution_overlap(d1::Distribution, d2::Distribution)
    states1 = Set(keys(d1))
    states2 = Set(keys(d2))
    common_states = intersect(states1, states2)
    
    overlap = sum(min(d1[s], d2[s]) for s in common_states)
    return overlap
end

"""
Hierarchical Bayesian aggregation across windows.
Accounts for both within-window and between-window variance.
"""
function extract_rates_hierarchical(
    learned_generators::Vector{LearnedGenerator},
    propensity_fn::Function,
    stoich_vecs::Vector{Vector{Int}}
)
    # Collect window-level estimates
    window_estimates = Dict{Int, Vector{Tuple{Float64, Float64}}}()  # (mean, variance)
    
    for gen in learned_generators
        rate_stats = extract_rates_flux_filtered(gen, propensity_fn, stoich_vecs)
        
        for (j, stats) in rate_stats
            if stats.n_transitions > 5  # Require minimum sample size
                if !haskey(window_estimates, j)
                    window_estimates[j] = Tuple{Float64, Float64}[]
                end
                
                # Collect mean and variance from this window
                push!(window_estimates[j], (stats.mean, stats.std^2))
            end
        end
    end
    
    # Hierarchical estimate: James-Stein shrinkage
    hierarchical_stats = Dict{Int, RateStatistics}()
    
    for (j, estimates) in window_estimates
        means = [e[1] for e in estimates]
        variances = [e[2] for e in estimates]
        
        # Grand mean
        grand_mean = mean(means)
        
        # Between-window variance
        tau2 = var(means)
        
        # Within-window variance (average)
        sigma2_mean = mean(variances)
        
        # Shrinkage weights (James-Stein)
        weights = tau2 ./ (tau2 .+ variances)
        
        # Shrink toward grand mean
        shrunk_estimates = weights .* grand_mean .+ (1 .- weights) .* means
        
        # Final estimate: weighted average
        final_mean = mean(shrunk_estimates)
        final_std = sqrt(tau2 + sigma2_mean / length(means))
        
        hierarchical_stats[j] = RateStatistics(
            median = median(shrunk_estimates),
            mean = final_mean,
            std = final_std,
            min = minimum(means),
            max = maximum(means),
            n_transitions = sum(length(gen.X_local) for gen in learned_generators)
        )
    end
    
    return hierarchical_stats
end

"""
Extract rates with flux-based filtering.
Only use transitions that carry significant probability flux.
"""
function extract_rates_flux_filtered(
    gen::LearnedGenerator,
    propensity_fn::Function,
    stoich_vecs::Vector{Vector{Int}};
    flux_threshold::Float64 = 1e-6  # Minimum flux to consider
)
    rate_samples = Dict{Int, Vector{Float64}}()
    flux_weights = Dict{Int, Vector{Float64}}()  # NEW
    
    for (i, state_i) in enumerate(gen.X_local)
        state_vec = [state_i[k] for k in 1:length(state_vec)]
        
        for j in 1:length(stoich_vecs)
            target_vec = state_vec + stoich_vecs[j]
            target_ci = CartesianIndex(target_vec...)
            
            target_idx = findfirst(==(target_ci), gen.X_local)
            if !isnothing(target_idx)
                propensity = propensity_fn(state_vec, j)
                
                if propensity > 0 && gen.A_learned[target_idx, i] > 0
                    rate = gen.A_learned[target_idx, i] / propensity
                    
                    # Compute flux through this transition
                    # Flux = rate × propensity × probability
                    # We don't have p(state) here, so approximate with uniform
                    flux = rate * propensity / length(gen.X_local)
                    
                    if flux > flux_threshold  # FILTER
                        if !haskey(rate_samples, j)
                            rate_samples[j] = Float64[]
                            flux_weights[j] = Float64[]
                        end
                        push!(rate_samples[j], rate)
                        push!(flux_weights[j], flux)
                    end
                end
            end
        end
    end
    
    # Compute flux-weighted statistics
    rate_stats = Dict{Int, RateStatistics}()
    for j in 1:length(stoich_vecs)
        if haskey(rate_samples, j) && !isempty(rate_samples[j])
            samples = rate_samples[j]
            weights = flux_weights[j]
            weights_norm = weights ./ sum(weights)
            
            # Weighted statistics
            weighted_mean = sum(samples .* weights_norm)
            weighted_median = median(samples)  # Could implement weighted median
            
            rate_stats[j] = RateStatistics(
                median = weighted_median,
                mean = weighted_mean,
                std = sqrt(sum(weights_norm .* (samples .- weighted_mean).^2)),
                min = minimum(samples),
                max = maximum(samples),
                n_transitions = length(samples)
            )
        end
    end
    
    return rate_stats
end

# Export
export run_comprehensive_experiment, run_parameter_sweep, ExperimentLogger
export compute_identifiability_bounds, analyze_error_sources1
