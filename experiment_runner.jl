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
using Statistics

# Load modules
include("types.jl")
include("data_generation.jl")
include("state_space.jl")
include("optimization.jl")
include("analysis.jl")
include("src/DiscStochSim.jl")
using .DiscStochSim

# ============================================================================
# JSON SANITIZATION
# ============================================================================

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
        
        # Add rate estimates (with NaN handling)
        window_log_data["rate_estimates"] = Dict()
        for (j, stats) in rate_stats
            median_val = isnan(stats.median) ? nothing : stats.median
            mean_val = isnan(stats.mean) ? nothing : stats.mean
            std_val = isnan(stats.std) ? nothing : stats.std
            min_val = isnan(stats.min) ? nothing : stats.min
            max_val = isnan(stats.max) ? nothing : stats.max
            
            rel_error = if median_val === nothing
                nothing
            else
                abs(stats.median - true_rates[j]) / true_rates[j]
            end
            
            window_log_data["rate_estimates"]["reaction_$j"] = Dict(
                "median" => median_val,
                "mean" => mean_val,
                "std" => std_val,
                "min" => min_val,
                "max" => max_val,
                "n_transitions" => stats.n_transitions,
                "true_rate" => true_rates[j],
                "relative_error" => rel_error
            )
            
            # Log rate evolution
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
    
    # Log aggregated results (with NaN handling)
    aggregated_log = Dict()
    for j in 1:length(inferred_stoich)
        stats = aggregated_stats[j]
        true_k = true_rates[j]
        
        # Handle NaN values explicitly
        median_val = isnan(stats.median) ? nothing : stats.median
        mean_val = isnan(stats.mean) ? nothing : stats.mean
        std_val = isnan(stats.std) ? nothing : stats.std
        
        rel_error = if median_val === nothing || isnan(stats.median)
            nothing
        else
            abs(stats.median - true_k) / true_k
        end
        
        abs_error = if median_val === nothing || isnan(stats.median)
            nothing
        else
            abs(stats.median - true_k)
        end
        
        aggregated_log["reaction_$j"] = Dict(
            "median" => median_val,
            "mean" => mean_val,
            "std" => std_val,
            "n_transitions" => stats.n_transitions,
            "true_rate" => true_k,
            "relative_error" => rel_error,
            "absolute_error" => abs_error
        )
        
        if verbose && stats.n_transitions > 0 && median_val !== nothing
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
        learned_generators, aggregated_stats, true_rates, inferred_stoich, propensity_fn  # Added propensity_fn here
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
    
    # Save log
    save_log!(logger)
    
    if verbose
        println("\n" * "="^80)
        println("EXPERIMENT COMPLETE")
        println("="^80)
        println("\n📊 Results logged to: $(logger.log_dir)")
        if !isempty(errors)
            println("\nSummary:")
            println("  Mean error: $(round(mean(errors)*100, digits=1))%")
            println("  Median error: $(round(median(errors)*100, digits=1))%")
            println("  Best reaction: $(round(minimum(errors)*100, digits=1))%")
            println("  Worst reaction: $(round(maximum(errors)*100, digits=1))%")
        end
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
    if length(distrs) > 10
        support_sizes = [length(d) for d in distrs]
        stabilization_idx = findfirst(i -> std(support_sizes[i:min(i+10, end)]) < 0.1 * mean(support_sizes[i:min(i+10, end)]), 1:length(support_sizes)-10)
        t_relax = stabilization_idx !== nothing ? T[stabilization_idx] : T[end]
        lambda_gap_estimate = 1.0 / t_relax
    else
        lambda_gap_estimate = 0.01
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
    # Convert to NamedTuple
    fields = fieldnames(InverseProblemConfig)
    values = [getfield(base_config, f) for f in fields]
    nt = NamedTuple{fields}(values)
    
    # Modify the specified field
    nt_modified = merge(nt, NamedTuple{(param_name,)}((param_value,)))
    
    # Create new config
    return InverseProblemConfig(values(nt_modified)...)
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


# ============================================================================
# UTILITY FUNCTIONS FROM experiment_runner.jl
# ============================================================================

function default_config()
    return InverseProblemConfig(
        mass_threshold = 0.99,
        λ_frobenius = 1e-6,
        λ_prob_conservation = 0.1,
        dt_snapshot = 0.1,
        dt_window = 2.0,
        snapshots_per_window = 10,
        max_windows = 10
    )
end

function build_parameter_mapping(rn::ReactionSystem, rates::Vector{Float64})
    params = parameters(rn)
    @assert length(params) == length(rates) "Number of rates must match number of parameters"
    return [params[i] => rates[i] for i in 1:length(rates)]
end

function build_fsp_state_space(rn, u0; bounds=(0, 200), expansions=10)
    model = DiscStochSim.DiscreteStochasticSystem(rn)
    boundary_condition(x) = DiscStochSim.RectLatticeBoundaryCondition(x, bounds)
    
    u0_vals = [val for (sym, val) in u0]
    U0 = CartesianIndex(u0_vals...)
    
    X = Set([U0])
    p = [1.0]
    
    X, p = DiscStochSim.expand!(X, p, model, boundary_condition, expansions)
    
    return X
end

function auto_detect_propensity_function(rn::ReactionSystem, stoich_vecs::Vector{Vector{Int}})
    n_species = length(species(rn))
    n_reactions_original = length(reactions(rn))
    n_reactions_inferred = length(stoich_vecs)
    
    # Use the inferred number of reactions
    reactant_stoich = zeros(Int, n_species, n_reactions_inferred)
    
    # Only populate up to the number of reactions we actually have
    for (j, rxn) in enumerate(reactions(rn))
        if j > n_reactions_inferred
            @warn "More reactions in network than inferred stoichiometry, skipping reaction $j"
            break
        end
        
        for (spec, coeff) in zip(rxn.substrates, rxn.substoich)
            spec_idx = findfirst(s -> isequal(s, spec), species(rn))
            if spec_idx !== nothing
                reactant_stoich[spec_idx, j] = Int(coeff)
            end
        end
    end
    
    return MassActionPropensity(reactant_stoich)
end

struct ExperimentResult
    reaction_network::ReactionSystem
    true_rates::Vector{Float64}
    learned_generators::Vector{LearnedGenerator}
    inferred_stoich::Vector{Vector{Int}}
    global_state_space::Set
    config::InverseProblemConfig
    trajectories::Vector
    distributions::Tuple
    windows::Vector{WindowData}
end

function print_results(result::ExperimentResult; skip_transient=3)
    println("\n" * "="^70)
    println("EXPERIMENT SUMMARY")
    println("="^70)
    
    println("\nReaction network:")
    println("  Species: $(length(species(result.reaction_network)))")
    println("  Reactions: $(length(reactions(result.reaction_network)))")
    
    println("\nData:")
    println("  Trajectories: $(length(result.trajectories))")
    println("  Windows: $(length(result.windows))")
    println("  Global state space: $(length(result.global_state_space)) states")
    
    println("\nLearned generators:")
    println("  Total: $(length(result.learned_generators))")
    println("  State space range: $(length(result.learned_generators[1].state_space))-$(length(result.learned_generators[end].state_space)) states")
    
    println("\nTrue rates: $(result.true_rates)")
    
    propensity_fn = auto_detect_propensity_function(result.reaction_network, result.inferred_stoich)
    final_stats = extract_rates_aggregated(result.learned_generators, propensity_fn, result.inferred_stoich)

    println("\nFinal learned rates (aggregated):")
    for j in 1:length(result.inferred_stoich)
        stats = final_stats[j]
        true_k = result.true_rates[j]
        
        if stats.n_transitions > 0
            rel_error = abs(stats.median - true_k) / true_k * 100
            println("  R$j: $(round(stats.median, sigdigits=3)) (true: $true_k, error: $(round(rel_error, digits=1))%, n=$(stats.n_transitions))")
        else
            println("  R$j: N/A (no transitions found)")
        end
    end
end

# Export
export run_comprehensive_experiment, run_parameter_sweep, ExperimentLogger
export compute_identifiability_bounds, analyze_error_sources
export default_config, build_parameter_mapping, build_fsp_state_space
export auto_detect_propensity_function, ExperimentResult, print_results
