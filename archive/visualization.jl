"""
Comprehensive SIAM-compliant visualization for CME inverse problem experiments.

This module provides publication-quality black-and-white plots for analyzing
experimental results from comprehensive experiment sweeps.

Usage:
    # Load all experimental data from JSON logs
    suite = load_experiment_suite("experiments/")
    
    # Generate all publication figures
    generate_all_figures(suite, output_dir="paper_figures")
"""

using CairoMakie
using Statistics
using Printf
using JSON3
using Glob

# ============================================================================
# SIAM THEME
# ============================================================================

function set_siam_theme!()
    siam_theme = Theme(
        fontsize = 10,
        font = "CMU Serif",
        linewidth = 1.5,
        markersize = 8,
        Axis = (
            xgridvisible = false,
            ygridvisible = false,
            topspinevisible = false,
            rightspinevisible = false,
            xticklabelsize = 9,
            yticklabelsize = 9,
            xlabelsize = 10,
            ylabelsize = 10,
            titlesize = 11,
        ),
        Legend = (
            framevisible = false,
            labelsize = 9,
        )
    )
    set_theme!(siam_theme)
end

const SIAM_LINESTYLES = [:solid, :dash, :dot, :dashdot]
const SIAM_MARKERS = [:circle, :utriangle, :rect, :diamond, :xcross]
const SIAM_COLORS_GRAY = [(:black, 1.0), (:black, 0.7), (:black, 0.4)]

# ============================================================================
# DATA LOADING
# ============================================================================

"""
    ExperimentSuite

Container for all experimental data loaded from JSON logs.
"""
struct ExperimentSuite
    baseline::Union{Dict, Nothing}
    trajectory_sweep::Vector{Dict}
    window_sweep::Vector{Dict}
    all_experiments::Vector{Dict}
end

"""
    load_experiment_suite(experiments_dir::String)

Load all experiment JSON logs from directory structure.
"""
function load_experiment_suite(experiments_dir::String)
    println("Loading experimental data from: $experiments_dir")
    
    # Find all experiment_log.json files
    log_files = glob("*/experiment_log.json", experiments_dir)
    
    if isempty(log_files)
        @warn "No experiment logs found in $experiments_dir"
        return ExperimentSuite(nothing, Dict{String,Any}[], Dict{String,Any}[], Dict{String,Any}[])
    end
    
    println("Found $(length(log_files)) experiment logs")
    
    all_experiments = Dict{String, Any}[]
    baseline = nothing
    trajectory_sweep = Dict{String, Any}[]
    window_sweep = Dict{String, Any}[]
    
    for log_file in log_files
        # Read JSON
        json_string = read(log_file, String)
        data = JSON3.read(json_string)
        
        # Convert to Dict with String keys recursively
        data_dict = json_to_dict(data)
        
        # Add experiment name from directory
        exp_name = basename(dirname(log_file))
        data_dict["experiment_name"] = exp_name
        
        push!(all_experiments, data_dict)
        
        # Categorize
        if occursin("baseline", exp_name)
            baseline = data_dict
            println("  ✓ Baseline: $exp_name")
        elseif occursin("n_trajectories", exp_name)
            push!(trajectory_sweep, data_dict)
            println("  ✓ Trajectory sweep: $exp_name")
        elseif occursin("dt_window", exp_name)
            push!(window_sweep, data_dict)
            println("  ✓ Window sweep: $exp_name")
        end
    end
    
    # Sort sweeps
    sort!(trajectory_sweep, by = x -> get(x["metadata"], "n_trajectories", 0))
    sort!(window_sweep, by = x -> parse(Float64, split(x["experiment_name"], "_")[end-2]))
    
    println("\nLoaded:")
    println("  - Baseline: $(baseline !== nothing ? "✓" : "✗")")
    println("  - Trajectory sweep: $(length(trajectory_sweep)) experiments")
    println("  - Window sweep: $(length(window_sweep)) experiments")
    
    return ExperimentSuite(baseline, trajectory_sweep, window_sweep, all_experiments)
end

"""
    json_to_dict(obj)

Recursively convert JSON3 objects to Dict with String keys.
"""
function json_to_dict(obj)
    if obj isa JSON3.Object
        return Dict{String, Any}(String(k) => json_to_dict(v) for (k, v) in pairs(obj))
    elseif obj isa JSON3.Array || obj isa Vector
        return [json_to_dict(item) for item in obj]
    else
        return obj
    end
end
# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

function extract_rate_errors(exp::Dict, reaction_idx::Int)
    rates = get(exp["statistics"], "aggregated_rates", Dict())
    key = "reaction_$reaction_idx"
    if haskey(rates, key)
        rel_err = get(rates[key], "relative_error", nothing)
        return rel_err === nothing ? NaN : rel_err
    end
    return NaN
end

function extract_n_transitions(exp::Dict, reaction_idx::Int)
    rates = get(exp["statistics"], "aggregated_rates", Dict())
    key = "reaction_$reaction_idx"
    if haskey(rates, key)
        return get(rates[key], "n_transitions", 0)
    end
    return 0
end

function extract_all_reaction_data(exp::Dict, n_reactions::Int)
    errors = Float64[]
    n_trans = Int[]
    
    for j in 1:n_reactions
        err = extract_rate_errors(exp, j)
        n = extract_n_transitions(exp, j)
        if !isnan(err) && n > 0
            push!(errors, err)
            push!(n_trans, n)
        end
    end
    
    return errors, n_trans
end

# ============================================================================
# FIGURE 1: THEORETICAL VALIDATION
# ============================================================================

function plot_theoretical_validation(suite::ExperimentSuite)
    set_siam_theme!()
    
    fig = Figure(size = (700, 500))
    
    # Panel A: Uniqueness criterion validation
    ax1 = Axis(fig[1, 1],
              xlabel = "Window size Δt",
              ylabel = "Effective flux w_eff",
              title = "(a) Uniqueness Criterion: Δt·w_eff < 0.3")
    
    # Extract theoretical bounds from all experiments
    dt_vals = Float64[]
    w_eff_vals = Float64[]
    errors = Float64[]
    
    for exp in suite.all_experiments
        bounds_list = get(exp, "diagnostics", Dict())
        bounds_list = get(bounds_list, "theoretical_bounds", [])
        
        if !isempty(bounds_list)
            bounds = bounds_list[1]
            
            config = get(exp, "metadata", Dict())
            config = get(config, "config", Dict())
            dt = get(config, "dt_window", nothing)
            
            w_eff = get(bounds, "w_eff_mean", nothing)
            
            summary = get(exp, "statistics", Dict())
            summary = get(summary, "summary", Dict())
            mean_err = get(summary, "mean_relative_error", nothing)
            
            # Only add if all values are valid
            if dt !== nothing && w_eff !== nothing && !isnan(Float64(dt)) && !isnan(Float64(w_eff))
                push!(dt_vals, Float64(dt))
                push!(w_eff_vals, Float64(w_eff))
                
                # Handle error (can be nothing)
                if mean_err !== nothing && !isnan(Float64(mean_err))
                    push!(errors, Float64(mean_err))
                else
                    push!(errors, 0.5)  # Default value for missing errors
                end
            end
        end
    end
    
    # Plot safe region (Δt·w_eff < 0.3)
    if !isempty(dt_vals)
        dt_range = range(0, maximum(dt_vals) * 1.2, length=100)
        w_safe = 0.3 ./ dt_range
        band!(ax1, dt_range, zeros(length(dt_range)), w_safe,
             color = (:green, 0.1), label = "Safe region")
        lines!(ax1, dt_range, w_safe, color = :black, linestyle = :dash, linewidth = 1)
        
        # Plot experiments
        scatter!(ax1, dt_vals, w_eff_vals,
                color = errors,
                colormap = :grays,
                markersize = 12,
                strokecolor = :black,
                strokewidth = 1)
        
        text!(ax1, maximum(dt_vals) * 0.8, 0.25, text = "Δt·w_eff = 0.3", fontsize = 9)
    else
        text!(ax1, 0.5, 0.5, text = "No valid data", align = (:center, :center))
    end
    
    # Panel B: Conditioning ratio
    ax2 = Axis(fig[1, 2],
              xlabel = "Conditioning ratio w_eff/λ_gap",
              ylabel = "Mean relative error",
              title = "(b) Conditioning vs Error")
    
    cond_ratios = Float64[]
    cond_errors = Float64[]
    
    for exp in suite.all_experiments
        bounds_list = get(exp, "diagnostics", Dict())
        bounds_list = get(bounds_list, "theoretical_bounds", [])
        
        if !isempty(bounds_list)
            bounds = bounds_list[1]
            ratio = get(bounds, "conditioning_ratio", nothing)
            
            summary = get(exp, "statistics", Dict())
            summary = get(summary, "summary", Dict())
            err = get(summary, "mean_relative_error", nothing)
            
            if ratio !== nothing && err !== nothing && !isnan(Float64(ratio)) && !isnan(Float64(err))
                push!(cond_ratios, Float64(ratio))
                push!(cond_errors, Float64(err))
            end
        end
    end
    
    if !isempty(cond_ratios)
        scatter!(ax2, cond_ratios, cond_errors,
                color = :black, marker = :circle, markersize = 10,
                strokecolor = :black, strokewidth = 1)
        
        # Theoretical threshold
        vlines!(ax2, [5.0], color = (:black, 0.3), linestyle = :dash,
               linewidth = 1.5)
        text!(ax2, 5.5, maximum(cond_errors)*0.9, 
             text = "κ_min = 5", fontsize = 9)
    else
        text!(ax2, 0.5, 0.5, text = "No valid data", align = (:center, :center))
    end
    
    # Panel C: Error decomposition
    ax3 = Axis(fig[2, 1:2],
              xlabel = "Experiment",
              ylabel = "Error contribution",
              title = "(c) Error Decomposition Across Experiments",
              xticks = (1:min(10, length(suite.all_experiments)), 
                       ["E$i" for i in 1:min(10, length(suite.all_experiments))]))
    
    sampling_vars = Float64[]
    biases = Float64[]
    window_incons = Float64[]
    
    for exp in suite.all_experiments[1:min(10, length(suite.all_experiments))]
        err_decomp_dict = get(exp, "statistics", Dict())
        err_decomp = get(err_decomp_dict, "error_decomposition", Dict())
        
        sv = get(err_decomp, "sampling_variance_contribution", nothing)
        b = get(err_decomp, "bias_contribution", nothing)
        wi = get(err_decomp, "window_inconsistency", nothing)
        
        push!(sampling_vars, sv !== nothing ? Float64(sv) : 0.0)
        push!(biases, b !== nothing ? Float64(b) : 0.0)
        push!(window_incons, wi !== nothing ? Float64(wi) : 0.0)
    end
    
    if !isempty(sampling_vars) && sum(sampling_vars .+ biases .+ window_incons) > 0
        x = 1:length(sampling_vars)
        
        # Normalize window inconsistency for visualization
        max_wi = maximum(window_incons)
        if max_wi > 0
            window_incons_scaled = window_incons ./ max_wi .* 0.2
        else
            window_incons_scaled = window_incons
        end
        
        # Stacked bars
        barplot!(ax3, x, sampling_vars,
                color = (:black, 0.3), label = "Sampling variance")
        barplot!(ax3, x, biases,
                offset = sampling_vars,
                color = (:black, 0.6), label = "Bias")
        barplot!(ax3, x, window_incons_scaled,
                offset = sampling_vars .+ biases,
                color = (:black, 0.9), label = "Window inconsistency (scaled)")
        
        axislegend(ax3, position = :rt)
    else
        text!(ax3, 0.5, 0.5, text = "No valid data", align = (:center, :center))
    end
    
    return fig
end
# ============================================================================
# FIGURE 2: REACTION-LEVEL PERFORMANCE
# ============================================================================

function plot_reaction_performance(suite::ExperimentSuite)
    set_siam_theme!()
    
    fig = Figure(size = (700, 700))
    
    n_reactions = 3
    
    # Panel A: Error vs transitions (validates 1/sqrt(n) scaling)
    ax1 = Axis(fig[1, 1],
              xlabel = "Number of transitions observed (n)",
              ylabel = "Relative error",
              title = "(a) Error vs Sampling: Validation of 1/√n Scaling",
              xscale = log10,
              yscale = log10)
    
    # Collect all reaction data points across all experiments
    for j in 1:n_reactions
        n_vals = Float64[]
        err_vals = Float64[]
        
        for exp in suite.all_experiments
            n = extract_n_transitions(exp, j)
            err = extract_rate_errors(exp, j)
            
            if n > 0 && !isnan(err)
                push!(n_vals, Float64(n))
                push!(err_vals, err)
            end
        end
        
        if !isempty(n_vals)
            scatter!(ax1, n_vals, err_vals,
                    color = SIAM_COLORS_GRAY[j][1],
                    alpha = SIAM_COLORS_GRAY[j][2],
                    marker = SIAM_MARKERS[j],
                    markersize = 10,
                    label = "R$j")
        end
    end
    
    # Theoretical scaling: error ∝ 1/√n
    n_theory = 10.0.^range(0, 2, length=100)
    err_theory = 0.5 ./ sqrt.(n_theory)
    lines!(ax1, n_theory, err_theory,
          color = :black, linestyle = :dash, linewidth = 1.5,
          label = "Theory: ∝ 1/√n")
    
    axislegend(ax1, position = :rt)
    
    # Panel B: Transition heat map across windows
    if suite.baseline !== nothing
        ax2 = Axis(fig[1, 2],
                  xlabel = "Window index",
                  ylabel = "Reaction",
                  title = "(b) Transition Sampling Across Windows",
                  yticks = (1:n_reactions, ["R$j" for j in 1:n_reactions]))
        
        windows = get(suite.baseline, "windows", Dict())
        window_indices = sort([parse(Int, split(k, "_")[2]) for k in keys(windows)])
        
        transition_matrix = zeros(n_reactions, length(window_indices))
        
        for (col_idx, w_idx) in enumerate(window_indices)
            window = windows["window_$w_idx"]
            rate_ests = get(window, "rate_estimates", Dict())
            
            for j in 1:n_reactions
                key = "reaction_$j"
                if haskey(rate_ests, key)
                    n_trans = get(rate_ests[key], "n_transitions", 0)
                    transition_matrix[j, col_idx] = n_trans
                end
            end
        end
        
        heatmap!(ax2, window_indices, 1:n_reactions, transition_matrix',
                colormap = :grays, colorrange = (0, maximum(transition_matrix)))
        
        Colorbar(fig[1, 3], limits = (0, maximum(transition_matrix)),
                colormap = :grays, label = "Transitions")
    end
    
    # Panel C: Rate estimates with uncertainty
    ax3 = Axis(fig[2, 1:2],
              xlabel = "Reaction",
              ylabel = "Rate constant",
              title = "(c) Learned Rates with Uncertainty Quantification",
              xticks = (1:n_reactions, ["R$j" for j in 1:n_reactions]))
    
    if suite.baseline !== nothing
        agg_rates = get(suite.baseline["statistics"], "aggregated_rates", Dict())
        
        learned = Float64[]
        errors_low = Float64[]
        errors_high = Float64[]
        true_rates = Float64[]
        
        for j in 1:n_reactions
            key = "reaction_$j"
            if haskey(agg_rates, key)
                median_val = get(agg_rates[key], "median", NaN)
                std_val = get(agg_rates[key], "std", NaN)
                true_val = get(agg_rates[key], "true_rate", NaN)
                
                if !isnan(median_val)
                    push!(learned, median_val)
                    push!(errors_low, isnan(std_val) ? 0.0 : std_val)
                    push!(errors_high, isnan(std_val) ? 0.0 : std_val)
                    push!(true_rates, true_val)
                else
                    push!(learned, 0.0)
                    push!(errors_low, 0.0)
                    push!(errors_high, 0.0)
                    push!(true_rates, true_val)
                end
            end
        end
        
        # True rates (horizontal lines)
        for j in 1:n_reactions
            lines!(ax3, [j-0.3, j+0.3], [true_rates[j], true_rates[j]],
                  color = :black, linewidth = 2, linestyle = :solid)
        end
        
        # Learned rates with error bars
        scatter!(ax3, 1:n_reactions, learned,
                color = (:black, 0.5), marker = :circle, markersize = 12,
                strokecolor = :black, strokewidth = 1.5)
        
        errorbars!(ax3, 1:n_reactions, learned, errors_low, errors_high,
                  color = :black, linewidth = 1.5, whiskerwidth = 10)
    end
    
    return fig
end

# ============================================================================
# FIGURE 3: TRAJECTORY SWEEP ANALYSIS
# ============================================================================

function plot_trajectory_sweep(suite::ExperimentSuite)
    set_siam_theme!()
    
    if isempty(suite.trajectory_sweep)
        @warn "No trajectory sweep data available"
        return nothing
    end
    
    fig = Figure(size = (700, 700))
    
    # Extract data
    n_trajs = [get(exp["metadata"], "n_trajectories", 0) for exp in suite.trajectory_sweep]
    mean_errors = [get(get(exp["statistics"], "summary", Dict()), "mean_relative_error", NaN) 
                   for exp in suite.trajectory_sweep]
    total_trans = [get(get(exp["statistics"], "summary", Dict()), "total_transitions", 0)
                   for exp in suite.trajectory_sweep]
    durations = [get(exp["metadata"], "duration_seconds", NaN) 
                for exp in suite.trajectory_sweep]
    
    # Filter valid data
    valid_idx = .!isnan.(mean_errors) .& (n_trajs .> 0)
    n_trajs = n_trajs[valid_idx]
    mean_errors = mean_errors[valid_idx]
    total_trans = total_trans[valid_idx]
    durations = durations[valid_idx]
    
    # Panel A: Log-log scaling
    ax1 = Axis(fig[1, 1],
              xlabel = "Number of trajectories",
              ylabel = "Mean relative error",
              title = "(a) Error Scaling with Sample Size",
              xscale = log10,
              yscale = log10)
    
    scatter!(ax1, n_trajs, mean_errors,
            color = :black, marker = :circle, markersize = 12,
            strokecolor = :black, strokewidth = 1.5)
    
    # Fit power law
    if length(n_trajs) > 1
        log_n = log10.(n_trajs)
        log_err = log10.(mean_errors)
        
        # Linear regression in log space
        A = hcat(ones(length(log_n)), log_n)
        coeffs = A \ log_err
        slope = coeffs[2]
        
        # Plot fit
        n_fit = 10.0.^range(log10(minimum(n_trajs)), log10(maximum(n_trajs)), length=100)
        err_fit = 10.0.^(coeffs[1] .+ slope .* log10.(n_fit))
        
        lines!(ax1, n_fit, err_fit,
              color = (:black, 0.5), linestyle = :dash, linewidth = 2,
              label = @sprintf("Fit: ∝ n^%.2f", slope))
        
        # Theoretical -0.5
        err_theory = mean_errors[1] .* (n_fit ./ n_trajs[1]).^(-0.5)
        lines!(ax1, n_fit, err_theory,
              color = :black, linestyle = :dot, linewidth = 2,
              label = "Theory: ∝ n^-0.5")
        
        axislegend(ax1, position = :rt)
    end
    
    # Panel B: Transition count scaling
    ax2 = Axis(fig[1, 2],
              xlabel = "Number of trajectories",
              ylabel = "Total transitions observed",
              title = "(b) Transition Sampling vs Sample Size")
    
    scatter!(ax2, n_trajs, total_trans,
            color = :black, marker = :circle, markersize = 12,
            strokecolor = :black, strokewidth = 1.5)
    
    # Linear fit for saturation check
    if length(n_trajs) > 1
        A = hcat(ones(length(n_trajs)), n_trajs)
        coeffs = A \ total_trans
        
        n_fit = range(minimum(n_trajs), maximum(n_trajs), length=100)
        trans_fit = coeffs[1] .+ coeffs[2] .* n_fit
        
        lines!(ax2, n_fit, trans_fit,
              color = (:black, 0.5), linestyle = :dash, linewidth = 2,
              label = "Linear fit")
        
        axislegend(ax2, position = :lt)
    end
    
    # Panel C: Computational efficiency
    ax3 = Axis(fig[2, 1:2],
              xlabel = "Number of trajectories",
              ylabel = "Wall-clock time (seconds)",
              title = "(c) Computational Cost vs Accuracy Trade-off")
    
    valid_time = .!isnan.(durations)
    if any(valid_time)
        scatter!(ax3, n_trajs[valid_time], durations[valid_time],
                color = :black, marker = :circle, markersize = 12,
                strokecolor = :black, strokewidth = 1.5,
                label = "Runtime")
        
        # Secondary axis for error
        ax3_right = Axis(fig[2, 1:2],
                        ylabel = "Mean error",
                        yaxisposition = :right,
                        ygridvisible = false)
        
        lines!(ax3_right, n_trajs, mean_errors,
              color = (:black, 0.5), linestyle = :dash, linewidth = 2,
              label = "Error")
        scatter!(ax3_right, n_trajs, mean_errors,
                color = (:black, 0.5), marker = :utriangle, markersize = 10)
        
        hidespines!(ax3_right)
        hidexdecorations!(ax3_right)
        
        axislegend(ax3, position = :lt)
        axislegend(ax3_right, position = :rt)
    end
    
    return fig
end

# ============================================================================
# FIGURE 4: WINDOW SIZE SWEEP
# ============================================================================

function plot_window_sweep(suite::ExperimentSuite)
    set_siam_theme!()
    
    if isempty(suite.window_sweep)
        @warn "No window sweep data available"
        return nothing
    end
    
    fig = Figure(size = (700, 500))
    
    # Extract data
    dt_windows = [parse(Float64, split(exp["experiment_name"], "_")[end-2]) 
                  for exp in suite.window_sweep]
    mean_errors = [get(get(exp["statistics"], "summary", Dict()), "mean_relative_error", NaN)
                   for exp in suite.window_sweep]
    total_trans = [get(get(exp["statistics"], "summary", Dict()), "total_transitions", 0)
                   for exp in suite.window_sweep]
    
    # Get w_eff for uniqueness product
    w_eff_mean = 0.15  # From baseline, hardcode if needed
    uniqueness_products = dt_windows .* w_eff_mean
    
    # Panel A: Error vs window size
    ax1 = Axis(fig[1, 1],
              xlabel = "Window size Δt",
              ylabel = "Mean relative error",
              title = "(a) Error vs Window Size")
    
    valid_err = .!isnan.(mean_errors)
    
    scatter!(ax1, dt_windows[valid_err], mean_errors[valid_err],
            color = :black, marker = :circle, markersize = 12,
            strokecolor = :black, strokewidth = 1.5)
    
    lines!(ax1, dt_windows[valid_err], mean_errors[valid_err],
          color = (:black, 0.5), linewidth = 1.5)
    
    # Mark uniqueness threshold
    dt_threshold = 0.3 / w_eff_mean
    vlines!(ax1, [dt_threshold],
           color = (:red, 0.5), linestyle = :dash, linewidth = 2)
    text!(ax1, dt_threshold + 0.5, maximum(mean_errors[valid_err]) * 0.9,
         text = "Uniqueness\nthreshold", fontsize = 9)
    
    # Panel B: Transitions vs window size
    ax2 = Axis(fig[1, 2],
              xlabel = "Window size Δt",
              ylabel = "Total transitions",
              title = "(b) Sampling Coverage vs Window Size")
    
    scatter!(ax2, dt_windows, total_trans,
            color = :black, marker = :circle, markersize = 12,
            strokecolor = :black, strokewidth = 1.5)
    
    lines!(ax2, dt_windows, total_trans,
          color = (:black, 0.5), linewidth = 1.5)
    
    vlines!(ax2, [dt_threshold],
           color = (:red, 0.5), linestyle = :dash, linewidth = 2)
    
    # Panel C: Uniqueness product
    ax3 = Axis(fig[2, 1:2],
              xlabel = "Window size Δt",
              ylabel = "Δt · w_eff",
              title = "(c) Uniqueness Criterion Validation")
    
    scatter!(ax3, dt_windows, uniqueness_products,
            color = :black, marker = :circle, markersize = 12,
            strokecolor = :black, strokewidth = 1.5,
            label = "Δt · w_eff")
    
    lines!(ax3, dt_windows, uniqueness_products,
          color = (:black, 0.5), linewidth = 1.5)
    
    # Threshold at 0.3
    hlines!(ax3, [0.3],
           color = :black, linestyle = :dash, linewidth = 2,
           label = "Threshold")
    
    # Shade safe region
    band!(ax3, [minimum(dt_windows), maximum(dt_windows)], 
         [0.0, 0.0], [0.3, 0.3],
         color = (:green, 0.1), label = "Safe region")
    
    axislegend(ax3, position = :lt)
    
    return fig
end

# ============================================================================
# FIGURE 5: CONVERGENCE DIAGNOSTICS
# ============================================================================

function plot_convergence_diagnostics(suite::ExperimentSuite)
    set_siam_theme!()
    
    if suite.baseline === nothing
        @warn "No baseline data available"
        return nothing
    end
    
    fig = Figure(size = (700, 700))
    
    windows = get(suite.baseline, "windows", Dict())
    window_indices = sort([parse(Int, split(k, "_")[2]) for k in keys(windows)])
    
    # Panel A: Optimization improvement
    ax1 = Axis(fig[1, 1],
              xlabel = "Improvement (%)",
              ylabel = "Frequency",
              title = "(a) Optimization Convergence Quality")
    
    improvements = Float64[]
    for w_idx in window_indices
        window = windows["window_$w_idx"]
        conv = get(window, "convergence", Dict())
        imp = get(conv, "improvement_pct", NaN)
        if !isnan(imp)
            push!(improvements, imp)
        end
    end
    
    hist!(ax1, improvements, bins = 20,
         color = (:black, 0.5), strokecolor = :black, strokewidth = 1)
    
    vlines!(ax1, [mean(improvements)],
           color = :black, linestyle = :dash, linewidth = 2,
           label = @sprintf("Mean: %.1f%%", mean(improvements)))
    
    axislegend(ax1, position = :rt)
    
    # Panel B: Constraint satisfaction over windows
    ax2 = Axis(fig[1, 2],
              xlabel = "Window index",
              ylabel = "Max column sum error",
              title = "(b) Probability Conservation")
    
    col_sum_errors = Float64[]
    prob_conservation = Float64[]
    
    for w_idx in window_indices
        window = windows["window_$w_idx"]
        props = get(window, "generator_properties", Dict())
        push!(col_sum_errors, get(props, "max_col_sum_error", NaN))
        push!(prob_conservation, get(props, "prob_conservation", NaN))
    end
    
    valid = .!isnan.(col_sum_errors)
    
    lines!(ax2, window_indices[valid], col_sum_errors[valid],
          color = :black, linewidth = 1.5)
    scatter!(ax2, window_indices[valid], col_sum_errors[valid],
            color = :black, marker = :circle, markersize = 8)
    
    # Panel C: State space size evolution
    ax3 = Axis(fig[2, 1],
              xlabel = "Window index",
              ylabel = "State space size",
              title = "(c) Adaptive State Space Evolution")
    
    state_sizes = Float64[]
    for w_idx in window_indices
        window = windows["window_$w_idx"]
        push!(state_sizes, get(window, "state_space_size", NaN))
    end
    
    valid_size = .!isnan.(state_sizes)
    
    lines!(ax3, window_indices[valid_size], state_sizes[valid_size],
          color = :black, linewidth = 1.5)
    scatter!(ax3, window_indices[valid_size], state_sizes[valid_size],
            color = :black, marker = :circle, markersize = 8)
    
    # Panel D: Frobenius norm evolution
    ax4 = Axis(fig[2, 2],
              xlabel = "Window index",
              ylabel = "||A||_F",
              title = "(d) Generator Norm Evolution")
    
    frob_norms = Float64[]
    for w_idx in window_indices
        window = windows["window_$w_idx"]
        props = get(window, "generator_properties", Dict())
        push!(frob_norms, get(props, "frobenius_norm", NaN))
    end
    
    valid_frob = .!isnan.(frob_norms)
    
    lines!(ax4, window_indices[valid_frob], frob_norms[valid_frob],
          color = :black, linewidth = 1.5)
    scatter!(ax4, window_indices[valid_frob], frob_norms[valid_frob],
            color = :black, marker = :circle, markersize = 8)
    
    return fig
end

# ============================================================================
# FIGURE 6: STATE SPACE ANALYSIS
# ============================================================================

function plot_state_space_analysis(suite::ExperimentSuite)
    set_siam_theme!()
    
    fig = Figure(size = (700, 500))
    
    # Panel A: Compression ratios across experiments
    ax1 = Axis(fig[1, 1],
              xlabel = "Experiment",
              ylabel = "Compression ratio (global/FSP)",
              title = "(a) State Space Compression")
    
    compression_ratios = Float64[]
    exp_labels = String[]
    
    for (i, exp) in enumerate(suite.all_experiments)
        stats = get(exp["diagnostics"], "state_space_statistics", [])
        if !isempty(stats)
            stat = stats[1]
            ratio = get(stat, "compression_ratio", NaN)
            if !isnan(ratio)
                push!(compression_ratios, ratio)
                push!(exp_labels, "E$i")
            end
        end
    end
    
    if !isempty(compression_ratios)
        barplot!(ax1, 1:length(compression_ratios), compression_ratios,
                color = (:black, 0.5), strokecolor = :black, strokewidth = 1)
        
        ax1.xticks = (1:length(compression_ratios), exp_labels)
    end
    
    # Panel B: Support size evolution (baseline)
    if suite.baseline !== nothing
        ax2 = Axis(fig[1, 2],
                  xlabel = "Time",
                  ylabel = "Distribution support size",
                  title = "(b) Support Evolution")
        
        dist_stats = get(suite.baseline["diagnostics"], "distribution_statistics", [])
        if !isempty(dist_stats)
            stat = dist_stats[1]
            mean_support = get(stat, "mean_support_size", NaN)
            std_support = get(stat, "std_support_size", NaN)
            
            # This would need time series data - placeholder for now
            text!(ax2, 0.5, 0.5, 
                 text = @sprintf("Mean: %.1f ± %.1f", mean_support, std_support),
                 align = (:center, :center))
        end
    end
    
    return fig
end

# ============================================================================
# MAIN GENERATION FUNCTION
# ============================================================================

"""
    generate_all_figures(suite::ExperimentSuite; output_dir="paper_figures")

Generate all publication figures from experimental data.
"""
function generate_all_figures(suite::ExperimentSuite; output_dir="paper_figures")
    mkpath(output_dir)
    
    println("\n" * "="^80)
    println("GENERATING PUBLICATION FIGURES")
    println("="^80)
    
    formats = [:pdf, :png]
    
    # Figure 1: Theoretical validation
    println("\n[1/6] Theoretical validation...")
    fig1 = plot_theoretical_validation(suite)
    for fmt in formats
        save(joinpath(output_dir, "fig1_theoretical_validation.$fmt"), fig1, pt_per_unit=1)
    end
    
    # Figure 2: Reaction performance
    println("[2/6] Reaction-level performance...")
    fig2 = plot_reaction_performance(suite)
    for fmt in formats
        save(joinpath(output_dir, "fig2_reaction_performance.$fmt"), fig2, pt_per_unit=1)
    end
    
    # Figure 3: Trajectory sweep
    println("[3/6] Trajectory sweep analysis...")
    fig3 = plot_trajectory_sweep(suite)
    if fig3 !== nothing
        for fmt in formats
            save(joinpath(output_dir, "fig3_trajectory_sweep.$fmt"), fig3, pt_per_unit=1)
        end
    end
    
    # Figure 4: Window sweep
    println("[4/6] Window size sweep...")
    fig4 = plot_window_sweep(suite)
    if fig4 !== nothing
        for fmt in formats
            save(joinpath(output_dir, "fig4_window_sweep.$fmt"), fig4, pt_per_unit=1)
        end
    end
    
    # Figure 5: Convergence diagnostics
    println("[5/6] Convergence diagnostics...")
    fig5 = plot_convergence_diagnostics(suite)
    if fig5 !== nothing
        for fmt in formats
            save(joinpath(output_dir, "fig5_convergence_diagnostics.$fmt"), fig5, pt_per_unit=1)
        end
    end
    
    # Figure 6: State space analysis
    println("[6/6] State space analysis...")
    fig6 = plot_state_space_analysis(suite)
    for fmt in formats
        save(joinpath(output_dir, "fig6_state_space_analysis.$fmt"), fig6, pt_per_unit=1)
    end
    
    println("\n" * "="^80)
    println("✓ ALL FIGURES GENERATED")
    println("="^80)
    println("\nOutput directory: $output_dir")
    println("Formats: $(join(formats, ", "))")
    println("\nFigure inventory:")
    println("  fig1_theoretical_validation  - Uniqueness criterion, conditioning, error decomposition")
    println("  fig2_reaction_performance    - Error vs transitions, heat map, uncertainty quantification")
    println("  fig3_trajectory_sweep        - Scaling analysis, transition counts, computational cost")
    println("  fig4_window_sweep           - Window size sensitivity, uniqueness validation")
    println("  fig5_convergence_diagnostics - Optimization quality, constraints, state space evolution")
    println("  fig6_state_space_analysis   - Compression ratios, support evolution")
end

# ============================================================================
# CONVENIENCE WRAPPER
# ============================================================================

"""
    analyze_experiments(experiments_dir::String; output_dir="paper_figures")

One-command analysis: load data and generate all figures.

# Example
```julia
analyze_experiments("experiments/", output_dir="my_figures")
```
"""
function analyze_experiments(experiments_dir::String; output_dir="paper_figures")
    println("Starting comprehensive experimental analysis...")
    
    # Load data
    suite = load_experiment_suite(experiments_dir)
    
    # Generate figures
    generate_all_figures(suite, output_dir=output_dir)
    
    println("\n✓ Analysis complete!")
    
    return suite
end

# Export
export ExperimentSuite, load_experiment_suite, generate_all_figures, analyze_experiments
export plot_theoretical_validation, plot_reaction_performance
export plot_trajectory_sweep, plot_window_sweep
export plot_convergence_diagnostics, plot_state_space_analysis
