"""
Visualization utilities for inverse problem results.

This module provides functions for plotting learned rates over time,
generator properties, and comparing with true values.
"""

using CairoMakie
using Statistics

include("types.jl")
include("analysis.jl")

"""
    plot_learned_rates_over_time(result::OptimizationResult, true_rates::Vector{Float64},
                                  propensity_fn::PropensityFunction)

Create a plot showing how learned rates evolve over time windows.

# Returns
Makie Figure object
"""
function plot_learned_rates_over_time(
    result::OptimizationResult,
    true_rates::Vector{Float64},
    propensity_fn::PropensityFunction
)
    n_reactions = length(result.inferred_stoich)
    n_windows = length(result.learned_generators)
    
    # Extract data
    times = [lg.t_start for lg in result.learned_generators]
    learned_rates = zeros(n_windows, n_reactions)
    rate_errors = zeros(n_windows, n_reactions)
    
    for (i, lg) in enumerate(result.learned_generators)
        rate_stats = extract_rates(lg, propensity_fn, result.inferred_stoich)
        
        for j in 1:n_reactions
            if rate_stats[j].n_transitions > 0
                learned_rates[i, j] = rate_stats[j].median
                rate_errors[i, j] = rate_stats[j].std
            else
                learned_rates[i, j] = NaN
                rate_errors[i, j] = NaN
            end
        end
    end
    
    # Create plot
    fig = Figure(resolution = (1200, 400 * n_reactions))
    
    for j in 1:n_reactions
        ax = Axis(fig[j, 1],
                 xlabel = "Time",
                 ylabel = "Rate constant",
                 title = "Reaction $j: $(result.inferred_stoich[j])")
        
        # Plot learned rates with error bands
        valid_idx = .!isnan.(learned_rates[:, j])
        if any(valid_idx)
            t_valid = times[valid_idx]
            rates_valid = learned_rates[valid_idx, j]
            errors_valid = rate_errors[valid_idx, j]
            
            # Main line
            lines!(ax, t_valid, rates_valid, 
                  color = :blue, linewidth = 2, label = "Learned")
            
            # Error bands
            band!(ax, t_valid, 
                 rates_valid .- errors_valid,
                 rates_valid .+ errors_valid,
                 color = (:blue, 0.2))
        end
        
        # True rate (horizontal line)
        hlines!(ax, [true_rates[j]], 
               color = :red, linestyle = :dash, linewidth = 2,
               label = "True")
        
        axislegend(ax, position = :rt)
    end
    
    return fig
end

"""
    plot_generator_heatmap(learned_gen::LearnedGenerator; max_states=20)

Create a heatmap visualization of the learned generator matrix.

# Arguments
- `learned_gen`: LearnedGenerator to visualize
- `max_states`: Maximum number of states to show (default: 20)
"""
function plot_generator_heatmap(learned_gen::LearnedGenerator; max_states=20)
    A = learned_gen.A
    n = min(size(A, 1), max_states)
    
    fig = Figure(resolution = (800, 700))
    
    ax = Axis(fig[1, 1],
             xlabel = "From state",
             ylabel = "To state", 
             title = "Generator Matrix ($(n)×$(n) block)",
             aspect = DataAspect())
    
    # Plot heatmap (use log scale for better visibility)
    A_plot = A[1:n, 1:n]
    A_log = sign.(A_plot) .* log10.(abs.(A_plot) .+ 1e-10)
    
    hm = heatmap!(ax, A_log, colormap = :RdBu)
    Colorbar(fig[1, 2], hm, label = "log₁₀|A|")
    
    return fig
end

"""
    plot_rate_convergence(result::OptimizationResult, true_rates::Vector{Float64},
                         propensity_fn::PropensityFunction)

Plot relative errors in learned rates over time.
"""
function plot_rate_convergence(
    result::OptimizationResult,
    true_rates::Vector{Float64},
    propensity_fn::PropensityFunction
)
    n_reactions = length(result.inferred_stoich)
    n_windows = length(result.learned_generators)
    
  # Extract data
    times = [lg.t_start for lg in result.learned_generators]
    rel_errors = zeros(n_windows, n_reactions)
    
    for (i, lg) in enumerate(result.learned_generators)
        rate_stats = extract_rates(lg, propensity_fn, result.inferred_stoich)
        
        for j in 1:n_reactions
            if rate_stats[j].n_transitions > 0
                rel_errors[i, j] = abs(rate_stats[j].median - true_rates[j]) / true_rates[j] * 100
            else
                rel_errors[i, j] = NaN
            end
        end
    end
    
    # Create plot
    fig = Figure(resolution = (1000, 600))
    
    ax = Axis(fig[1, 1],
             xlabel = "Time",
             ylabel = "Relative Error (%)",
             title = "Rate Estimation Error Over Time",
             yscale = log10)
    
    colors = [:blue, :red, :green, :orange, :purple]
    
    for j in 1:n_reactions
        valid_idx = .!isnan.(rel_errors[:, j])
        if any(valid_idx)
            lines!(ax, times[valid_idx], rel_errors[valid_idx, j],
                  color = colors[mod1(j, length(colors))],
                  linewidth = 2,
                  label = "Reaction $j")
        end
    end
    
    # Reference line at 1% and 10%
    hlines!(ax, [1.0, 10.0], color = :gray, linestyle = :dash, alpha = 0.5)
    
    axislegend(ax, position = :rt)
    
    return fig
end

"""
    plot_state_space_size(result::OptimizationResult)

Plot how local state space size changes over windows.
"""
function plot_state_space_size(result::OptimizationResult)
    times = [lg.t_start for lg in result.learned_generators]
    sizes = [length(lg.state_space) for lg in result.learned_generators]
    
    fig = Figure(resolution = (800, 500))
    
    ax = Axis(fig[1, 1],
             xlabel = "Time",
             ylabel = "State Space Size",
             title = "Local State Space Evolution")
    
    lines!(ax, times, sizes, color = :blue, linewidth = 2)
    scatter!(ax, times, sizes, color = :blue, markersize = 10)
    
    return fig
end

"""
    plot_distribution_snapshot(dist::Dict, title="Distribution")

Visualize a probability distribution (works best for 2D projections).
"""
function plot_distribution_snapshot(dist::Dict, title="Distribution")
    # Extract states and probabilities
    states = collect(keys(dist))
    probs = collect(values(dist))
    
    # Sort by probability
    sorted_idx = sortperm(probs, rev=true)
    top_n = min(20, length(sorted_idx))
    
    fig = Figure(resolution = (800, 600))
    
    ax = Axis(fig[1, 1],
             xlabel = "State index (sorted by probability)",
             ylabel = "Probability",
             title = title)
    
    barplot!(ax, 1:top_n, probs[sorted_idx[1:top_n]],
            color = :blue, alpha = 0.7)
    
    return fig
end

"""
    save_all_plots(result::OptimizationResult, true_rates::Vector{Float64},
                   propensity_fn::PropensityFunction; 
                   output_dir="plots")

Generate and save all standard plots for an optimization result.
"""
function save_all_plots(
    result::OptimizationResult,
    true_rates::Vector{Float64},
    propensity_fn::PropensityFunction;
    output_dir="plots"
)
    mkpath(output_dir)
    
    println("Generating plots...")
    
    # 1. Rates over time
    println("  - Learned rates over time")
    fig1 = plot_learned_rates_over_time(result, true_rates, propensity_fn)
    save(joinpath(output_dir, "rates_over_time.png"), fig1)
    
    # 2. Rate convergence
    println("  - Rate convergence")
    fig2 = plot_rate_convergence(result, true_rates, propensity_fn)
    save(joinpath(output_dir, "rate_convergence.png"), fig2)
    
    # 3. State space size
    println("  - State space size evolution")
    fig3 = plot_state_space_size(result)
    save(joinpath(output_dir, "state_space_size.png"), fig3)
    
    # 4. Generator heatmap (first and last windows)
    if length(result.learned_generators) >= 2
        println("  - Generator heatmaps")
        fig4a = plot_generator_heatmap(result.learned_generators[1])
        save(joinpath(output_dir, "generator_first_window.png"), fig4a)
        
        fig4b = plot_generator_heatmap(result.learned_generators[end])
        save(joinpath(output_dir, "generator_last_window.png"), fig4b)
    end
    
    println("All plots saved to $output_dir/")
end
