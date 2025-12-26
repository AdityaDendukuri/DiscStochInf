"""
SIAM-compliant visualization for ExperimentResult outputs.

This module provides publication-quality black-and-white plots that work directly
with the ExperimentResult structure from experiment_runner.jl.

Usage:
    result = run_inverse_experiment(rn, u0, true_rates)
    
    # Generate all SIAM figures
    save_experiment_figures_siam(result, output_dir="paper_figures")
    
    # Or individual figures
    fig = plot_experiment_convergence(result)
    save("convergence.pdf", fig, pt_per_unit=1)
"""

using CairoMakie
using Statistics
using Printf

# SIAM-compliant theme settings
function set_siam_theme!()
    siam_theme = Theme(
        fontsize = 10,
        font = "CMU Serif",  # Computer Modern Unicode (LaTeX default)
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

# Line styles and markers for distinguishing series
const SIAM_LINESTYLES = [:solid, :dash, :dot, :dashdot, :dashdotdot]
const SIAM_MARKERS = [:circle, :utriangle, :rect, :diamond, :xcross, :+]

"""
    plot_experiment_convergence(result::ExperimentResult; skip_transient=0)

Plot rate convergence over time windows (SIAM style).

Shows relative errors for all reactions on log scale, demonstrating convergence
of the learning algorithm.

# Arguments
- `result`: ExperimentResult from run_inverse_experiment
- `skip_transient`: Number of initial windows to skip (default: 0)

# Returns
Publication-ready Figure showing convergence
"""
function plot_experiment_convergence(result::ExperimentResult; skip_transient=0)
    set_siam_theme!()
    
    n_reactions = length(result.inferred_stoich)
    n_windows = length(result.learned_generators)
    
    # Extract propensity function
    propensity_fn = auto_detect_propensity_function(result.reaction_network, result.inferred_stoich)
    
    # Extract data
    times = [lg.t_start for lg in result.learned_generators]
    rel_errors = zeros(n_windows, n_reactions)
    
    for (i, lg) in enumerate(result.learned_generators)
        rate_stats = extract_rates_global(lg, result.global_state_space, propensity_fn, result.inferred_stoich)
        
        for j in 1:n_reactions
            if rate_stats[j].n_transitions > 0
                rel_errors[i, j] = abs(rate_stats[j].median - result.true_rates[j]) / result.true_rates[j] * 100
            else
                rel_errors[i, j] = NaN
            end
        end
    end
    
    # Skip transient windows if requested
    start_idx = skip_transient + 1
    if start_idx > n_windows
        @warn "skip_transient=$skip_transient >= n_windows=$n_windows, using all windows"
        start_idx = 1
    end
    
    times = times[start_idx:end]
    rel_errors = rel_errors[start_idx:end, :]
    
    # Create plot
    fig = Figure(size = (500, 350))
    
    ax = Axis(fig[1, 1],
             xlabel = "Time",
             ylabel = "Relative Error (%)",
             title = "Rate Convergence",
             yscale = log10)
    
    # Plot each reaction with different line style and marker
    for j in 1:n_reactions
        valid_idx = .!isnan.(rel_errors[:, j])
        if any(valid_idx)
            t_valid = times[valid_idx]
            err_valid = rel_errors[valid_idx, j]
            
            # Lines
            lines!(ax, t_valid, err_valid,
                  color = :black,
                  linestyle = SIAM_LINESTYLES[mod1(j, length(SIAM_LINESTYLES))],
                  linewidth = 1.5,
                  label = "k$(j)")
            
            # Markers
            scatter!(ax, t_valid, err_valid,
                    color = :black,
                    marker = SIAM_MARKERS[mod1(j, length(SIAM_MARKERS))],
                    markersize = 6)
        end
    end
    
    # Reference lines at 1% and 10% error
    hlines!(ax, [1.0, 10.0], 
           color = (:black, 0.3), 
           linestyle = :dot, 
           linewidth = 1.0)
    
    axislegend(ax, position = :rt)
    
    return fig
end

"""
    plot_experiment_rates_evolution(result::ExperimentResult)

Plot evolution of learned rates over time with error bands (SIAM style).

Shows each rate constant evolving over windows with true value as reference.
"""
function plot_experiment_rates_evolution(result::ExperimentResult)
    set_siam_theme!()
    
    n_reactions = length(result.inferred_stoich)
    n_windows = length(result.learned_generators)
    
    propensity_fn = auto_detect_propensity_function(result.reaction_network, result.inferred_stoich)
    
    # Extract data
    times = [lg.t_start for lg in result.learned_generators]
    learned_rates = zeros(n_windows, n_reactions)
    rate_errors = zeros(n_windows, n_reactions)
    
    for (i, lg) in enumerate(result.learned_generators)
        rate_stats = extract_rates_global(lg, result.global_state_space, propensity_fn, result.inferred_stoich)
        
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
    
    # Create figure
    fig = Figure(size = (500, 150 * n_reactions))
    
    for j in 1:n_reactions
        ax = Axis(fig[j, 1],
                 xlabel = "Time",
                 ylabel = "Rate Constant k$(j)",
                 title = "Reaction $(j): $(format_stoichiometry(result.inferred_stoich[j]))")
        
        # Plot learned rates with error bands
        valid_idx = .!isnan.(learned_rates[:, j])
        if any(valid_idx)
            t_valid = times[valid_idx]
            rates_valid = learned_rates[valid_idx, j]
            errors_valid = rate_errors[valid_idx, j]
            
            # Error bands (light gray)
            band!(ax, t_valid, 
                 rates_valid .- errors_valid,
                 rates_valid .+ errors_valid,
                 color = (:black, 0.15))
            
            # Main line (solid black)
            lines!(ax, t_valid, rates_valid, 
                  color = :black, 
                  linestyle = :solid,
                  linewidth = 1.5)
            
            # Markers
            scatter!(ax, t_valid, rates_valid,
                    color = :black,
                    marker = SIAM_MARKERS[mod1(j, length(SIAM_MARKERS))],
                    markersize = 6,
                    label = "Learned")
        end
        
        # True rate (dashed horizontal line)
        hlines!(ax, [result.true_rates[j]], 
               color = :black, 
               linestyle = :dash, 
               linewidth = 1.5,
               label = "True")
        
        if j == 1  # Legend only on first panel
            axislegend(ax, position = :rt)
        end
    end
    
    return fig
end

"""
    plot_experiment_state_space(result::ExperimentResult)

Plot evolution of local state space sizes over time (SIAM style).
"""
function plot_experiment_state_space(result::ExperimentResult)
    set_siam_theme!()
    
    times = [lg.t_start for lg in result.learned_generators]
    sizes = [length(lg.state_space) for lg in result.learned_generators]
    
    fig = Figure(size = (500, 350))
    
    ax = Axis(fig[1, 1],
             xlabel = "Time",
             ylabel = "Local State Space Size",
             title = "Adaptive State Space Evolution")
    
    # Line
    lines!(ax, times, sizes, 
          color = :black, 
          linewidth = 1.5)
    
    # Scatter points
    scatter!(ax, times, sizes, 
            color = :black, 
            marker = :circle,
            markersize = 8)
    
    # Add reference line for global state space size
    global_size = length(result.global_state_space)
    hlines!(ax, [global_size],
           color = (:black, 0.3),
           linestyle = :dash,
           linewidth = 1.0,
           label = "Global space")
    
    axislegend(ax, position = :rt)
    
    return fig
end

"""
    plot_experiment_final_comparison(result::ExperimentResult)

Create bar chart comparing final learned rates to true rates (SIAM style).
"""
function plot_experiment_final_comparison(result::ExperimentResult)
    set_siam_theme!()
    
    n_reactions = length(result.inferred_stoich)
    propensity_fn = auto_detect_propensity_function(result.reaction_network, result.inferred_stoich)
    
    # Extract final rates
    final_gen = result.learned_generators[end]
    final_stats = extract_rates_global(final_gen, result.global_state_space, propensity_fn, result.inferred_stoich)
    
    learned_rates = [final_stats[j].n_transitions > 0 ? final_stats[j].median : 0.0 for j in 1:n_reactions]
    rate_errors = [final_stats[j].n_transitions > 0 ? final_stats[j].std : 0.0 for j in 1:n_reactions]
    
    # Create figure
    fig = Figure(size = (500, 350))
    
    ax = Axis(fig[1, 1],
             xlabel = "Reaction",
             ylabel = "Rate Constant",
             title = "Final Learned vs True Rates",
             xticks = (1:n_reactions, ["k$j" for j in 1:n_reactions]))
    
    # Position bars
    x = 1:n_reactions
    width = 0.35
    
    # True rates (left bars)
    barplot!(ax, x .- width/2, result.true_rates,
            color = :black,
            strokecolor = :white,
            strokewidth = 1,
            width = width,
            label = "True")
    
    # Learned rates (right bars with pattern)
    barplot!(ax, x .+ width/2, learned_rates,
            color = (:black, 0.5),  # Gray fill
            strokecolor = :black,
            strokewidth = 1,
            width = width,
            label = "Learned")
    
    # Error bars on learned rates
    errorbars!(ax, x .+ width/2, learned_rates, rate_errors,
              color = :black,
              linewidth = 1.5,
              whiskerwidth = 8)
    
    axislegend(ax, position = :rt)
    
    return fig
end

"""
    plot_experiment_combined(result::ExperimentResult; skip_transient=0)

Create combined multi-panel figure for paper (SIAM style).

Creates a comprehensive figure with:
- Panel (a): Rate convergence
- Panel (b): State space evolution
- Panel (c): Final rate comparison
"""
function plot_experiment_combined(result::ExperimentResult; skip_transient=0)
    set_siam_theme!()
    
    n_reactions = length(result.inferred_stoich)
    propensity_fn = auto_detect_propensity_function(result.reaction_network, result.inferred_stoich)
    
    # Create figure with 3 panels
    fig = Figure(size = (500, 900))
    
    # Panel A: Rate convergence
    ax1 = Axis(fig[1, 1],
              xlabel = "Time",
              ylabel = "Relative Error (%)",
              title = "(a) Rate Convergence",
              yscale = log10)
    
    times = [lg.t_start for lg in result.learned_generators]
    start_idx = skip_transient + 1
    
    for j in 1:n_reactions
        rel_errors = Float64[]
        t_valid = Float64[]
        
        for (i, lg) in enumerate(result.learned_generators[start_idx:end])
            rate_stats = extract_rates_global(lg, result.global_state_space, propensity_fn, result.inferred_stoich)
            if rate_stats[j].n_transitions > 0
                push!(rel_errors, abs(rate_stats[j].median - result.true_rates[j]) / result.true_rates[j] * 100)
                push!(t_valid, times[start_idx + i - 1])
            end
        end
        
        if !isempty(t_valid)
            lines!(ax1, t_valid, rel_errors,
                  color = :black,
                  linestyle = SIAM_LINESTYLES[mod1(j, length(SIAM_LINESTYLES))],
                  linewidth = 1.5,
                  label = "k$j")
            scatter!(ax1, t_valid, rel_errors,
                    color = :black,
                    marker = SIAM_MARKERS[mod1(j, length(SIAM_MARKERS))],
                    markersize = 6)
        end
    end
    
    hlines!(ax1, [1.0], color = (:black, 0.3), linestyle = :dot, linewidth = 1.0)
    axislegend(ax1, position = :rt)
    
    # Panel B: State space size
    ax2 = Axis(fig[2, 1],
              xlabel = "Time",
              ylabel = "State Space Size",
              title = "(b) Adaptive State Space")
    
    sizes = [length(lg.state_space) for lg in result.learned_generators]
    lines!(ax2, times, sizes, color = :black, linewidth = 1.5)
    scatter!(ax2, times, sizes, color = :black, marker = :circle, markersize = 8)
    
    global_size = length(result.global_state_space)
    hlines!(ax2, [global_size], color = (:black, 0.3), linestyle = :dash, linewidth = 1.0)
    
    # Panel C: Final rate comparison
    ax3 = Axis(fig[3, 1],
              xlabel = "Reaction",
              ylabel = "Rate Constant",
              title = "(c) Final Learned vs True Rates",
              xticks = (1:n_reactions, ["k$j" for j in 1:n_reactions]))
    
    final_gen = result.learned_generators[end]
    final_stats = extract_rates_global(final_gen, result.global_state_space, propensity_fn, result.inferred_stoich)
    
    learned_rates = [final_stats[j].n_transitions > 0 ? final_stats[j].median : 0.0 for j in 1:n_reactions]
    rate_errors = [final_stats[j].n_transitions > 0 ? final_stats[j].std : 0.0 for j in 1:n_reactions]
    
    x = 1:n_reactions
    width = 0.35
    
    barplot!(ax3, x .- width/2, result.true_rates,
            color = :black, strokecolor = :white, strokewidth = 1, width = width, label = "True")
    barplot!(ax3, x .+ width/2, learned_rates,
            color = (:black, 0.5), strokecolor = :black, strokewidth = 1, width = width, label = "Learned")
    errorbars!(ax3, x .+ width/2, learned_rates, rate_errors,
              color = :black, linewidth = 1.5, whiskerwidth = 8)
    
    axislegend(ax3, position = :rt)
    
    return fig
end

"""
    plot_experiment_trajectory_statistics(result::ExperimentResult)

Plot statistics about the generated trajectories (SIAM style).
"""
function plot_experiment_trajectory_statistics(result::ExperimentResult)
    set_siam_theme!()
    
    # Extract trajectory lengths
    traj_lengths = [length(traj.t) for traj in result.trajectories]
    
    fig = Figure(size = (500, 350))
    
    ax = Axis(fig[1, 1],
             xlabel = "Number of Time Points",
             ylabel = "Frequency",
             title = "Trajectory Length Distribution")
    
    hist!(ax, traj_lengths,
         bins = 30,
         color = (:black, 0.5),
         strokecolor = :black,
         strokewidth = 1)
    
    # Add mean line
    mean_length = mean(traj_lengths)
    vlines!(ax, [mean_length],
           color = :black,
           linestyle = :dash,
           linewidth = 1.5,
           label = @sprintf("Mean: %.1f", mean_length))
    
    axislegend(ax, position = :rt)
    
    return fig
end

"""
    save_experiment_figures_siam(result::ExperimentResult;
                                 output_dir="figures_siam",
                                 formats=[:pdf, :eps],
                                 skip_transient=0)

Generate and save all SIAM-compliant figures for an experiment.

# Arguments
- `result`: ExperimentResult from run_inverse_experiment
- `output_dir`: Directory to save figures (default: "figures_siam")
- `formats`: Vector of output formats (default: [:pdf, :eps])
- `skip_transient`: Number of initial windows to skip in convergence plot
"""
function save_experiment_figures_siam(
    result::ExperimentResult;
    output_dir="figures_siam",
    formats=[:pdf, :eps],
    skip_transient=0
)
    mkpath(output_dir)
    
    println("\nGenerating SIAM-compliant figures for experiment...")
    
    # 1. Convergence plot
    println("  - Rate convergence")
    fig1 = plot_experiment_convergence(result, skip_transient=skip_transient)
    for fmt in formats
        save(joinpath(output_dir, "convergence.$fmt"), fig1, pt_per_unit=1)
    end
    
    # 2. Rate evolution
    println("  - Rate evolution over time")
    fig2 = plot_experiment_rates_evolution(result)
    for fmt in formats
        save(joinpath(output_dir, "rates_evolution.$fmt"), fig2, pt_per_unit=1)
    end
    
    # 3. State space size
    println("  - State space evolution")
    fig3 = plot_experiment_state_space(result)
    for fmt in formats
        save(joinpath(output_dir, "state_space.$fmt"), fig3, pt_per_unit=1)
    end
    
    # 4. Final comparison
    println("  - Final rate comparison")
    fig4 = plot_experiment_final_comparison(result)
    for fmt in formats
        save(joinpath(output_dir, "final_comparison.$fmt"), fig4, pt_per_unit=1)
    end
    
    # 5. Combined figure
    println("  - Combined multi-panel figure")
    fig5 = plot_experiment_combined(result, skip_transient=skip_transient)
    for fmt in formats
        save(joinpath(output_dir, "combined_results.$fmt"), fig5, pt_per_unit=1)
    end
    
    # 6. Trajectory statistics
    println("  - Trajectory statistics")
    fig6 = plot_experiment_trajectory_statistics(result)
    for fmt in formats
        save(joinpath(output_dir, "trajectory_stats.$fmt"), fig6, pt_per_unit=1)
    end
    
    println("\n✓ All figures saved to $output_dir/")
    println("  Formats: $(join(formats, ", "))")
    println("\nFigures generated:")
    println("  - convergence.{pdf,eps}      : Rate convergence over windows")
    println("  - rates_evolution.{pdf,eps}  : Individual rate evolution with error bands")
    println("  - state_space.{pdf,eps}      : Adaptive state space size")
    println("  - final_comparison.{pdf,eps} : Bar chart of final vs true rates")
    println("  - combined_results.{pdf,eps} : Three-panel comprehensive figure")
    println("  - trajectory_stats.{pdf,eps} : SSA trajectory length distribution")
end

"""
    format_stoichiometry(stoich::Vector{Int})

Format stoichiometry vector as readable string.
"""
function format_stoichiometry(stoich::Vector{Int})
    return "[" * join(stoich, ", ") * "]"
end

# Export all visualization functions
export plot_experiment_convergence
export plot_experiment_rates_evolution
export plot_experiment_state_space
export plot_experiment_final_comparison
export plot_experiment_combined
export plot_experiment_trajectory_statistics
export save_experiment_figures_siam
export set_siam_theme!
