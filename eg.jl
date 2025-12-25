"""
Complete example: Running inverse experiment and generating SIAM figures.

This script demonstrates the complete workflow from defining a reaction network
to generating publication-ready SIAM-compliant figures.
"""

using Catalyst
using Printf

# Load experiment framework
include("experiment_runner.jl")
include("visualization.jl")

"""
Example 1: Simple enzyme kinetics (Michaelis-Menten)
"""
function example_enzyme_kinetics()
    println("\n" * "="^70)
    println("EXAMPLE 1: ENZYME KINETICS")
    println("="^70)
    
    # Define reaction network
    rn = @reaction_network begin
        k1, S + E --> SE
        k2, SE --> S + E
        k3, SE --> P + E
    end
    
    # True parameters
    true_rates = [0.01, 0.1, 0.1]
    
    # Initial condition
    u0 = [:S => 50, :E => 10, :SE => 1, :P => 1]
    
    # Run experiment
    result = run_inverse_experiment(
        rn,
        u0,
        true_rates,
        n_trajectories = 5000,
        tspan = (0.0, 200.0),
        tspan_learning = (0.0, 150.0),
        verbose = true
    )
    
    # Print summary
    print_results(result)
    
    # Generate all SIAM figures
    save_experiment_figures_siam(
        result,
        output_dir = "example1_enzyme",
        formats = [:pdf, :eps],
        skip_transient = 2  # Skip first 2 windows
    )
    
    println("\n✓ Example 1 complete! Figures saved to example1_enzyme/")
    
    return result
end

"""
Example 2: Gene expression (birth-death process)
"""
function example_gene_expression()
    println("\n" * "="^70)
    println("EXAMPLE 2: GENE EXPRESSION")
    println("="^70)
    
    # Simple gene expression: production and degradation
    rn = @reaction_network begin
        k_on, ∅ --> mRNA
        k_off, mRNA --> ∅
    end
    
    true_rates = [5.0, 0.1]
    u0 = [:mRNA => 20]
    
    result = run_inverse_experiment(
        rn,
        u0,
        true_rates,
        n_trajectories = 10000,
        tspan = (0.0, 100.0),
        tspan_learning = (0.0, 80.0),
        verbose = true
    )
    
    print_results(result)
    
    # Generate only convergence and combined figures
    println("\nGenerating selected figures...")
    
    fig_conv = plot_experiment_convergence(result, skip_transient=1)
    save("example2_gene/convergence.pdf", fig_conv, pt_per_unit=1)
    
    fig_comb = plot_experiment_combined(result, skip_transient=1)
    save("example2_gene/combined.pdf", fig_comb, pt_per_unit=1)
    
    println("\n✓ Example 2 complete! Figures saved to example2_gene/")
    
    return result
end

"""
Example 3: Custom configuration and detailed analysis
"""
function example_custom_config()
    println("\n" * "="^70)
    println("EXAMPLE 3: CUSTOM CONFIGURATION")
    println("="^70)
    
    # Reaction network
    rn = @reaction_network begin
        k1, A --> B
        k2, B --> C
        k3, C --> A
    end
    
    true_rates = [0.5, 0.3, 0.2]
    u0 = [:A => 30, :B => 20, :C => 10]
    
    # Custom configuration with finer resolution
    custom_config = InverseProblemConfig(
        mass_threshold = 0.98,        # Higher threshold
        λ_frobenius = 1e-7,           # Less regularization
        λ_prob_conservation = 1e-6,
        dt_snapshot = 0.05,           # Finer time resolution
        dt_window = 1.5,              # Smaller windows
        snapshots_per_window = 15,    # More snapshots per window
        max_windows = 20              # More windows
    )
    
    result = run_inverse_experiment(
        rn,
        u0,
        true_rates,
        n_trajectories = 8000,
        tspan = (0.0, 100.0),
        tspan_learning = (0.0, 80.0),
        config = custom_config,
        verbose = true
    )
    
    print_results(result, skip_transient=3)
    
    # Generate all figures
    save_experiment_figures_siam(
        result,
        output_dir = "example3_custom",
        formats = [:pdf],
        skip_transient = 3
    )
    
    println("\n✓ Example 3 complete! Figures saved to example3_custom/")
    
    return result
end

"""
Example 4: Creating publication figure with custom styling
"""
function example_publication_figure(result::ExperimentResult)
    println("\n" * "="^70)
    println("EXAMPLE 4: CUSTOM PUBLICATION FIGURE")
    println("="^70)
    
    set_siam_theme!()
    
    # Create custom two-panel figure
    fig = Figure(size = (500, 600))
    
    # Extract data
    n_reactions = length(result.inferred_stoich)
    propensity_fn = auto_detect_propensity_function(result.reaction_network, result.inferred_stoich)
    times = [lg.t_start for lg in result.learned_generators]
    
    # Panel A: Convergence (skip first 2 windows)
    ax1 = Axis(fig[1, 1],
              xlabel = "Time",
              ylabel = "Relative Error (%)",
              title = "(a) Rate Convergence",
              yscale = log10)
    
    for j in 1:n_reactions
        rel_errors = Float64[]
        t_valid = Float64[]
        
        for (i, lg) in enumerate(result.learned_generators[3:end])  # Skip first 2
            rate_stats = extract_rates_global(lg, result.global_state_space, propensity_fn, result.inferred_stoich)
            if rate_stats[j].n_transitions > 0
                err = abs(rate_stats[j].median - result.true_rates[j]) / result.true_rates[j] * 100
                push!(rel_errors, err)
                push!(t_valid, times[i + 2])
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
    axislegend(ax1, position = :rt, nbanks = 2)  # Two-column legend
    
    # Panel B: Final comparison with annotations
    ax2 = Axis(fig[2, 1],
              xlabel = "Reaction",
              ylabel = "Rate Constant",
              title = "(b) Final Learned Rates",
              xticks = (1:n_reactions, ["k$j" for j in 1:n_reactions]))
    
    final_gen = result.learned_generators[end]
    final_stats = extract_rates_global(final_gen, result.global_state_space, propensity_fn, result.inferred_stoich)
    
    learned_rates = [stats.n_transitions > 0 ? stats.median : 0.0 for stats in final_stats]
    rate_errors = [stats.n_transitions > 0 ? stats.std : 0.0 for stats in final_stats]
    
    x = 1:n_reactions
    width = 0.35
    
    barplot!(ax2, x .- width/2, result.true_rates,
            color = :black, strokecolor = :white, strokewidth = 1, width = width)
    barplot!(ax2, x .+ width/2, learned_rates,
            color = (:black, 0.5), strokecolor = :black, strokewidth = 1, width = width)
    errorbars!(ax2, x .+ width/2, learned_rates, rate_errors,
              color = :black, linewidth = 1.5, whiskerwidth = 8)
    
    # Add percentage error annotations
    for j in 1:n_reactions
        if learned_rates[j] > 0
            rel_err = abs(learned_rates[j] - result.true_rates[j]) / result.true_rates[j] * 100
            y_pos = max(result.true_rates[j], learned_rates[j] + rate_errors[j]) * 1.1
            text!(ax2, j, y_pos,
                 text = @sprintf("%.1f%%", rel_err),
                 align = (:center, :bottom),
                 fontsize = 8)
        end
    end
    
    save("custom_publication_figure.pdf", fig, pt_per_unit=1)
    
    println("\n✓ Custom publication figure saved as custom_publication_figure.pdf")
    println("  Features:")
    println("    - Two-column legend in panel (a)")
    println("    - Percentage error annotations in panel (b)")
    println("    - Optimized for SIAM submission")
end

"""
Example 5: Comparing multiple experiments
"""
function example_compare_experiments()
    println("\n" * "="^70)
    println("EXAMPLE 5: COMPARING MULTIPLE EXPERIMENTS")
    println("="^70)
    
    # Define reaction network
    rn = @reaction_network begin
        k1, A --> B
        k2, B --> A
    end
    
    true_rates = [0.5, 0.3]
    u0 = [:A => 40, :B => 20]
    
    # Run with different numbers of trajectories
    println("\nRunning with 1000 trajectories...")
    result_1k = run_inverse_experiment(
        rn, u0, true_rates,
        n_trajectories = 1000,
        tspan = (0.0, 100.0),
        tspan_learning = (0.0, 80.0),
        verbose = false
    )
    
    println("Running with 5000 trajectories...")
    result_5k = run_inverse_experiment(
        rn, u0, true_rates,
        n_trajectories = 5000,
        tspan = (0.0, 100.0),
        tspan_learning = (0.0, 80.0),
        verbose = false
    )
    
    # Create comparison figure
    set_siam_theme!()
    fig = Figure(size = (500, 350))
    
    ax = Axis(fig[1, 1],
             xlabel = "Time",
             ylabel = "Relative Error (%)",
             title = "Convergence: 1K vs 5K Trajectories",
             yscale = log10)
    
    propensity_fn = auto_detect_propensity_function(rn, result_1k.inferred_stoich)
    
    # Plot for 1K trajectories
    times_1k = [lg.t_start for lg in result_1k.learned_generators]
    errors_1k = Float64[]
    
    for lg in result_1k.learned_generators
        stats = extract_rates_global(lg, result_1k.global_state_space, propensity_fn, result_1k.inferred_stoich)
        if stats[1].n_transitions > 0
            err = abs(stats[1].median - true_rates[1]) / true_rates[1] * 100
            push!(errors_1k, err)
        end
    end
    
    lines!(ax, times_1k[1:length(errors_1k)], errors_1k,
          color = :black, linestyle = :solid, linewidth = 1.5, label = "1K traj")
    scatter!(ax, times_1k[1:length(errors_1k)], errors_1k,
            color = :black, marker = :circle, markersize = 6)
    
    # Plot for 5K trajectories
    times_5k = [lg.t_start for lg in result_5k.learned_generators]
    errors_5k = Float64[]
    
    for lg in result_5k.learned_generators
        stats = extract_rates_global(lg, result_5k.global_state_space, propensity_fn, result_5k.inferred_stoich)
        if stats[1].n_transitions > 0
            err = abs(stats[1].median - true_rates[1]) / true_rates[1] * 100
            push!(errors_5k, err)
        end
    end
    
    lines!(ax, times_5k[1:length(errors_5k)], errors_5k,
          color = :black, linestyle = :dash, linewidth = 1.5, label = "5K traj")
    scatter!(ax, times_5k[1:length(errors_5k)], errors_5k,
            color = :black, marker = :utriangle, markersize = 6)
    
    hlines!(ax, [1.0], color = (:black, 0.3), linestyle = :dot, linewidth = 1.0)
    axislegend(ax, position = :rt)
    
    save("comparison_trajectories.pdf", fig, pt_per_unit=1)
    
    println("\n✓ Comparison figure saved as comparison_trajectories.pdf")
end

# Main execution
if abspath(PROGRAM_FILE) == @__FILE__
    println("Running all examples...")
    
    # Example 1: Enzyme kinetics
    result1 = example_enzyme_kinetics()
    
    # Example 2: Gene expression
    result2 = example_gene_expression()
    
    # Example 3: Custom config
    result3 = example_custom_config()
    
    # Example 4: Custom publication figure (using result from example 1)
    example_publication_figure(result1)
    
    # Example 5: Compare experiments
    example_compare_experiments()
    
    println("\n" * "="^70)
    println("ALL EXAMPLES COMPLETED!")
    println("="^70)
    println("\nFigures generated:")
    println("  example1_enzyme/     - Complete figure set for enzyme kinetics")
    println("  example2_gene/       - Selected figures for gene expression")
    println("  example3_custom/     - Figures with custom configuration")
    println("  custom_publication_figure.pdf - Custom styled figure")
    println("  comparison_trajectories.pdf   - Multi-experiment comparison")
end
