# File: examples/visualize_generators.jl
"""
Visualization of learned generators for Brusselator.
Creates publication-quality figures showing generator evolution.
"""

using Plots
using LinearAlgebra

include("../src/visualization.jl")

# Assuming you've already run brusselator.jl and have:
# - generators: Vector of learned generators
# - state_spaces: Vector of state spaces
# - window_pairs: Vector of window indices
# - trajectories: Vector of SSA trajectories
# - dt: time step

println("="^70)
println("VISUALIZING LEARNED GENERATORS")
println("="^70)

# =============================================================================
# 1. Full Visualization (Trajectories + 6 Generators)
# =============================================================================

println("\nCreating full visualization...")

# Window times for vertical lines
window_times = [0.0]
for i in 1:length(window_pairs)
    push!(window_times, window_times[end] + dt)
end

# Create the full visualization
p_full = create_full_visualization(trajectories, generators, state_spaces,
                                   window_pairs, dt, 
                                   n_windows=length(window_pairs))

display(p_full)

# Save
savefig(p_full, "brusselator_full_visualization.png")
savefig(p_full, "brusselator_full_visualization.pdf")

println("✓ Saved: brusselator_full_visualization.{png,pdf}")

# =============================================================================
# 2. Generator Evolution Plots
# =============================================================================

println("\nCreating generator evolution plots...")

# Extract statistics
norms = [norm(A) for A in generators]
sizes = [length(ss) for ss in state_spaces]
nonzeros = [count(abs.(A) .> 1e-8) for A in generators]
sparsities = [100 * (1 - nz/length(A)) for (nz, A) in zip(nonzeros, generators)]
max_col_sums = [maximum(abs.(sum(A, dims=1)[:])) for A in generators]

# Create subplots
p1 = plot(window_pairs, norms, 
          marker=:circle, markersize=6, linewidth=2,
          xlabel="Window", ylabel="||A||_F",
          title="Generator Norm Evolution",
          legend=false, grid=true, gridalpha=0.3)

p2 = plot(window_pairs, sizes,
          marker=:square, markersize=6, linewidth=2,
          xlabel="Window", ylabel="# States",
          title="State Space Size",
          legend=false, grid=true, gridalpha=0.3)

p3 = plot(window_pairs, sparsities,
          marker=:diamond, markersize=6, linewidth=2,
          xlabel="Window", ylabel="Sparsity (%)",
          title="Generator Sparsity",
          legend=false, grid=true, gridalpha=0.3,
          ylims=(0, 100))

p4 = plot(window_pairs, max_col_sums,
          marker=:circle, markersize=6, linewidth=2,
          xlabel="Window", ylabel="max |col sum|",
          title="Column Sum Violation",
          legend=false, grid=true, gridalpha=0.3,
          yscale=:log10)

p_evolution = plot(p1, p2, p3, p4, 
                   layout=(2,2), 
                   size=(1200, 800),
                   margin=5Plots.mm)

display(p_evolution)
savefig(p_evolution, "brusselator_generator_evolution.png")
savefig(p_evolution, "brusselator_generator_evolution.pdf")

println("✓ Saved: brusselator_generator_evolution.{png,pdf}")

# =============================================================================
# 3. Individual Generator Heatmaps (Larger)
# =============================================================================

println("\nCreating individual generator heatmaps...")

for (w, A, ss) in zip(window_pairs, generators, state_spaces)
    p = plot_generator_bw(A, ss,
                          title="Window $w→$(w+1): n=$(length(ss)), ||A||=$(round(norm(A), digits=1))",
                          show_colorbar=true)
    
    display(p)
    savefig(p, "brusselator_generator_W$(w).png")
end

println("✓ Saved: brusselator_generator_W*.png")

# =============================================================================
# 4. Sparsity Pattern Comparison
# =============================================================================

println("\nCreating sparsity pattern comparison...")

# Find global max for consistent color scale
global_max = maximum(maximum(abs.(A)) for A in generators)

sparsity_plots = []
for (w, A, ss) in zip(window_pairs, generators, state_spaces)
    # Binary sparsity pattern
    A_binary = abs.(A) .> 1e-8
    
    p = heatmap(A_binary,
                yflip=true,
                color=:grays,
                aspect_ratio=:equal,
                title="W$w: $(count(A_binary)) nz",
                colorbar=false,
                titlefontsize=10,
                tickfontsize=7,
                framestyle=:box)
    
    push!(sparsity_plots, p)
end

p_sparsity = plot(sparsity_plots...,
                  layout=(2, 3),
                  size=(1200, 800),
                  plot_title="Sparsity Patterns (Black = Nonzero)",
                  margin=3Plots.mm)

display(p_sparsity)
savefig(p_sparsity, "brusselator_sparsity_patterns.png")

println("✓ Saved: brusselator_sparsity_patterns.png")

# =============================================================================
# 5. Propensity Evolution Across Windows
# =============================================================================

println("\nAnalyzing propensity evolution...")

# Pick a few representative states
test_states = [[1, 0], [2, 1], [3, 2], [4, 3]]

# For each reaction, track how propensity changes
n_reactions = length(stoich_basis)

# Create plots for each test state
propensity_plots = []

for test_state in test_states
    x, y = test_state[1], test_state[2]
    
    # Get propensities at this state for all windows
    props_by_window = []
    
    for (w, (θ, data)) in enumerate(zip(parameters, learned_data))
        if test_state in data.state_space
            props = evaluate_propensities(θ, data, [test_state])
            push!(props_by_window, props[test_state])
        else
            push!(props_by_window, fill(NaN, n_reactions))
        end
    end
    
    # Plot propensity evolution for each reaction
    p = plot(xlabel="Window", ylabel="Propensity",
             title="State [X=$x, Y=$y]",
             legend=:topright, grid=true, gridalpha=0.3)
    
    for k in 1:n_reactions
        vals = [props[k] for props in props_by_window]
        plot!(p, window_pairs, vals,
              marker=:circle, linewidth=2,
              label="Reaction $k")
    end
    
    push!(propensity_plots, p)
end

p_propensities = plot(propensity_plots...,
                      layout=(2, 2),
                      size=(1200, 800),
                      plot_title="Propensity Evolution Across Windows",
                      margin=5Plots.mm)

display(p_propensities)
savefig(p_propensities, "brusselator_propensity_evolution.png")

println("✓ Saved: brusselator_propensity_evolution.png")

# =============================================================================
# 6. Summary Table
# =============================================================================

println("\n" * "="^70)
println("GENERATOR SUMMARY TABLE")
println("="^70)

println("\n" * rpad("Window", 10) * rpad("States", 10) * rpad("||A||", 12) * 
        rpad("Nonzeros", 12) * rpad("Sparsity", 12) * "max |col sum|")
println("-"^70)

for (w, A, ss) in zip(window_pairs, generators, state_spaces)
    n_states = length(ss)
    norm_A = norm(A)
    nz = count(abs.(A) .> 1e-8)
    sparsity = 100 * (1 - nz/length(A))
    col_sum_err = maximum(abs.(sum(A, dims=1)[:]))
    
    println(rpad("$w→$(w+1)", 10) * 
            rpad(n_states, 10) * 
            rpad(round(norm_A, digits=1), 12) * 
            rpad(nz, 12) * 
            rpad("$(round(sparsity, digits=1))%", 12) * 
            "$(round(col_sum_err, sigdigits=3))")
end

println("\n" * "="^70)
println("All visualizations saved!")
println("="^70)
