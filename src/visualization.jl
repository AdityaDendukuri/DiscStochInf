# File: src/visualization.jl

"""
Enhanced visualization for generators and trajectories.
"""

using Plots
using Statistics
using LinearAlgebra

# Update to src/visualization.jl

"""
    plot_trajectories_with_windows(trajectories, window_times; n_display=20, species_to_plot=1:2)

Plot trajectories with mean trajectory and vertical lines at window boundaries.
For systems with >2 species, specify which species to plot.
"""
function plot_trajectories_with_windows(trajectories, window_times; 
                                       n_display=20, species_to_plot=1:2)
    # Determine number of species
    n_species = length(trajectories[1].u[1])
    
    # Sample trajectories to display
    n_traj = length(trajectories)
    display_indices = 1:min(n_display, n_traj)
    
    # Create plot
    p = plot(xlabel="Time (s)", ylabel="Population", 
             legend=:topright, legendfontsize=9,
             foreground_color_legend=nothing,
             background_color_legend=nothing,
             framestyle=:box,
             grid=true, gridalpha=0.15, gridstyle=:dot,
             size=(1800, 400))
    
    # Define line styles for different species
    line_styles = [:solid, :dash, :dot, :dashdot]
    
    # Plot individual trajectories (very light gray)
    for i in display_indices
        traj = trajectories[i]
        for (idx, sp) in enumerate(species_to_plot)
            plot!(p, traj.t, [u[sp] for u in traj.u], 
                  color=:gray, alpha=0.15, linewidth=0.3, 
                  linestyle=line_styles[idx], label="")
        end
    end
    
    # Compute mean trajectories
    t_grid = range(0, stop=trajectories[1].t[end], length=500)
    means = [zeros(length(t_grid)) for _ in species_to_plot]
    
    for t_idx in 1:length(t_grid)
        t = t_grid[t_idx]
        vals = [Float64[] for _ in species_to_plot]
        
        for traj in trajectories
            idx = searchsortedfirst(traj.t, t)
            if idx > length(traj.t)
                for (sp_idx, sp) in enumerate(species_to_plot)
                    push!(vals[sp_idx], traj.u[end][sp])
                end
            elseif idx == 1
                for (sp_idx, sp) in enumerate(species_to_plot)
                    push!(vals[sp_idx], traj.u[1][sp])
                end
            else
                t1, t2 = traj.t[idx-1], traj.t[idx]
                u1, u2 = traj.u[idx-1], traj.u[idx]
                α = (t - t1) / (t2 - t1)
                for (sp_idx, sp) in enumerate(species_to_plot)
                    push!(vals[sp_idx], (1-α)*u1[sp] + α*u2[sp])
                end
            end
        end
        
        for sp_idx in 1:length(species_to_plot)
            means[sp_idx][t_idx] = mean(vals[sp_idx])
        end
    end
    
    # Species labels (generic)
    species_labels = ["Species $sp" for sp in species_to_plot]
    
    # Override for common models
    if n_species == 4
        # Assume Michaelis-Menten: E, S, ES, P
        all_labels = ["⟨E⟩", "⟨S⟩", "⟨ES⟩", "⟨P⟩"]
        species_labels = [all_labels[sp] for sp in species_to_plot]
    elseif n_species == 2
        # Brusselator: X, Y
        all_labels = ["⟨X⟩", "⟨Y⟩"]
        species_labels = [all_labels[sp] for sp in species_to_plot]
    end
    
    # Plot mean trajectories (thicker, solid black)
    for (sp_idx, sp) in enumerate(species_to_plot)
        plot!(p, t_grid, means[sp_idx], 
              color=:black, linewidth=2.5, 
              linestyle=line_styles[sp_idx],
              label=species_labels[sp_idx])
    end
    
    # Add vertical lines at window boundaries
    for t in window_times
        vline!(p, [t], color=:black, linewidth=1.5, linestyle=:dot, 
               label="", alpha=0.6)
    end
    
    # Add window labels
    y_max = maximum(maximum(m) for m in means) * 0.95
    for i in 1:(length(window_times)-1)
        t_mid = (window_times[i] + window_times[i+1]) / 2
        annotate!(p, t_mid, y_max, text("W$i", 10, :center, :bold))
    end
    
    return p
end

# File: src/visualization.jl

"""
    plot_generator_bw(A, state_space; title="", show_colorbar=true, global_max=nothing)

Plot generator matrix in black and white SIAM style.
If global_max is provided, normalize all matrices to same scale.
"""
function plot_generator_bw(A, state_space; title="", show_colorbar=true, global_max=nothing)
    n = size(A, 1)
    
    # Normalize for visualization
    A_vis = copy(A)
    
    # Use global max if provided, otherwise local max
    if global_max !== nothing
        max_abs = global_max
    else
        max_abs = maximum(abs.(A))
    end
    
    if max_abs > 0
        A_vis = A_vis / max_abs
    end
    
    # Create heatmap (grayscale)
    p = heatmap(A_vis, 
                yflip=true,
                color=:grays,
                aspect_ratio=:equal,
                title=title,
                xlabel="State index",
                ylabel="State index",
                colorbar=show_colorbar,
                titlefontsize=10,
                guidefontsize=8,
                tickfontsize=7,
                clims=(-1, 1),
                framestyle=:box)
    
    # Add grid for better readability
    n_ticks = min(5, n)
    tick_positions = round.(Int, range(1, stop=n, length=n_ticks))
    plot!(p, xticks=tick_positions, yticks=tick_positions)
    
    return p
end

"""
    plot_generator_structure(A, state_space; title="")

Plot generator with structure emphasis: diagonal vs off-diagonal.
"""
function plot_generator_structure(A, state_space; title="")
    n = size(A, 1)
    
    # Separate diagonal and off-diagonal
    A_diag = diagm(diag(A))
    A_off = A - A_diag
    
    # Create visualization matrix: 
    # -1 for negative diagonal, +1 for positive off-diagonal, 0 for zero
    A_struct = zeros(n, n)
    
    for i in 1:n
        for j in 1:n
            if i == j && A[i,j] < -1e-10
                A_struct[i,j] = -1  # Negative diagonal (outflow)
            elseif i != j && A[i,j] > 1e-10
                A_struct[i,j] = 1   # Positive off-diagonal (inflow)
            end
        end
    end
    
    p = heatmap(A_struct,
                yflip=true,
                color=:grays,
                aspect_ratio=:equal,
                title=title,
                xlabel="State index",
                ylabel="State index",
                colorbar=false,
                titlefontsize=10,
                guidefontsize=8,
                tickfontsize=7)
    
    return p
end

"""
    create_full_visualization(trajectories, generators, state_spaces, 
                              window_pairs, dt; n_windows=6, species_to_plot=nothing)

Create full visualization with trajectories on top and generators in 2×3 grid below.
"""
function create_full_visualization(trajectories, generators, state_spaces, 
                                   window_pairs, dt; n_windows=6, 
                                   species_to_plot=nothing)
    
    # Auto-detect species to plot
    if species_to_plot === nothing
        n_species = length(trajectories[1].u[1])
        if n_species == 2
            species_to_plot = 1:2  # Plot both for 2D systems
        elseif n_species == 4
            species_to_plot = [2, 3, 4]  # For MM: S, ES, P
        else
            species_to_plot = 1:min(3, n_species)
        end
    end
    
    # Window times
    window_times = [0.0]
    for i in 1:n_windows
        push!(window_times, window_times[end] + dt)
    end
    
    # Top panel: Trajectories
    p_traj = plot_trajectories_with_windows(trajectories, window_times, 
                                            n_display=30,
                                            species_to_plot=species_to_plot)
    
    # Find global max for consistent normalization
    global_max = maximum(maximum(abs.(A)) for A in generators)
    
    # Bottom panels: 2×3 grid of generators
    generator_plots = []
    for (i, (A, state_space, w)) in enumerate(zip(generators, state_spaces, window_pairs))
        show_cbar = (i == 3)  # Colorbar on rightmost of top row
        p_gen = plot_generator_bw(A, state_space, 
                                  title="W$w→$(w+1): n=$(length(state_space)), ‖A‖=$(round(norm(A), digits=0))",
                                  show_colorbar=show_cbar,
                                  global_max=global_max)
        push!(generator_plots, p_gen)
    end
    
    # Combine
    layout = @layout [
        a{0.3h}
        grid(2, 3){0.7h}
    ]
    
    p_full = plot(p_traj, generator_plots..., 
                  layout=layout,
                  size=(1800, 1400),
                  margin=5Plots.mm,
                  dpi=300)
    
    return p_full
end

"""
    plot_trajectories_with_windows(trajectories, window_times; n_display=20, species_to_plot=1:2)

Plot trajectories with mean trajectory and vertical lines at window boundaries.
"""
function plot_trajectories_with_windows(trajectories, window_times; 
                                       n_display=20, species_to_plot=1:2)
    n_species = length(trajectories[1].u[1])
    n_traj = length(trajectories)
    display_indices = 1:min(n_display, n_traj)
    
    # Create plot
    p = plot(xlabel="Time (s)", ylabel="Population", 
             legend=:right, legendfontsize=9,
             framestyle=:box,
             grid=true, gridalpha=0.15, gridstyle=:dot,
             size=(1800, 400))
    
    # Line styles
    line_styles = [:solid, :dash, :dot, :dashdot]
    
    # Plot individual trajectories (light gray)
    for i in display_indices
        traj = trajectories[i]
        for (idx, sp) in enumerate(species_to_plot)
            plot!(p, traj.t, [u[sp] for u in traj.u], 
                  color=:gray, alpha=0.15, linewidth=0.3, 
                  linestyle=line_styles[idx], label="")
        end
    end
    
    # Compute mean trajectories
    t_grid = range(0, stop=trajectories[1].t[end], length=500)
    means = [zeros(length(t_grid)) for _ in species_to_plot]
    
    for t_idx in 1:length(t_grid)
        t = t_grid[t_idx]
        vals = [Float64[] for _ in species_to_plot]
        
        for traj in trajectories
            idx = searchsortedfirst(traj.t, t)
            if idx > length(traj.t)
                for (sp_idx, sp) in enumerate(species_to_plot)
                    push!(vals[sp_idx], traj.u[end][sp])
                end
            elseif idx == 1
                for (sp_idx, sp) in enumerate(species_to_plot)
                    push!(vals[sp_idx], traj.u[1][sp])
                end
            else
                t1, t2 = traj.t[idx-1], traj.t[idx]
                u1, u2 = traj.u[idx-1], traj.u[idx]
                α = (t - t1) / (t2 - t1)
                for (sp_idx, sp) in enumerate(species_to_plot)
                    push!(vals[sp_idx], (1-α)*u1[sp] + α*u2[sp])
                end
            end
        end
        
        for sp_idx in 1:length(species_to_plot)
            means[sp_idx][t_idx] = mean(vals[sp_idx])
        end
    end
    
    # Species labels
    if n_species == 2
        species_labels = ["⟨X⟩", "⟨Y⟩"]
    elseif n_species == 4
        all_labels = ["⟨E⟩", "⟨S⟩", "⟨ES⟩", "⟨P⟩"]
        species_labels = [all_labels[sp] for sp in species_to_plot]
    else
        species_labels = ["Species $sp" for sp in species_to_plot]
    end
    
    # Plot mean trajectories
    for (sp_idx, sp) in enumerate(species_to_plot)
        plot!(p, t_grid, means[sp_idx], 
              color=:black, linewidth=2.5, 
              linestyle=line_styles[sp_idx],
              label=species_labels[sp_idx])
    end
    
    # Add window boundaries
    for t in window_times
        vline!(p, [t], color=:black, linewidth=1.5, linestyle=:dot, 
               label="", alpha=0.6)
    end
    
    # Add window labels
    y_max = maximum(maximum(m) for m in means) * 0.95
    for i in 1:(length(window_times)-1)
        t_mid = (window_times[i] + window_times[i+1]) / 2
        annotate!(p, t_mid, y_max, text("W$i", 10, :center, :bold))
    end
    
    return p
end
