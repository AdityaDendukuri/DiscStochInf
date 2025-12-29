"""
Brusselator Adaptive Windowing Experiment
Tests adaptive windowing on oscillating/limit cycle dynamics with extreme multiscale stiffness.
"""

using Catalyst

include("experiment_runner.jl")
include("experiments.jl")
include("adaptive_windowing.jl")
include("visualization.jl")

# ============================================================================
# DEFINE BRUSSELATOR NETWORK
# ============================================================================

brusselator_rn = @reaction_network begin
    k1, ∅ --> X      # Inflow
    k2, 2X + Y --> 3X  # Autocatalysis (oscillation driver)
    k3, X --> Y      # Conversion
    k4, X --> ∅      # Outflow
end

u0_brusselator = [:X => 50, :Y => 50]
true_rates_bruss = [1.0, 0.0001, 1.0, 1.0]  # k2=0.0001 creates extreme multiscale

# ============================================================================
# BASELINE EXPERIMENTS (commented out - already run)
# ============================================================================

# # ============================================================================
# # EXPERIMENT 1: BASELINE
# # ============================================================================
# 
# println("\n" * "="^80)
# println("EXPERIMENT 1: BRUSSELATOR BASELINE")
# println("="^80)
# println("\nKey challenge: k2=0.0001 is 10,000× slower than k1,k3,k4=1.0")
# println("This tests extreme multiscale performance")
# 
# bruss_config = InverseProblemConfig(
#     mass_threshold = 0.98,
#     λ_frobenius = 1e-6,
#     λ_prob_conservation = 0.1,
#     dt_snapshot = 0.2,
#     dt_window = 1.0,
#     snapshots_per_window = 5,
#     max_windows = 20
# )
# 
# result_bruss, logger_bruss = run_comprehensive_experiment(
#     brusselator_rn,
#     u0_brusselator,
#     true_rates_bruss;
#     n_trajectories = 5000,
#     tspan = (0.0, 20.0),
#     tspan_learning = (0.0, 15.0),
#     config = bruss_config,
#     experiment_name = "brusselator_baseline",
#     compute_theoretical_bounds = true,
#     analyze_convergence = true
# )
# 
# # ============================================================================
# # EXPERIMENT 2: TRAJECTORY SWEEP
# # ============================================================================
# 
# println("\n" * "="^80)
# println("EXPERIMENT 2: BRUSSELATOR TRAJECTORY SWEEP")
# println("="^80)
# println("Testing if rare k2 reaction can be recovered with more sampling")
# 
# sweep_trajectories_bruss = run_parameter_sweep(
#     brusselator_rn,
#     u0_brusselator,
#     true_rates_bruss,
#     :n_trajectories,
#     [1000, 2000, 5000, 10000];
#     base_config = bruss_config,
#     experiment_name = "brusselator_trajectory_sweep"
# )
# 
# # ============================================================================
# # EXPERIMENT 3: WINDOW SIZE SWEEP
# # ============================================================================
# 
# println("\n" * "="^80)
# println("EXPERIMENT 3: BRUSSELATOR WINDOW SIZE SWEEP")
# println("="^80)
# println("Testing oscillation phase capture with different window sizes")
# 
# sweep_window_bruss = run_parameter_sweep(
#     brusselator_rn,
#     u0_brusselator,
#     true_rates_bruss,
#     :dt_window,
#     [0.5, 1.0, 2.0, 5.0];
#     base_config = bruss_config,
#     experiment_name = "brusselator_window_sweep"
# )
# 
# # ============================================================================
# # EXPERIMENT 4: SNAPSHOT FREQUENCY SWEEP
# # ============================================================================
# 
# println("\n" * "="^80)
# println("EXPERIMENT 4: BRUSSELATOR SNAPSHOT FREQUENCY")
# println("="^80)
# println("Testing if finer temporal resolution helps capture oscillations")
# 
# sweep_snapshot_bruss = run_parameter_sweep(
#     brusselator_rn,
#     u0_brusselator,
#     true_rates_bruss,
#     :dt_snapshot,
#     [0.1, 0.2, 0.5, 1.0];
#     base_config = bruss_config,
#     experiment_name = "brusselator_snapshot_sweep"
# )

# ============================================================================
# ADAPTIVE WINDOWING EXPERIMENTS
# ============================================================================

println("Starting adaptive windowing experiments for Brusselator...")
println("This will test if adaptive windows can handle oscillations and extreme multiscale")

bruss_config = InverseProblemConfig(
    mass_threshold = 0.98,
    λ_frobenius = 1e-6,
    λ_prob_conservation = 0.1,
    dt_snapshot = 0.2,
    dt_window = 1.0,
    snapshots_per_window = 5,
    max_windows = 20
)

# # Experiment 1: Fixed windowing (control) - COMMENTED OUT
# println("\n" * "="^80)
# println("EXPERIMENT 1: FIXED WINDOWING (CONTROL)")
# println("="^80)
# 
# result_bruss_fixed, logger_bruss_fixed = run_comprehensive_experiment(
#     brusselator_rn,
#     u0_brusselator,
#     true_rates_bruss;
#     n_trajectories = 5000,
#     tspan = (0.0, 20.0),
#     tspan_learning = (0.0, 15.0),
#     config = bruss_config,
#     experiment_name = "brusselator_fixed_control",
#     compute_theoretical_bounds = true,
#     analyze_convergence = true
# )

# Experiment 2: Adaptive windowing (experimental)
println("\n" * "="^80)
println("EXPERIMENT: ADAPTIVE WINDOWING (Brusselator)")
println("="^80)
println("Expected behavior:")
println("  - w_eff should vary with limit cycle oscillations")
println("  - Windows should be MUCH smaller than M-M (w_eff ~ 1.0 vs 0.001)")
println("  - dt should adapt to track oscillation phase")
println()

result_bruss_adaptive, logger_bruss_adaptive, (times_bruss, w_eff_bruss) = run_adaptive_experiment(
    brusselator_rn,
    u0_brusselator,
    true_rates_bruss;
    n_trajectories = 5000,
    tspan = (0.0, 20.0),
    tspan_learning = (0.0, 15.0),
    experiment_name = "brusselator_adaptive_experimental",
    adaptive_windowing = true,
    target_uniqueness = 0.20,  # More conservative for difficult system
    dt_bounds = (0.3, 5.0),     # Smaller windows expected
    compute_theoretical_bounds = true
)

# ============================================================================
# DETAILED ANALYSIS
# ============================================================================

println("\n" * "="^80)
println("BRUSSELATOR ADAPTIVE WINDOWING ANALYSIS")
println("="^80)

# Adaptive windowing results
println("\n📊 ADAPTIVE WINDOWING RESULTS:")
propensity_adaptive = auto_detect_propensity_function(brusselator_rn, result_bruss_adaptive.inferred_stoich)
agg_adaptive = extract_rates_aggregated(
    result_bruss_adaptive.learned_generators,
    propensity_adaptive,
    result_bruss_adaptive.inferred_stoich
)

errors_adaptive = Float64[]
for j in 1:4
    if agg_adaptive[j].n_transitions > 0
        err = abs(agg_adaptive[j].median - true_rates_bruss[j]) / true_rates_bruss[j]
        push!(errors_adaptive, err)
        println("  R$j: $(round(agg_adaptive[j].median, sigdigits=4)) (true: $(true_rates_bruss[j]), error: $(round(err*100, digits=1))%, n=$(agg_adaptive[j].n_transitions))")
    else
        println("  R$j: N/A (no transitions observed)")
    end
end

dt_sizes_adaptive = [w.times[end] - w.times[1] for w in result_bruss_adaptive.windows]
println("\n  Summary:")
if !isempty(errors_adaptive)
    println("    Mean error: $(round(mean(errors_adaptive)*100, digits=1))%")
    println("    Median error: $(round(median(errors_adaptive)*100, digits=1))%")
else
    println("    Mean error: N/A (no reactions recovered)")
end
println("    Windows created: $(length(result_bruss_adaptive.windows))")
println("    Window sizes: $(round(minimum(dt_sizes_adaptive), digits=2)) - $(round(maximum(dt_sizes_adaptive), digits=2))")
println("    Mean window size: $(round(mean(dt_sizes_adaptive), digits=2))")

# Transition sampling
total_trans_adaptive = sum(agg_adaptive[j].n_transitions for j in 1:4)
println("    Total transitions: $total_trans_adaptive")

# w_eff evolution analysis
println("\n📊 w_eff EVOLUTION (Limit Cycle Tracking):")
if length(times_bruss) > 0
    println("  Number of estimates: $(length(w_eff_bruss))")
    println("  Min w_eff: $(round(minimum(w_eff_bruss), sigdigits=3))")
    println("  Max w_eff: $(round(maximum(w_eff_bruss), sigdigits=3))")
    println("  Mean w_eff: $(round(mean(w_eff_bruss), sigdigits=3))")
    println("  Std w_eff: $(round(std(w_eff_bruss), sigdigits=3))")
    println("  Range: $(round(maximum(w_eff_bruss) / minimum(w_eff_bruss), sigdigits=2))× variation")
    
    println("\n  w_eff profile (first 10 time points):")
    for i in 1:min(10, length(times_bruss))
        println("    t=$(round(times_bruss[i], digits=2)): w_eff=$(round(w_eff_bruss[i], sigdigits=3))")
    end
    
    if length(times_bruss) > 10
        println("  ...")
    end
    
    # Check for oscillations
    if std(w_eff_bruss) / mean(w_eff_bruss) > 0.3
        println("\n  ✓ Significant w_eff variation detected (CV=$(round(std(w_eff_bruss)/mean(w_eff_bruss), digits=2)))")
        println("    This likely reflects limit cycle oscillations!")
    else
        println("\n  ⚠ Low w_eff variation (CV=$(round(std(w_eff_bruss)/mean(w_eff_bruss), digits=2)))")
        println("    Oscillations may be weak or damped")
    end
    
    # Compare to Michaelis-Menten expectations
    println("\n  Comparison to Michaelis-Menten:")
    println("    M-M w_eff: ~0.001 (very slow system)")
    println("    Brusselator w_eff: ~$(round(mean(w_eff_bruss), sigdigits=2)) ($(round(mean(w_eff_bruss)/0.001, digits=0))× faster)")
end

# Window size adaptation analysis
println("\n📊 WINDOW SIZE ADAPTATION:")
println("  Window size distribution:")
for (i, w) in enumerate(result_bruss_adaptive.windows)
    dt = w.times[end] - w.times[1]
    t_mid = (w.times[1] + w.times[end]) / 2
    
    # Find nearest w_eff estimate
    if !isempty(times_bruss)
        idx = argmin(abs.(times_bruss .- t_mid))
        w_est = w_eff_bruss[idx]
        uniqueness_prod = dt * w_est
        
        println("    Window $i (t=$(round(t_mid, digits=1))): dt=$(round(dt, digits=2)), w_eff=$(round(w_est, sigdigits=3)), Δt·w=$(round(uniqueness_prod, sigdigits=3))")
    end
end

# ============================================================================
# SAVE LOCATIONS
# ============================================================================

println("\n" * "="^80)
println("📁 EXPERIMENT LOGS SAVED")
println("="^80)
# println("\n  Fixed windowing:")
# println("    $(logger_bruss_fixed.log_dir)")
println("\n  Adaptive windowing:")
println("    $(logger_bruss_adaptive.log_dir)")

println("\n" * "="^80)
println("BRUSSELATOR ADAPTIVE EXPERIMENT COMPLETE!")
println("="^80)
println("\nKey findings to look for:")
println("  1. Is w_eff much larger than M-M (~1.0 vs ~0.001)?")
println("  2. Does w_eff oscillate (reflecting limit cycle)?")
println("  3. Are window sizes smaller and more variable?")
println("  4. Does adaptive windowing help with the catastrophic errors (~170%)?")
