"""
Michaelis-Menten: Testing flux-based improvements
"""

using Catalyst
using Statistics

include("experiment_runner.jl")
include("experiments.jl")

# ============================================================================
# SETUP
# ============================================================================

rn_mm = @reaction_network begin
    k1, S + E --> SE
    k2, SE --> S + E
    k3, SE --> P + E
end

u0_mm = [:S => 50, :E => 10, :SE => 1, :P => 1]
true_rates_mm = [0.01, 0.1, 0.1]

println("Michaelis-Menten Experiments: Testing Improvements")

# ============================================================================
# EXPERIMENT 1: BASELINE (Your current best)
# ============================================================================

println("\n" * "="^80)
println("EXPERIMENT 1: BASELINE (5k trajectories, dt=10.0)")
println("="^80)

result_baseline, logger_baseline = run_comprehensive_experiment(
    rn_mm, u0_mm, true_rates_mm;
    n_trajectories = 5000,
    experiment_name = "mm_baseline_5k"
)

agg_baseline = logger_baseline.data["statistics"]["aggregated_rates"]
errors_baseline = [agg_baseline["reaction_$j"]["relative_error"] 
                   for j in 1:3 
                   if !isnan(agg_baseline["reaction_$j"]["relative_error"])]

println("\n📊 Baseline Results:")
println("  Mean error: $(round(mean(errors_baseline)*100, digits=1))%")
for j in 1:3
    println("  R$j: $(round(agg_baseline["reaction_$j"]["median"], sigdigits=4)) " *
            "(true: $(true_rates_mm[j]), error: $(round(agg_baseline["reaction_$j"]["relative_error"]*100, digits=1))%, " *
            "n=$(agg_baseline["reaction_$j"]["n_transitions"]))")
end

# ============================================================================
# EXPERIMENT 2: MORE DATA (10k trajectories)
# ============================================================================

println("\n" * "="^80)
println("EXPERIMENT 2: MORE DATA (10k trajectories)")
println("="^80)

result_10k, logger_10k = run_comprehensive_experiment(
    rn_mm, u0_mm, true_rates_mm;
    n_trajectories = 10000,
    experiment_name = "mm_10k_trajectories"
)

agg_10k = logger_10k.data["statistics"]["aggregated_rates"]
errors_10k = [agg_10k["reaction_$j"]["relative_error"] 
              for j in 1:3 
              if !isnan(agg_10k["reaction_$j"]["relative_error"])]

println("\n📊 10k Trajectory Results:")
println("  Mean error: $(round(mean(errors_10k)*100, digits=1))%")
for j in 1:3
    println("  R$j: $(round(agg_10k["reaction_$j"]["median"], sigdigits=4)) " *
            "(true: $(true_rates_mm[j]), error: $(round(agg_10k["reaction_$j"]["relative_error"]*100, digits=1))%, " *
            "n=$(agg_10k["reaction_$j"]["n_transitions"]))")
end

# ============================================================================
# EXPERIMENT 3: LONGER TIME WINDOW (to t=300)
# ============================================================================

println("\n" * "="^80)
println("EXPERIMENT 3: LONGER TIME WINDOW (to t=300)")
println("="^80)

result_long, logger_long = run_comprehensive_experiment(
    rn_mm, u0_mm, true_rates_mm;
    n_trajectories = 5000,
    tspan = (0.0, 400.0),
    tspan_learning = (0.0, 300.0),
    experiment_name = "mm_long_time"
)

agg_long = logger_long.data["statistics"]["aggregated_rates"]
errors_long = [agg_long["reaction_$j"]["relative_error"] 
               for j in 1:3 
               if !isnan(agg_long["reaction_$j"]["relative_error"])]

println("\n📊 Long Time Results:")
println("  Mean error: $(round(mean(errors_long)*100, digits=1))%")
for j in 1:3
    println("  R$j: $(round(agg_long["reaction_$j"]["median"], sigdigits=4)) " *
            "(true: $(true_rates_mm[j]), error: $(round(agg_long["reaction_$j"]["relative_error"]*100, digits=1))%, " *
            "n=$(agg_long["reaction_$j"]["n_transitions"]))")
end

# ============================================================================
# EXPERIMENT 4: SMALLER dt_snapshot (better temporal resolution)
# ============================================================================

println("\n" * "="^80)
println("EXPERIMENT 4: SMALLER dt_snapshot (0.05 instead of 0.1)")
println("="^80)

config_fine = InverseProblemConfig(
    mass_threshold = 0.999,
    λ_frobenius = 0.1,
    λ_prob_conservation = 10.0,
    dt_snapshot = 0.05,  # Finer temporal resolution
    dt_window = 2.0,      # Smaller windows too
    snapshots_per_window = 20,
    max_windows = 30
)

result_fine, logger_fine = run_comprehensive_experiment(
    rn_mm, u0_mm, true_rates_mm;
    n_trajectories = 5000,
    config = config_fine,
    experiment_name = "mm_fine_temporal"
)

agg_fine = logger_fine.data["statistics"]["aggregated_rates"]
errors_fine = [agg_fine["reaction_$j"]["relative_error"] 
               for j in 1:3 
               if !isnan(agg_fine["reaction_$j"]["relative_error"])]

println("\n📊 Fine Temporal Results:")
println("  Mean error: $(round(mean(errors_fine)*100, digits=1))%")
for j in 1:3
    println("  R$j: $(round(agg_fine["reaction_$j"]["median"], sigdigits=4)) " *
            "(true: $(true_rates_mm[j]), error: $(round(agg_fine["reaction_$j"]["relative_error"]*100, digits=1))%, " *
            "n=$(agg_fine["reaction_$j"]["n_transitions"]))")
end

# ============================================================================
# FINAL COMPARISON
# ============================================================================

println("\n" * "="^80)
println("FINAL COMPARISON")
println("="^80)

println("\nExperiment          | Mean Error | R1 Error | R2 Error | R3 Error | R1 Trans | R2 Trans | R3 Trans")
println("-"^110)

experiments = [
    ("Baseline (5k)", errors_baseline, agg_baseline),
    ("10k trajectories", errors_10k, agg_10k),
    ("Long time", errors_long, agg_long),
    ("Fine temporal", errors_fine, agg_fine)
]

for (name, errors, agg) in experiments
    mean_err = mean(errors) * 100
    r1_err = agg["reaction_1"]["relative_error"] * 100
    r2_err = agg["reaction_2"]["relative_error"] * 100
    r3_err = agg["reaction_3"]["relative_error"] * 100
    r1_n = agg["reaction_1"]["n_transitions"]
    r2_n = agg["reaction_2"]["n_transitions"]
    r3_n = agg["reaction_3"]["n_transitions"]
    
    println("$(rpad(name, 19)) | $(lpad(round(mean_err, digits=1), 10))% | " *
            "$(lpad(round(r1_err, digits=1), 8))% | " *
            "$(lpad(round(r2_err, digits=1), 8))% | " *
            "$(lpad(round(r3_err, digits=1), 8))% | " *
            "$(lpad(r1_n, 8)) | $(lpad(r2_n, 8)) | $(lpad(r3_n, 8))")
end

println("\n" * "="^80)
println("KEY FINDINGS")
println("="^80)

# Find best for each reaction
best_r1 = findmin([agg["reaction_1"]["relative_error"] for (_, _, agg) in experiments])[2]
best_r2 = findmin([agg["reaction_2"]["relative_error"] for (_, _, agg) in experiments])[2]
best_r3 = findmin([agg["reaction_3"]["relative_error"] for (_, _, agg) in experiments])[2]

println("\nBest configuration for each reaction:")
println("  R1 (S+E→SE):  $(experiments[best_r1][1])")
println("  R2 (SE→S+E):  $(experiments[best_r2][1])")
println("  R3 (SE→P+E):  $(experiments[best_r3][1])")

println("\nRecommendation:")
best_overall = findmin([mean(e) for (_, e, _) in experiments])[2]
println("  Use: $(experiments[best_overall][1])")
println("  Mean error: $(round(mean(experiments[best_overall][2])*100, digits=1))%")

println("\n📁 All logs saved in experiments/ directory")
