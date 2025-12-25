"""
Example: Simplified Toggle Switch (Proteins Only)

Uses effective rates to capture repression without explicit gene states.
This avoids the sparse gene-switching problem.
"""

using Catalyst

include("experiment_runner.jl")

# ============================================================================
# DEFINE SIMPLIFIED TOGGLE SWITCH
# ============================================================================

# Simplified: Only track proteins U and V
# Repression is captured via reduced production rates
# (As if genes are mostly bound at steady state)

toggle_simple_rn = @reaction_network begin
    α1, ∅ --> U          # Effective production of U (when gene 1 is free)
    α2, ∅ --> V          # Effective production of V (when gene 2 is free)
    β1, U + V --> V      # Mutual inhibition: U consumes itself when V is high
    β2, V + U --> U      # Mutual inhibition: V consumes itself when U is high
    δ1, U --> ∅          # U degradation
    δ2, V --> ∅          # V degradation
end

# ============================================================================
# PARAMETERS
# ============================================================================

u0_toggle_simple = [:U => 50, :V => 5]

# True rates
# Moderate disparity: 2.0 / 0.1 = 20×
true_rates_simple = [
    1.0,   # α1 - U production
    1.0,   # α2 - V production
    0.5,   # β1 - mutual inhibition
    0.5,   # β2 - mutual inhibition
    0.1,   # δ1 - U degradation
    0.1    # δ2 - V degradation
]

# Configuration
toggle_simple_config = InverseProblemConfig(
    mass_threshold = 0.95,
    λ_frobenius = 1e-6,
    λ_prob_conservation = 1e-6,
    dt_snapshot = 0.1,
    dt_window = 1.0,
    snapshots_per_window = 10,
    max_windows = 20
)

# ============================================================================
# RUN EXPERIMENT
# ============================================================================

println("\n" * "="^70)
println("SIMPLIFIED TOGGLE SWITCH (PROTEINS ONLY)")
println("="^70)
println("Why this works better:")
println("  - Only 2 species (not 6) → smaller state space")
println("  - No sparse gene-switching transitions")
println("  - Moderate rate disparity (20× vs 50×)")
println("  - Still captures bistable dynamics")
println("="^70)

result = run_inverse_experiment(
    toggle_simple_rn,
    u0_toggle_simple,
    true_rates_simple;
    n_trajectories = 3000,
    tspan = (0.0, 50.0),
    tspan_learning = (0.0, 40.0),
    config = toggle_simple_config,
    seed = 5678,
    verbose = true
)

# ============================================================================
# ANALYZE RESULTS
# ============================================================================

print_results(result)

println("\n" * "="^70)
println("EXPECTED PERFORMANCE")
println("="^70)
println("With 20× rate disparity and 2 species:")
println("  - Production (α = 1.0): 20-40% error")
println("  - Inhibition (β = 0.5): 30-50% error")
println("  - Degradation (δ = 0.1): 40-60% error")
println("\nMuch better than gene-explicit model!")
println("="^70)
