"""
Example: Brusselator Oscillating System
Fully automatic - no hardcoded propensities or stoichiometry!
"""

using Catalyst

include("experiment_runner.jl")

# ============================================================================
# DEFINE BRUSSELATOR NETWORK
# ============================================================================

brusselator_rn = @reaction_network begin
    k1, ∅ --> X        # Inflow
    k2, 2X + Y --> 3X  # Autocatalysis (the oscillation driver)
    k3, X --> Y        # Conversion
    k4, X --> ∅        # Outflow
end

# ============================================================================
# CONFIGURATION
# ============================================================================

u0_brusselator = [:X => 50, :Y => 50]

# TRUE RATES in Catalyst network order
true_rates_bruss = [1.0, 0.0001, 1.0, 1.0]

# Configuration for oscillations
bruss_config = InverseProblemConfig(
    mass_threshold = 0.98,
    λ_frobenius = 1e-6,
    λ_prob_conservation = 1e-6,
    dt_snapshot = 0.2,
    dt_window = 1.0,
    snapshots_per_window = 5,
    max_windows = 20
)

# ============================================================================
# RUN EXPERIMENT (Fully Automatic!)
# ============================================================================

println("\n" * "="^70)
println("BRUSSELATOR OSCILLATING EXPERIMENT")
println("="^70)
println("Testing automatic stoichiometry matching:")
println("  - Stoichiometry inferred from trajectories")
println("  - Automatically matched to Catalyst network order")
println("  - Propensity function auto-detected with binomial coefficients")
println("="^70)

result = run_inverse_experiment(
    brusselator_rn,
    u0_brusselator,
    true_rates_bruss;
    n_trajectories = 2000,
    tspan = (0.0, 20.0),
    tspan_learning = (0.0, 15.0),
    config = bruss_config,
    seed = 1234,
    verbose = true
)

# ============================================================================
# ANALYZE RESULTS
# ============================================================================

print_results(result)

println("\n" * "="^70)
println("STOICHIOMETRY MATCHING")
println("="^70)
println("Permutation: $(result.stoich_permutation)")
println("\nCatalyst network reactions:")
for (i, rxn) in enumerate(reactions(result.reaction_network))
    println("  R$i: $rxn → ν = $(result.inferred_stoich[i])")
end

println("\n" * "="^70)
println("Done! Results are now in Catalyst network order.")
println("="^70)
