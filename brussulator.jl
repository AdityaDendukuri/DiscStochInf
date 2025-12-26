"""
Example: Brusselator Oscillating System
SIAM-compliant experiment for limit cycle discovery.
"""

using Catalyst
include("experiment_runner.jl")
include("visualization.jl")

# ============================================================================
# DEFINE BRUSSELATOR NETWORK
# ============================================================================
# The Classic Brusselator:
# A -> X         (Inflow)
# 2X + Y -> 3X   (Autocatalysis - the oscillation driver)
# B + X -> Y + D (Conversion)
# X -> E         (Outflow)
# (A and B are buffered/constant, absorbed into rate constants)

brusselator_rn = @reaction_network begin
    k1, ∅ --> X      # A -> X
    k2, 2X + Y --> 3X
    k3, X --> Y      # B + X -> Y
    k4, X --> ∅      # X -> E
end

# ============================================================================
# CONFIGURATION
# ============================================================================

# These parameters produce a stable limit cycle with reasonable copy numbers
# X ~ 30-80, Y ~ 30-80
u0_brusselator = [:X => 50, :Y => 50]

# True rates ensuring oscillations
# k1=1.0 (Inflow)
# k2=0.0001 (Autocatalysis - needs to be small for cubic term 2X+Y)
# k3=1.0 (Conversion)
# k4=1.0 (Outflow)
true_rates_bruss = [1.0, 0.0001, 1.0, 1.0]

# Configuration tailored for oscillations
# We need strictly small dt to resolve the phase of the cycle
bruss_config = InverseProblemConfig(
    mass_threshold = 0.98,          # Higher threshold to capture the "tail" of the wave
    λ_frobenius = 1e-6,
    λ_prob_conservation = 1e-6,
    dt_snapshot = 0.2,              # Frequent snapshots to capture phase
    dt_window = 1.0,                # Window covers ~10-20% of a cycle
    snapshots_per_window = 5,
    max_windows = 20                # Observe ~2 full cycles
)

# ============================================================================
# RUN EXPERIMENT
# ============================================================================

println("\n" * "="^70)
println("BRUSSELATOR OSCILLATING EXPERIMENT")
println("="^70)
println("Why this works:")
println("  - 2 Species (X,Y) -> State space ~50x50 = 2500 states (Fast!)")
println("  - 3rd Order Reaction (2X+Y->3X) tests your method's nonlinearity handling")
println("  - Limit Cycle: Probability mass rotates in a ring")
println("="^70)

result = run_inverse_experiment(
    brusselator_rn,
    u0_brusselator,
    true_rates_bruss;
    n_trajectories = 2000,          # Need reasonable density for 2D space
    tspan = (0.0, 20.0),            # ~2-3 full periods
    tspan_learning = (0.0, 15.0),
    config = bruss_config,
    seed = 1234,
    verbose = true
)

# ============================================================================
# ANALYZE & VISUALIZE
# ============================================================================

print_results(result, skip_transient=5)

println("\n" * "="^70)
println("GENERATING FIGURES")
println("="^70)

save_experiment_figures_siam(
    result,
    output_dir = "brusselator_figures",
    formats = [:pdf],
    skip_transient = 5
)
