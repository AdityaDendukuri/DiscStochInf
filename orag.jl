"""
Example: Oregonator oscillating reaction network
Learning constant rates from oscillatory probability distributions.
SCALED VERSION with manageable state space.
"""

using Catalyst
include("experiment_runner.jl")
include("visualization.jl")

# ============================================================================
# DEFINE OREGONATOR NETWORK
# ============================================================================

oregonator_rn = @reaction_network begin
    k1, Y --> X          
    k2, X + Y --> ∅      
    k3, X --> 2X + Z     
    k4, 2X --> ∅          
    k5, Z --> Y          
end

# ============================================================================
# SCALED OREGONATOR (MANAGEABLE STATE SPACE)
# ============================================================================

# Original parameters are for high copy numbers
# Scale down by factor of ~50 to keep state space tractable

# Scaled initial condition (divide by ~50)
u0_scaled = [:X => 10, :Y => 20, :Z => 40]

# Scaled rates - need to adjust for different copy number regime
# For mass action: rate ~ k * (copy_number)^(order-1)
# So we need to rescale k2, k3, k4 which have bimolecular or higher order

scale_factor = 50.0

true_rates_scaled = [
    2.0,                    # k1: Y → X (first order, no change)
    0.1 / scale_factor,     # k2: X+Y → ∅ (second order, divide by scale)
    104.0,                  # k3: X → 2X+Z (effectively first order in X)
    0.016 / scale_factor,   # k4: 2X → ∅ (second order, divide by scale)
    26.0                    # k5: Z → Y (first order, no change)
]

# Time spans - oscillations will be on similar timescale
tspan_total = (0.0, 2.0)        
tspan_learning = (0.0, 1.0)     

# Configuration for scaled system
oregonator_config = InverseProblemConfig(
    mass_threshold = 0.97,           
    λ_frobenius = 1e-6,              
    λ_prob_conservation = 1e-6,
    dt_snapshot = 0.1,              
    dt_window = 0.5,                 
    snapshots_per_window = 10,       
    max_windows = 16                 
)

# ============================================================================
# RUN EXPERIMENT
# ============================================================================

println("\n" * "="^70)
println("OREGONATOR OSCILLATING SYSTEM (SCALED)")
println("="^70)
println("\nScaling strategy:")
println("  Original: X=500, Y=1000, Z=2000 → STATE SPACE TOO LARGE")
println("  Scaled:   X=10,  Y=20,   Z=40   → Manageable (~10³-10⁴ states)")
println()
println("Rate scaling for mass-action kinetics:")
println("  k1 (Y→X):     2.0     (first order, unchanged)")
println("  k2 (X+Y→∅):   0.002   (second order, scaled by 1/50)")
println("  k3 (X→2X+Z):  104.0   (autocatalytic, unchanged)")
println("  k4 (2X→∅):    0.00032 (second order, scaled by 1/50)")
println("  k5 (Z→Y):     26.0    (first order, unchanged)")
println()
println("Expected behavior:")
println("  - Still exhibits oscillatory dynamics")
println("  - State space ~10³-10⁴ states (vs 10⁷-10⁸ unscaled)")
println("  - Constant rates should be learned across oscillation cycles")
println()

result = run_inverse_experiment(
    oregonator_rn,
    u0_scaled,
    true_rates_scaled;
    n_trajectories = 500,           
    tspan = tspan_total,
    tspan_learning = tspan_learning,
    config = oregonator_config,
    seed = 1234,
    verbose = true
)

# ============================================================================
# ANALYZE RESULTS
# ============================================================================

print_results(result, skip_transient=3)

# ============================================================================
# GENERATE SIAM FIGURES
# ============================================================================

println("\n" * "="^70)
println("GENERATING PUBLICATION FIGURES")
println("="^70)

save_experiment_figures_siam(
    result,
    output_dir = "oregonator_figures",
    formats = [:pdf],
    skip_transient = 3  
)

println("\n" * "="^70)
println("OREGONATOR EXPERIMENT COMPLETE")
println("="^70)
println("\nTrue (scaled) rates: ", true_rates_scaled)
println("\nFigures in oregonator_figures/")
println("\nFor SIAM paper:")
println("  - Demonstrates learning constant rates from oscillatory distributions")
println("  - Tests robustness to autocatalytic reactions and feedback loops")
println("  - Shows adaptive state space tracking oscillatory probability flow")
println("  - Validates method on non-equilibrium dynamics")
println("="^70)
