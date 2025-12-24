"""
Example: Michaelis-Menten enzyme kinetics

Simple usage of the high-level API.
"""

using Catalyst

include("experiment_runner.jl")

# ============================================================================
# DEFINE REACTION NETWORK
# ============================================================================

rn = @reaction_network begin
    k1, S + E --> SE
    k2, SE --> S + E
    k3, SE --> P + E
end

# ============================================================================
# RUN EXPERIMENT
# ============================================================================

result = run_inverse_experiment(
    rn,
    [:S => 50, :E => 10, :SE => 1, :P => 1],  # Initial condition
    [0.01, 0.1, 0.1];                         # True rates
    n_trajectories = 5000,
    tspan = (0.0, 200.0),
    tspan_learning = (0.0, 150.0),
    seed = 1234,
    verbose = true
)

# ============================================================================
# ANALYZE RESULTS
# ============================================================================

print_results(result, skip_transient=3)

println("\n" * "="^70)
println("Done! Results stored in 'result' variable.")
println("="^70)
