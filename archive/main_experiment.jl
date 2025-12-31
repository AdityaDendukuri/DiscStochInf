"""
Main experiment script for inverse problem: learning CME generators from trajectory data.

This script demonstrates the full workflow:
1. Generate SSA trajectory data
2. Convert to probability distributions
3. Infer stoichiometry  
4. Learn generators using sliding windows
5. Extract and compare rates
"""

push!(LOAD_PATH, @__DIR__)

using Catalyst
using JumpProcesses
using LinearAlgebra
using Printf

# Load inverse problem modules
include("types.jl")
include("data_generation.jl")
include("state_space.jl")
include("optimization.jl")
include("analysis.jl")

# Also need DiscStochSim
include("src/DiscStochSim.jl")
using .DiscStochSim

# ============================================================================
# CONFIGURATION
# ============================================================================

# Define the reaction network (Michaelis-Menten enzyme kinetics)
rn = @reaction_network begin
    k1, S + E --> SE
    k2, SE --> S + E
    k3, SE --> P + E
end

# True parameters
u0_integers = [:S => 50, :E => 10, :SE => 1, :P => 1]
tspan = (0.0, 200.0)
true_rates = [0.01, 0.1, 0.1]
ps = [:k1 => true_rates[1], :k2 => true_rates[2], :k3 => true_rates[3]]

# Inverse problem configuration
config = InverseProblemConfig(
    mass_threshold = 0.99,        # Increased from 0.95
    λ_frobenius = 1e-6,
    λ_prob_conservation = 0.1,    # Increased from 1e-6
    dt_snapshot = 0.1,
    dt_window = 2.0,
    snapshots_per_window = 10,
    max_windows = 10
)

println("="^70)
println("INVERSE CME EXPERIMENT")
println("="^70)
println("\nConfiguration:")
println(config)
println("\nTrue parameters:")
println("  Rates: $true_rates")
println("  Initial condition: $u0_integers")
println("  Time span: $tspan")

# ============================================================================
# STEP 1: GENERATE SSA DATA
# ============================================================================

println("\n" * "="^70)
println("STEP 1: GENERATING SSA TRAJECTORIES")
println("="^70)

n_trajs = 10000
ssa_trajs = generate_ssa_data(rn, u0_integers, tspan, ps, n_trajs; seed=1234)

println("✓ Generated $n_trajs trajectories")
println("  Average length: $(round(mean(length(t.t) for t in ssa_trajs), digits=1)) time points")

# ============================================================================
# STEP 2: CONVERT TO DISTRIBUTIONS
# ============================================================================

println("\n" * "="^70)
println("STEP 2: CONVERTING TO DISTRIBUTIONS")
println("="^70)

T, distrs = convert_to_distributions(ssa_trajs, (0.0, 150.0), config.dt_snapshot)

println("✓ Created $(length(distrs)) distributions")
println("  Time range: [$(T[1]), $(T[end])]")
println("  Unique states in first dist: $(length(distrs[1]))")

# ============================================================================
# STEP 3: BUILD STATE SPACE
# ============================================================================

println("\n" * "="^70)
println("STEP 3: BUILDING STATE SPACE")
println("="^70)

# Build state space using DiscStochSim
model = DiscStochSim.DiscreteStochasticSystem(rn)
bounds = (0, 200)
boundary_condition(x) = DiscStochSim.RectLatticeBoundaryCondition(x, bounds)

U0 = CartesianIndex(50, 10, 1, 1)
X = Set([U0])
p = [1.0]

# Expand state space
X, p = DiscStochSim.expand!(X, p, model, boundary_condition, 10)

println("✓ Built state space: $(length(X)) states")

# Pad distributions
padded_dists = pad_distributions(distrs, X)

# ============================================================================
# STEP 4: INFER STOICHIOMETRY
# ============================================================================

println("\n" * "="^70)
println("STEP 4: INFERRING STOICHIOMETRY")
println("="^70)

inferred_stoich = infer_reactions_from_trajectories(ssa_trajs)

println("✓ Inferred $(length(inferred_stoich)) reactions:")
for (i, ν) in enumerate(inferred_stoich)
    println("  R$i: $ν")
end

# ============================================================================
# STEP 5: CREATE WINDOWS
# ============================================================================

println("\n" * "="^70)
println("STEP 5: CREATING SLIDING WINDOWS")
println("="^70)

windows = create_windows(T, padded_dists, config, inferred_stoich)

println("✓ Created $(length(windows)) windows")
println("  Window duration: $(config.dt_window) time units")
println("  Snapshots per window: $(config.snapshots_per_window)")

# ============================================================================
# STEP 6: LEARN GENERATORS
# ============================================================================

println("\n" * "="^70)
println("STEP 6: LEARNING GENERATORS")
println("="^70)

learned_generators = LearnedGenerator[]

for window_data in windows
    A_learned, X_local, conv_info = optimize_local_generator(window_data, config)
    
    learned_gen = LearnedGenerator(
        window_data.times[1],
        A_learned,
        X_local,
        conv_info
    )
    
    push!(learned_generators, learned_gen)
    
    if !conv_info.success
        @warn "Optimization did not converge for window $(window_data.window_idx)"
    end
end

println("\n✓ Learned $(length(learned_generators)) generators")

# ============================================================================
# STEP 7: EXTRACT AND COMPARE RATES
# ============================================================================

println("\n" * "="^70)
println("STEP 7: EXTRACTING RATES")
println("="^70)

# Define propensity function for Michaelis-Menten
struct EnzymePropensity <: PropensityFunction end

function (prop::EnzymePropensity)(state::Tuple, reaction_idx::Int)
    S, E, SE, P = state
    
    if reaction_idx == 1
        return float(S * E)  # R1: S + E -> SE
    elseif reaction_idx == 2
        return float(SE)     # R2: SE -> S + E
    elseif reaction_idx == 3
        return float(SE)     # R3: SE -> P + E
    else
        return 1.0
    end
end

propensity_fn = EnzymePropensity()

# Create result object
result = OptimizationResult(learned_generators, inferred_stoich, config)

# Print comprehensive comparison
print_rate_comparison(result, true_rates, propensity_fn)

# ============================================================================
# STEP 8: ANALYZE FINAL GENERATOR
# ============================================================================

if !isempty(learned_generators)
    println("\n" * "="^70)
    println("STEP 8: ANALYZING FINAL GENERATOR")
    println("="^70)
    
    analyze_generator_properties(learned_generators[end])
end

# ============================================================================
# SUMMARY
# ============================================================================

println("\n" * "="^70)
println("EXPERIMENT SUMMARY")
println("="^70)

println("\nReaction network:")
println("  Species: 4")
println("  Reactions: 3")

println("\nData:")
println("  Trajectories: $n_trajs")
println("  Windows: $(length(windows))")

println("\nLearned generators:")
println("  Total: $(length(learned_generators))")
println("  State space range: $(length(learned_generators[1].state_space))-$(length(learned_generators[end].state_space)) states")

println("\nTrue rates: $true_rates")

# Extract aggregated rates
if !isempty(learned_generators)
    aggregated_stats = extract_rates_aggregated(learned_generators, propensity_fn, inferred_stoich)
    
    println("\nFinal learned rates (aggregated across all windows):")
    for j in 1:length(inferred_stoich)
        stats = aggregated_stats[j]
        true_k = true_rates[j]
        
        if stats.n_transitions > 0
            rel_error = abs(stats.median - true_k) / true_k * 100
            println("  R$j: $(round(stats.median, sigdigits=4)) (true: $true_k, error: $(round(rel_error, digits=1))%, n=$(stats.n_transitions))")
        else
            println("  R$j: N/A (no transitions found)")
        end
    end
end

println("\n" * "="^70)
println("Done! Results stored in 'result' variable.")
println("="^70)
