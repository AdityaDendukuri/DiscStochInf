"""
Main experiment script for inverse problem: learning CME generators from trajectory data.

This script demonstrates the full workflow:
1. Generate SSA trajectory data
2. Convert to probability distributions
3. Infer stoichiometry  
4. Learn generators using sliding windows
5. Extract and compare rates
"""

# Add src directory to load path if needed
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

# Also need DiscStochSim for certain utilities
# Adjust path as needed based on your setup
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
    mass_threshold = 0.95,
    λ_frobenius = 1e-6,
    λ_prob_conservation = 1e-6,
    dt_snapshot = 0.1,
    dt_window = 2.0,
    snapshots_per_window = 20,
    max_windows = 10
)

println("="^60)
println("INVERSE PROBLEM EXPERIMENT")
println("="^60)
println("\nConfiguration:")
println(config)
println("\nTrue parameters:")
println("  Rates: $true_rates")
println("  Initial condition: $u0_integers")
println("  Time span: $tspan")

# ============================================================================
# STEP 1: GENERATE SSA DATA
# ============================================================================

println("\n" * "="^60)
println("STEP 1: GENERATING SSA TRAJECTORIES")
println("="^60)

n_trajs = 10000
ssa_trajs = generate_ssa_data(rn, u0_integers, tspan, ps, n_trajs; seed=1234)

println("Generated $n_trajs trajectories")
println("Example trajectory length: $(length(ssa_trajs[1].t)) time points")

# ============================================================================
# STEP 2: CONVERT TO DISTRIBUTIONS
# ============================================================================

println("\n" * "="^60)
println("STEP 2: CONVERTING TO DISTRIBUTIONS")
println("="^60)

T, distrs = convert_to_distributions(ssa_trajs, (0.0, 150.0), config.dt_snapshot)

println("Created $(length(distrs)) probability distributions")
println("Time range: [$(T[1]), $(T[end])]")
println("Number of unique states in first distribution: $(length(distrs[1]))")

# ============================================================================
# STEP 3: BUILD STATE SPACE AND PAD DISTRIBUTIONS
# ============================================================================

println("\n" * "="^60)
println("STEP 3: BUILDING STATE SPACE")
println("="^60)

# Build state space using DiscStochSim for consistency
model = DiscStochSim.DiscreteStochasticSystem(rn)
bounds = (0, 200)
boundary_condition(x) = DiscStochSim.RectLatticeBoundaryCondition(x, bounds)

dt = 0.001
U0 = CartesianIndex(50, 10, 1, 1)
X = Set([U0])
p = [1.0]

# Expand state space
X, p = DiscStochSim.expand!(X, p, model, boundary_condition, 10)

println("State space size: $(length(X))")
println("Sample states: $(collect(X)[1:min(5, length(X))])")

# Pad distributions to include all states
padded_dists = pad_distributions(distrs, X)

println("Padded all distributions to $(length(X)) states")

# ============================================================================
# STEP 4: INFER STOICHIOMETRY
# ============================================================================

println("\n" * "="^60)
println("STEP 4: INFERRING STOICHIOMETRY")
println("="^60)

inferred_stoich = infer_reactions_from_trajectories(ssa_trajs)

println("Inferred $(length(inferred_stoich)) reactions:")
for (i, ν) in enumerate(inferred_stoich)
    println("  R$i: $ν")
end

# ============================================================================
# STEP 4b: BUILD GLOBAL STATE SPACE (for rate extraction)
# ============================================================================

println("\n" * "="^60)
println("STEP 4b: BUILDING GLOBAL STATE SPACE FOR RATE EXTRACTION")
println("="^60)

# Build comprehensive global state space from all distributions
# This is used for rate extraction to ensure complete coverage
X_global = build_global_state_space(distrs, inferred_stoich, connectivity_depth=2)

println("Global state space size: $(length(X_global))")
println("Sample global states: $(collect(X_global)[1:min(5, length(X_global))])")

# Pad distributions to global state space for rate extraction
padded_dists_global = pad_distributions(distrs, X_global)

println("\nNote: Local optimization will use smaller state spaces,")
println("      but rate extraction will use this global space for better coverage.")

# ============================================================================
# STEP 5: CREATE WINDOWS
# ============================================================================

println("\n" * "="^60)
println("STEP 5: CREATING SLIDING WINDOWS")
println("="^60)

windows = create_windows(T, padded_dists, config, inferred_stoich)

println("Created $(length(windows)) windows")
println("Each window covers $(config.dt_window) time units")
println("Each window contains $(config.snapshots_per_window) snapshots")

# ============================================================================
# STEP 6: LEARN GENERATORS
# ============================================================================

println("\n" * "="^60)
println("STEP 6: LEARNING GENERATORS")
println("="^60)

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

# ============================================================================
# STEP 7: EXTRACT AND COMPARE RATES
# ============================================================================

println("\n" * "="^60)
println("STEP 7: EXTRACTING RATES WITH GLOBAL STATE SPACE")
println("="^60)

# Define propensity function for this system
# For the Michaelis-Menten system:
# R1: S + E -> SE  (propensity = S * E)
# R2: SE -> S + E  (propensity = SE)
# R3: SE -> P + E  (propensity = SE)

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

# Print comprehensive comparison using GLOBAL state space
print_rate_comparison(result, true_rates, propensity_fn, X_global)

# ============================================================================
# STEP 8: ANALYZE FINAL GENERATOR
# ============================================================================

if !isempty(learned_generators)
    println("\n" * "="^60)
    println("STEP 8: ANALYZING FINAL GENERATOR")
    println("="^60)
    
    analyze_generator_properties(learned_generators[end])
end

# ============================================================================
# SUMMARY
# ============================================================================

println("\n" * "="^60)
println("EXPERIMENT COMPLETE")
println("="^60)
println("\nProcessed $(length(windows)) windows")
println("Learned $(length(learned_generators)) generator matrices")
println("\nUsed GLOBAL state space ($(length(X_global)) states) for rate extraction")
println("vs LOCAL state spaces ($(length(learned_generators[1].state_space))-$(length(learned_generators[end].state_space)) states) for optimization")

if !isempty(learned_generators)
    final_stats = extract_rates_global(learned_generators[end], X_global, propensity_fn, inferred_stoich)
    
    println("\nFinal rate estimates (median):")
    for j in 1:length(inferred_stoich)
        stats = final_stats[j]
        true_k = true_rates[j]
        
        if stats.n_transitions > 0
            rel_error = abs(stats.median - true_k) / true_k * 100
            println("  R$j: $(round(stats.median, sigdigits=3)) " *
                   "(true: $true_k, error: $(round(rel_error, digits=1))%)")
        else
            println("  R$j: N/A (no transitions found)")
        end
    end
end

println("\n" * "="^60)
