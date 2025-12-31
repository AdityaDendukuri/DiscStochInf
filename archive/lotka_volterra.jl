"""
Self-contained Lotka-Volterra test with memory-efficient generation.
Everything in one file to avoid world age issues.
"""

push!(LOAD_PATH, @__DIR__)

using Catalyst, JumpProcesses, LinearAlgebra, Statistics, Random

# Load base modules first
include("data_generation.jl")
include("types.jl")
include("state_space.jl") 
include("optimization.jl")
include("analysis.jl")
include("experiment_runner.jl")
include("src/DiscStochSim.jl")
using .DiscStochSim

# ============================================================================
# MEMORY-EFFICIENT DATA GENERATION (inline to avoid world age)
# ============================================================================

function build_fsp_state_space(rn, u0; bounds=(0, 200), expansions=10)
    model = DiscStochSim.DiscreteStochasticSystem(rn)
    boundary_condition(x) = DiscStochSim.RectLatticeBoundaryCondition(x, bounds)
    
    u0_vals = [val for (sym, val) in u0]
    U0 = CartesianIndex(u0_vals...)
    
    X = Set([U0])
    p = [1.0]
    
    X, p = DiscStochSim.expand!(X, p, model, boundary_condition, expansions)
    
    return X
end
function generate_distributions_memory_efficient(
    rn::ReactionSystem,
    u0::Vector{Pair{Symbol, Int}},
    tspan::Tuple{Float64, Float64},
    ps::Vector{Pair{Symbol, Float64}},
    n_trajectories::Int;
    dt_snapshot::Float64 = 0.1,
    seed::Int = 1234
)
    Random.seed!(seed)
    
    time_grid = collect(tspan[1]:dt_snapshot:tspan[2])
    n_times = length(time_grid)
    
    # Count states
    distributions = [Dict{Vector{Int}, Int}() for _ in 1:n_times]
    
    println("Generating $n_trajectories trajectories (memory-efficient)...")
    prog_interval = max(1, n_trajectories ÷ 10)
    
    for i in 1:n_trajectories
        dprob = DiscreteProblem(rn, u0, tspan, ps)
        jprob = JumpProblem(rn, dprob, Direct())
        sol = solve(jprob, SSAStepper(), saveat=time_grid)
        
        for (t_idx, t) in enumerate(time_grid)
            if t_idx <= length(sol.t)
                state = Int.(sol.u[t_idx])
                distributions[t_idx][state] = get(distributions[t_idx], state, 0) + 1
            end
        end
        
        if i % prog_interval == 0
            println("  Progress: $i/$n_trajectories ($(round(i/n_trajectories*100, digits=1))%)")
            GC.gc()
        end
    end
    
    # Convert to probabilities
    prob_distributions = Vector{Dict{Vector{Int}, Float64}}()
    for counts in distributions
        prob_dist = Dict{Vector{Int}, Float64}()
        total = sum(values(counts))
        if total > 0
            for (state, count) in counts
                prob_dist[state] = count / total
            end
        end
        push!(prob_distributions, prob_dist)
    end
    
    println("✓ Generated distributions from $n_trajectories trajectories")
    return time_grid, prob_distributions
end

# ============================================================================
# System
# ============================================================================

rn = @reaction_network begin
    k1, P + Y --> 2Y    # Predation (k=0.01)
    k2, Y --> 0         # Death (k=0.3)
    k3, P --> 2P        # Birth (k=1.0)
end

true_rates = [0.01, 0.3, 1.0]
u0 = [:P => 50, :Y => 100]
ps = [:k1 => true_rates[1], :k2 => true_rates[2], :k3 => true_rates[3]]

# ============================================================================
# Config
# ============================================================================

config = InverseProblemConfig(
    mass_threshold = 0.99,
    λ_frobenius = 1.0e-6,
    λ_prob_conservation = 0.1,
    dt_snapshot = 0.5,
    dt_window = 0.01,
    snapshots_per_window = 10,
    max_windows = 5
)

# ============================================================================
# Run experiment
# ============================================================================

println("="^80)
println("MINIMAL LOTKA-VOLTERRA EXPERIMENT")
println("="^80)
println("\nSystem:")
println("  R1: P + Y → 2Y  (k₁ = 0.01)  - Predation")
println("  R2: Y → ∅       (k₂ = 0.3)   - Death")
println("  R3: P → 2P      (k₃ = 1.0)   - Birth")

# STEP 1: Generate data (memory-efficient)
println("\n" * "="^80)
println("STEP 1: Data Generation")
println("="^80)

T, distrs = generate_distributions_memory_efficient(
    rn, u0, (0.0, 40.0), ps, 1500;
    dt_snapshot = config.dt_snapshot,
    seed = 1234
)

println("\nDistribution statistics:")
println("  Time points: $(length(T))")
println("  Mean support: $(round(mean([length(d) for d in distrs]), digits=1))")

# STEP 2: Build state space
println("\n" * "="^80)
println("STEP 2: State Space Construction")
println("="^80)

# Generate one trajectory for stoichiometry
dprob = DiscreteProblem(rn, u0, (0.0, 20.0), ps)
jprob = JumpProblem(rn, dprob, Direct())
test_sol = solve(jprob, SSAStepper())

inferred_stoich = infer_reactions_from_trajectories([test_sol])
X_global = build_global_state_space(distrs, inferred_stoich, connectivity_depth=2)

println("✓ Inferred $(length(inferred_stoich)) reactions")
println("✓ Global state space: $(length(X_global)) states")

# STEP 3: Prepare for optimization
println("\n" * "="^80)
println("STEP 3: Optimization Setup")
println("="^80)

# Pad distributions to common state space
X_fsp = build_fsp_state_space(rn, u0)
padded_dists = pad_distributions(distrs, X_fsp)

# Create windows
windows = create_windows(T, padded_dists, config, inferred_stoich)
println("✓ Created $(length(windows)) windows")

# STEP 4: Run optimization
println("\n" * "="^80)
println("STEP 4: Sliding Window Optimization")
println("="^80)

learned_generators = LearnedGenerator[]
propensity_fn = auto_detect_propensity_function(rn, inferred_stoich)

for (w_idx, window_data) in enumerate(windows)
    println("\nWindow $w_idx: t ∈ [$(round(window_data.times[1], digits=1)), $(round(window_data.times[end], digits=1))]")
    
    A_learned, X_local, conv_info = optimize_local_generator(window_data, config)
    
    learned_gen = LearnedGenerator(
        window_data.times[1],
        A_learned,
        X_local,
        conv_info
    )
    push!(learned_generators, learned_gen)
    
    if conv_info.success
        improvement_pct = 100 * conv_info.improvement / conv_info.initial_objective
        println("  ✓ Converged: $(round(improvement_pct, digits=1))% improvement")
    else
        println("  ✗ Failed: $(conv_info.return_code)")
    end
end

# STEP 5: Extract rates
println("\n" * "="^80)
println("STEP 5: Rate Extraction")
println("="^80)

aggregated_stats = extract_rates_aggregated(learned_generators, propensity_fn, inferred_stoich)

println("\nResults:")
for j in 1:length(inferred_stoich)
    stats = aggregated_stats[j]
    true_k = true_rates[j]
    
    if stats.n_transitions > 0
        err_pct = abs(stats.median - true_k) / true_k * 100
        println("  R$j: learned=$(round(stats.median, sigdigits=3)), true=$true_k")
        println("      error=$(round(err_pct, digits=1))%, transitions=$(stats.n_transitions)")
    else
        println("  R$j: NO TRANSITIONS")
    end
end

# Summary
errors = [abs(aggregated_stats[j].median - true_rates[j]) / true_rates[j] 
          for j in 1:length(inferred_stoich) 
          if aggregated_stats[j].n_transitions > 0]

if !isempty(errors)
    mean_error = mean(errors)
    println("\nMean relative error: $(round(mean_error*100, digits=1))%")
end

println("\n" * "="^80)
println("✅ Experiment complete!")
println("="^80)
