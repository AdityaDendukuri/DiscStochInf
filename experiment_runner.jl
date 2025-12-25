"""
High-level experiment runner for inverse CME problems.

Usage:
    # Define your reaction network
    rn = @reaction_network begin
        k1, S + E --> SE
        k2, SE --> S + E
        k3, SE --> P + E
    end
    
    # Run experiment
    result = run_inverse_experiment(
        rn, 
        u0 = [:S => 50, :E => 10, :SE => 1, :P => 1],
        true_rates = [0.01, 0.1, 0.1],
        n_trajectories = 5000
    )
    
    # Analyze results
    print_results(result)
"""

push!(LOAD_PATH, @__DIR__)

using Catalyst
using JumpProcesses
using LinearAlgebra
using Printf

# Load modules
include("types.jl")
include("data_generation.jl")
include("state_space.jl")
include("optimization.jl")
include("analysis.jl")

# Load DiscStochSim
include("src/DiscStochSim.jl")
using .DiscStochSim

"""
    run_inverse_experiment(rn, u0, true_rates; kwargs...)

Run complete inverse CME experiment pipeline.

# Arguments
- `rn`: Catalyst reaction network
- `u0`: Initial condition (e.g., [:S => 50, :E => 10])
- `true_rates`: True rate constants (for validation)

# Keyword Arguments
- `n_trajectories::Int = 5000`: Number of SSA trajectories
- `tspan::Tuple = (0.0, 200.0)`: Time span for simulation
- `tspan_learning::Tuple = (0.0, 150.0)`: Time span for learning (subset of tspan)
- `config::InverseProblemConfig = default_config()`: Learning configuration
- `propensity_fn::PropensityFunction = nothing`: Custom propensity function (auto-detected if not provided)
- `seed::Int = 1234`: Random seed
- `verbose::Bool = true`: Print progress

# Returns
- `ExperimentResult`: Contains all results, generators, and diagnostics
"""
function run_inverse_experiment(
    rn::ReactionSystem,
    u0::Vector{Pair{Symbol, Int}},
    true_rates::Vector{Float64};
    n_trajectories::Int = 5000,
    tspan::Tuple{Float64, Float64} = (0.0, 200.0),
    tspan_learning::Tuple{Float64, Float64} = (0.0, 150.0),
    config::InverseProblemConfig = default_config(),
    propensity_fn::Union{PropensityFunction, Nothing} = nothing,
    seed::Int = 1234,
    verbose::Bool = true
)
    
    if verbose
        println("="^70)
        println("INVERSE CME EXPERIMENT")
        println("="^70)
        println("\nReaction network: $(length(reactions(rn))) reactions, $(length(species(rn))) species")
        println("True rates: $true_rates")
        println("Trajectories: $n_trajectories")
        println("Time span: $tspan")
        println("Learning span: $tspan_learning")
        println("\nConfiguration:")
        println(config)
    end
    
    # Build parameter mapping
    ps = build_parameter_mapping(rn, true_rates)
    
    # Step 1: Generate trajectories
    if verbose
        println("\n" * "="^70)
        println("STEP 1: GENERATING SSA TRAJECTORIES")
        println("="^70)
    end
    
    ssa_trajs = generate_ssa_data(rn, u0, tspan, ps, n_trajectories; seed=seed)
    
    if verbose
        println("✓ Generated $n_trajectories trajectories")
        println("  Average length: $(mean(length(t.t) for t in ssa_trajs)) time points")
    end
    
    # Step 2: Convert to distributions
    if verbose
        println("\n" * "="^70)
        println("STEP 2: CONVERTING TO DISTRIBUTIONS")
        println("="^70)
    end
    
    T, distrs = convert_to_distributions(ssa_trajs, tspan_learning, config.dt_snapshot)
    
    if verbose
        println("✓ Created $(length(distrs)) distributions")
        println("  Time range: [$(T[1]), $(T[end])]")
        println("  Unique states in first dist: $(length(distrs[1]))")
    end
    
    # Step 3: Build state space (for compatibility)
    if verbose
        println("\n" * "="^70)
        println("STEP 3: BUILDING STATE SPACE")
        println("="^70)
    end
    
    X = build_fsp_state_space(rn, u0)
    padded_dists = pad_distributions(distrs, X)
    
    if verbose
        println("✓ Built state space: $(length(X)) states")
    end
    
    # Step 4: Infer stoichiometry
    if verbose
        println("\n" * "="^70)
        println("STEP 4: INFERRING STOICHIOMETRY")
        println("="^70)
    end
    
    inferred_stoich_raw = infer_reactions_from_trajectories(ssa_trajs)
    
    if verbose
        println("✓ Inferred $(length(inferred_stoich_raw)) reactions (raw order):")
        for (i, ν) in enumerate(inferred_stoich_raw)
            println("  R$i: $ν")
        end
    end
    
    # Match to Catalyst network order
    if verbose
        println("\n  Matching to Catalyst network order...")
    end
    
    perm, inferred_stoich = match_stoichiometries(rn, inferred_stoich_raw)
    
    if verbose
        println("✓ Matched stoichiometry (Catalyst order):")
        for (i, ν) in enumerate(inferred_stoich)
            println("  R$i: $ν (from inferred R$(perm[i]))")
        end
    end
    
    # Step 4b: Build global state space
    if verbose
        println("\n" * "="^70)
        println("STEP 4b: BUILDING GLOBAL STATE SPACE FOR RATE EXTRACTION")
        println("="^70)
    end
    
    X_global = build_global_state_space(distrs, inferred_stoich, connectivity_depth=2)
    
    if verbose
        println("✓ Global state space: $(length(X_global)) states")
        println("  (vs $(length(X)) states in FSP space)")
    end
    
    # Step 5: Create windows
    if verbose
        println("\n" * "="^70)
        println("STEP 5: CREATING SLIDING WINDOWS")
        println("="^70)
    end
    
    windows = create_windows(T, padded_dists, config, inferred_stoich)
    
    if verbose
        println("✓ Created $(length(windows)) windows")
        println("  Window duration: $(config.dt_window) time units")
        println("  Snapshots per window: $(config.snapshots_per_window)")
    end
    
    # Step 6: Learn generators
    if verbose
        println("\n" * "="^70)
        println("STEP 6: LEARNING GENERATORS")
        println("="^70)
    end
    
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
        
        if !conv_info.success && verbose
            @warn "Optimization did not converge for window $(window_data.window_idx)"
        end
    end
    
    if verbose
        println("\n✓ Learned $(length(learned_generators)) generators")
    end
    
    # Step 7: Extract rates
    if verbose
        println("\n" * "="^70)
        println("STEP 7: EXTRACTING RATES WITH GLOBAL STATE SPACE")
        println("="^70)
    end
    
    # Auto-detect or use provided propensity function
    if propensity_fn === nothing
        propensity_fn = auto_detect_propensity_function(rn, inferred_stoich)
        if verbose
            println("✓ Auto-detected propensity function")
        end
    end
    
    # Create result object
    result = OptimizationResult(learned_generators, inferred_stoich, config)
    
    # Print comparison (without skip_transient for now - can be added manually)
    print_rate_comparison(result, true_rates, propensity_fn, X_global)
    
    # Return comprehensive results
    return ExperimentResult(
         rn,
        true_rates,
         learned_generators,
         inferred_stoich,
         perm,  # Store permutation for rate reordering
         X_global,
         config,
         ssa_trajs,
         (T, distrs),
         windows
    )
end

"""
    default_config()

Return default configuration for inverse problem.
"""
function default_config()
    return InverseProblemConfig(
        mass_threshold = 0.95,
        λ_frobenius = 1e-6,
        λ_prob_conservation = 1e-6,
        dt_snapshot = 0.1,
        dt_window = 2.0,
        snapshots_per_window = 10,
        max_windows = 10
    )
end

"""
    match_stoichiometries(catalyst_rn, inferred_stoich)

Match inferred stoichiometry vectors to the Catalyst reaction network.
Returns a permutation vector and reordered stoichiometry.

# Returns
- `perm`: Permutation such that inferred_stoich[perm[i]] matches catalyst reaction i
- `reordered_stoich`: Stoichiometry in Catalyst network order
"""
function match_stoichiometries(catalyst_rn, inferred_stoich)
    # Extract true stoichiometry from Catalyst network
    n_species = length(species(catalyst_rn))
    n_reactions = length(reactions(catalyst_rn))
    
    true_stoich = []
    
    for rxn in reactions(catalyst_rn)
        # Compute net stoichiometry: products - reactants
        ν = zeros(Int, n_species)
        
        # Products
        for (spec, coeff) in zip(rxn.products, rxn.prodstoich)
            spec_idx = findfirst(s -> isequal(s, spec), species(catalyst_rn))
            if spec_idx !== nothing
                ν[spec_idx] += Int(coeff)
            end
        end
        
        # Reactants (subtract)
        for (spec, coeff) in zip(rxn.substrates, rxn.substoich)
            spec_idx = findfirst(s -> isequal(s, spec), species(catalyst_rn))
            if spec_idx !== nothing
                ν[spec_idx] -= Int(coeff)
            end
        end
        
        push!(true_stoich, ν)
    end
    
    # Match inferred to true
    perm = Int[]
    matched_indices = Set{Int}()
    
    for true_ν in true_stoich
        # Find matching inferred stoichiometry
        match_idx = findfirst(inferred_ν -> inferred_ν == true_ν, inferred_stoich)
        
        if match_idx === nothing
            error("Could not match true stoichiometry $true_ν to any inferred stoichiometry")
        end
        
        if match_idx ∈ matched_indices
            error("Stoichiometry $true_ν matched multiple reactions")
        end
        
        push!(perm, match_idx)
        push!(matched_indices, match_idx)
    end
    
    # Reorder inferred stoichiometry to match Catalyst order
    reordered_stoich = [inferred_stoich[i] for i in perm]
    
    return perm, reordered_stoich
end

"""
    build_fsp_state_space(rn, u0; bounds=(0, 200), expansions=10)

Build FSP state space using DiscStochSim.
"""
function build_fsp_state_space(rn, u0; bounds=(0, 200), expansions=10)
    model = DiscStochSim.DiscreteStochasticSystem(rn)
    boundary_condition(x) = DiscStochSim.RectLatticeBoundaryCondition(x, bounds)
    
    # Convert u0 to CartesianIndex
    u0_vals = [val for (sym, val) in u0]
    U0 = CartesianIndex(u0_vals...)
    
    X = Set([U0])
    p = [1.0]
    
    # Expand state space
    X, p = DiscStochSim.expand!(X, p, model, boundary_condition, expansions)
    
    return X
end

"""
    auto_detect_propensity_function(rn, stoich_vecs)

Automatically create propensity function from reaction network structure.
Uses mass-action kinetics.
"""
function auto_detect_propensity_function(rn::ReactionSystem, stoich_vecs::Vector{Vector{Int}})
    # Extract stoichiometric matrix for reactants
    n_species = length(species(rn))
    n_reactions = length(reactions(rn))
    
    reactant_stoich = zeros(Int, n_species, n_reactions)
    
    for (j, rxn) in enumerate(reactions(rn))
        # Get reactant stoichiometry from the reaction
        # rxn.substrates contains the reactant species
        # rxn.substoich contains their coefficients
        
        for (spec, coeff) in zip(rxn.substrates, rxn.substoich)
            # Find index of this species in the full species list
            # Use isequal for symbolic comparison
            spec_idx = findfirst(s -> isequal(s, spec), species(rn))
            if spec_idx !== nothing
                reactant_stoich[spec_idx, j] = Int(coeff)
            end
        end
    end
    
    return MassActionPropensity(reactant_stoich)
end

"""
    ExperimentResult

Container for all experiment results.
"""
struct ExperimentResult
    reaction_network::ReactionSystem
    true_rates::Vector{Float64}
    learned_generators::Vector{LearnedGenerator}
    inferred_stoich::Vector{Vector{Int}}
    global_state_space::Set
    config::InverseProblemConfig
    trajectories::Vector
    distributions::Tuple
    windows::Vector{WindowData}
end

"""
    print_results(result; skip_transient=3)

Print comprehensive results summary.
"""
function print_results(result::ExperimentResult; skip_transient=3)
    println("\n" * "="^70)
    println("EXPERIMENT SUMMARY")
    println("="^70)
    
    println("\nReaction network:")
    println("  Species: $(length(species(result.reaction_network)))")
    println("  Reactions: $(length(reactions(result.reaction_network)))")
    
    println("\nData:")
    println("  Trajectories: $(length(result.trajectories))")
    println("  Windows: $(length(result.windows))")
    println("  Global state space: $(length(result.global_state_space)) states")
    
    println("\nLearned generators:")
    println("  Total: $(length(result.learned_generators))")
    println("  State space range: $(length(result.learned_generators[1].state_space))-$(length(result.learned_generators[end].state_space)) states")
    
    println("\nTrue rates: $(result.true_rates)")
    
    # Extract final rates
    propensity_fn = auto_detect_propensity_function(result.reaction_network, result.inferred_stoich)
    final_gen = result.learned_generators[end]
    final_stats = extract_rates_global(final_gen, result.global_state_space, propensity_fn, result.inferred_stoich)
    
    println("\nFinal learned rates (median):")
    for j in 1:length(result.inferred_stoich)
        stats = final_stats[j]
        true_k = result.true_rates[j]
        
        if stats.n_transitions > 0
            rel_error = abs(stats.median - true_k) / true_k * 100
            println("  R$j: $(round(stats.median, sigdigits=3)) (true: $true_k, error: $(round(rel_error, digits=1))%, n=$(stats.n_transitions))")
        else
            println("  R$j: N/A (no transitions found)")
        end
    end
end

# Export main functions
export run_inverse_experiment, default_config, print_results, ExperimentResult
