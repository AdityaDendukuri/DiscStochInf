"""
Analysis and rate extraction utilities for learned generators.
"""

using Statistics
using LinearAlgebra

include("types.jl")

"""
    PropensityFunction

Abstract type for propensity functions. Subtypes should implement:
    (propfn::PropensityFunction)(state, reaction_idx) -> Float64
"""
abstract type PropensityFunction end

"""
    MassActionPropensity

Mass-action propensity for a specific reaction network.

# Fields
- `stoich_reactants`: Matrix where column j gives reactant stoichiometry for reaction j
"""
struct MassActionPropensity <: PropensityFunction
    stoich_reactants::Matrix{Int}
end

function (prop::MassActionPropensity)(state::Tuple, reaction_idx::Int)
    reactants = prop.stoich_reactants[:, reaction_idx]
    propensity = 1.0
    
    for (species_idx, coeff) in enumerate(reactants)
        if coeff > 0
            # Compute binomial coefficient: state[species_idx] choose coeff
            propensity *= prod(state[species_idx] - k for k in 0:(coeff-1))
        end
    end
    
    return propensity
end

"""
    extract_rates_global(learned_gen::LearnedGenerator, 
                        global_state_space::Union{Vector,Set},
                        propensity_fn::PropensityFunction, 
                        stoich_vecs::Vector{Vector{Int}})

Extract reaction rates using a global state space for complete coverage.

This function maps the learned generator (optimized on local state space) to the 
global state space and extracts rates with better coverage.

# Arguments
- `learned_gen`: LearnedGenerator containing matrix A and local state space
- `global_state_space`: Complete state space (CartesianIndex vector or set)
- `propensity_fn`: Function to compute propensity for each reaction
- `stoich_vecs`: Vector of stoichiometry vectors

# Returns
Dictionary mapping reaction index to NamedTuple with statistics
"""
function extract_rates_global(
    learned_gen::LearnedGenerator,
    global_state_space::Union{Vector,Set},
    propensity_fn::PropensityFunction,
    stoich_vecs::Vector{Vector{Int}}
)
    A_local = learned_gen.A
    X_local = learned_gen.state_space
    
    # Convert to sets for fast lookup
    X_local_set = Set(X_local)
    X_global_vec = isa(global_state_space, Vector) ? global_state_space : collect(global_state_space)
    X_global_set = Set(X_global_vec)
    
    # Build mapping from global indices to local indices
    local_idx_map = Dict{CartesianIndex{4}, Int}()
    for (idx, state) in enumerate(X_local)
        local_idx_map[state] = idx
    end
    
    rate_stats = Dict{Int, NamedTuple}()
    
    for (j, ν) in enumerate(stoich_vecs)
        rates_found = Float64[]
        
        # Iterate over GLOBAL state space
        for state_from in X_global_vec
            state_to = CartesianIndex(tuple((collect(Tuple(state_from)) .+ ν)...))
            
            # Check if this transition is valid in global space
            if state_to ∈ X_global_set
                # Try to get rate from learned generator
                rate_entry = nothing
                
                if (state_from ∈ X_local_set) && (state_to ∈ X_local_set)
                    # Both states in local space - direct lookup
                    idx_from_local = local_idx_map[state_from]
                    idx_to_local = local_idx_map[state_to]
                    rate_entry = A_local[idx_to_local, idx_from_local]
                    
                elseif state_from ∈ X_local_set
                    # Only source state in local space
                    # Estimate from diagonal (exit rate contribution)
                    idx_from_local = local_idx_map[state_from]
                    
                    # The diagonal entry includes exit rates for all reactions
                    # We can estimate this reaction's contribution
                    propensity = propensity_fn(Tuple(state_from), j)
                    if propensity > 0
                        # This is an approximation - we're estimating the rate
                        # from the total exit rate, assuming this is a significant contributor
                        total_exit = -A_local[idx_from_local, idx_from_local]
                        
                        # Count how many reactions can fire from this state
                        n_active_reactions = 0
                        for ν_test in stoich_vecs
                            state_test = CartesianIndex(tuple((collect(Tuple(state_from)) .+ ν_test)...))
                            if all(Tuple(state_test) .>= 0)
                                n_active_reactions += 1
                            end
                        end
                        
                        if n_active_reactions > 0
                            # Simple heuristic: divide exit rate equally among active reactions
                            # This is rough but better than nothing
                            rate_entry = total_exit / n_active_reactions
                        end
                    end
                end
                
                # Extract rate if we have a valid entry
                if rate_entry !== nothing && abs(rate_entry) > 1e-10
                    propensity = propensity_fn(Tuple(state_from), j)
                    
                    if propensity > 0
                        k_estimate = rate_entry / propensity
                        # Sanity check: rates should be positive and reasonable
                        if k_estimate > 0 && k_estimate < 100.0
                            push!(rates_found, k_estimate)
                        end
                    end
                end
            end
        end
        
        # Compute statistics
        if !isempty(rates_found)
            rate_stats[j] = (
                mean = mean(rates_found),
                median = median(rates_found),
                std = std(rates_found),
                min = minimum(rates_found),
                max = maximum(rates_found),
                n_transitions = length(rates_found)
            )
        else
            rate_stats[j] = (
                mean = NaN,
                median = NaN,
                std = NaN,
                min = NaN,
                max = NaN,
                n_transitions = 0
            )
        end
    end
    
    return rate_stats
end

"""
    extract_rates(learned_gen::LearnedGenerator, propensity_fn::PropensityFunction, 
                  stoich_vecs::Vector{Vector{Int}})

Extract reaction rates from learned generator by dividing out propensities.

# Arguments
- `learned_gen`: LearnedGenerator containing matrix A and state space
- `propensity_fn`: Function to compute propensity for each reaction
- `stoich_vecs`: Vector of stoichiometry vectors

# Returns
Dictionary mapping reaction index to NamedTuple with statistics:
- `mean`: Mean extracted rate
- `median`: Median extracted rate  
- `std`: Standard deviation
- `min`, `max`: Range of extracted rates
- `n_transitions`: Number of transitions found
"""
function extract_rates(
    learned_gen::LearnedGenerator,
    propensity_fn::PropensityFunction,
    stoich_vecs::Vector{Vector{Int}}
)
    A = learned_gen.A
    X = learned_gen.state_space
    
    rate_stats = Dict{Int, NamedTuple}()
    
    for (j, ν) in enumerate(stoich_vecs)
        rates_found = Float64[]
        
        for (idx_from, state_from) in enumerate(X)
            # Compute target state
            state_to = tuple((collect(Tuple(state_from)) .+ ν)...)
            state_to_cart = CartesianIndex(state_to...)
            
            # Check if transition exists in state space
            if state_to_cart ∈ Set(X)
                idx_to = findfirst(==(state_to_cart), X)
                
                if idx_to !== nothing
                    rate_entry = A[idx_to, idx_from]
                    
                    if abs(rate_entry) > 1e-10
                        # Compute propensity
                        propensity = propensity_fn(Tuple(state_from), j)
                        
                        if propensity > 0
                            k_estimate = rate_entry / propensity
                            push!(rates_found, k_estimate)
                        end
                    end
                end
            end
        end
        
        # Compute statistics
        if !isempty(rates_found)
            rate_stats[j] = (
                mean = mean(rates_found),
                median = median(rates_found),
                std = std(rates_found),
                min = minimum(rates_found),
                max = maximum(rates_found),
                n_transitions = length(rates_found)
            )
        else
            rate_stats[j] = (
                mean = NaN,
                median = NaN,
                std = NaN,
                min = NaN,
                max = NaN,
                n_transitions = 0
            )
        end
    end
    
    return rate_stats
end

"""
    print_rate_comparison(result::OptimizationResult, true_rates::Vector{Float64},
                         propensity_fn::PropensityFunction, global_state_space=nothing)

Print comparison of learned rates versus true rates across all windows.
If global_state_space is provided, uses global extraction method.
"""
function print_rate_comparison(
    result::OptimizationResult,
    true_rates::Vector{Float64},
    propensity_fn::PropensityFunction,
    global_state_space=nothing
)
    println("\n" * "="^60)
    println("RATE COMPARISON: LEARNED vs TRUE")
    println("="^60)
    
    println("\nTrue rates: $true_rates")
    println("True stoichiometry:")
    for (j, ν) in enumerate(result.inferred_stoich)
        println("  R$j: $ν  (k=$(true_rates[j]))")
    end
    
    # Choose extraction method
    extraction_method = isnothing(global_state_space) ? "LOCAL" : "GLOBAL"
    println("\nExtraction method: $extraction_method")
    
    # Analyze each window
    println("\n" * "-"^60)
    println("Learned rates over time:")
    println("-"^60)
    
    for learned_gen in result.learned_generators
        # Use global extraction if available
        if isnothing(global_state_space)
            rate_stats = extract_rates(learned_gen, propensity_fn, result.inferred_stoich)
        else
            rate_stats = extract_rates_global(learned_gen, global_state_space, propensity_fn, result.inferred_stoich)
        end
        
        println("\nWindow at t=$(learned_gen.t_start):")
        for (j, ν) in enumerate(result.inferred_stoich)
            stats = rate_stats[j]
            true_k = true_rates[j]
            
            if stats.n_transitions > 0
                rel_error = abs(stats.median - true_k) / true_k * 100
                println("  R$j: median=$(round(stats.median, sigdigits=3)), " *
                       "true=$(true_k), error=$(round(rel_error, digits=1))%, " *
                       "n=$(stats.n_transitions)")
            else
                println("  R$j: NO TRANSITIONS FOUND")
            end
        end
    end
    
    # Final summary
    if !isempty(result.learned_generators)
        println("\n" * "="^60)
        println("FINAL WINDOW SUMMARY")
        println("="^60)
        
        final_gen = result.learned_generators[end]
        
        if isnothing(global_state_space)
            final_stats = extract_rates(final_gen, propensity_fn, result.inferred_stoich)
        else
            final_stats = extract_rates_global(final_gen, global_state_space, propensity_fn, result.inferred_stoich)
        end
        
        println("\nReaction | True Rate | Learned (median) | Rel. Error | Transitions")
        println("-"^70)
        
        for j in 1:length(result.inferred_stoich)
            stats = final_stats[j]
            true_k = true_rates[j]
            
            if stats.n_transitions > 0
                rel_error = abs(stats.median - true_k) / true_k * 100
                @printf("   R%-2d   |   %.3f    |      %.3f       |   %5.1f%%   |     %d\n",
                       j, true_k, stats.median, rel_error, stats.n_transitions)
            else
                println("   R$j   |   $true_k    |       N/A        |     N/A    |     0")
            end
        end
    end
end

"""
    analyze_generator_properties(learned_gen::LearnedGenerator)

Analyze mathematical properties of the learned generator.
"""
function analyze_generator_properties(learned_gen::LearnedGenerator)
    A = learned_gen.A
    
    println("\n" * "="^60)
    println("GENERATOR PROPERTIES")
    println("="^60)
    
    # Basic statistics
    println("\nMatrix statistics:")
    println("  Size: $(size(A))")
    println("  Frobenius norm: $(round(norm(A), sigdigits=4))")
    println("  Max absolute entry: $(round(maximum(abs.(A)), sigdigits=4))")
    println("  Nonzeros: $(count(abs.(A) .> 1e-10))")
    println("  Sparsity: $(round(100 * (1 - count(abs.(A) .> 1e-10) / length(A)), digits=1))%")
    
    # Column sum errors (should be zero)
    col_sums = vec(sum(A, dims=1))
    println("\nColumn sum errors:")
    println("  Max: $(round(maximum(abs.(col_sums)), sigdigits=3))")
    println("  Mean: $(round(mean(abs.(col_sums)), sigdigits=3))")
    
    # Diagonal entries (exit rates)
    println("\nDiagonal entries (exit rates):")
    diag_entries = [A[i,i] for i in 1:min(size(A,1), 10)]
    for (i, val) in enumerate(diag_entries)
        println("  State $i: $(round(val, sigdigits=4))")
    end
    
    # Probability conservation test
    println("\nProbability conservation test:")
    p0 = ones(size(A,1)) / size(A,1)
    for dt in [0.1, 1.0, 10.0]
        p = expv(dt, sparse(A), p0)
        println("  After Δt=$dt: sum(p) = $(round(sum(p), sigdigits=6))")
    end
end

# Make extract_rates available for LearnedGenerator
function extract_rates(
    learned_gen::LearnedGenerator,
    stoich_vecs::Vector{Vector{Int}},
    propensity_fn::PropensityFunction
)
    extract_rates(learned_gen, propensity_fn, stoich_vecs)
end
