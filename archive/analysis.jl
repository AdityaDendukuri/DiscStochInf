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
    extract_rates(learned_gen, propensity_fn, stoich_vecs)

Extract rates from a single learned generator.
"""
function extract_rates(
    learned_gen::LearnedGenerator,
    propensity_fn::PropensityFunction,
    stoich_vecs::Vector{Vector{Int}}
)
    A = learned_gen.A
    X = learned_gen.state_space
    X_set = Set(X)
    
    rate_stats = Dict{Int, NamedTuple}()
    
    for (j, ν) in enumerate(stoich_vecs)
        rates_found = Float64[]
        
        # Only iterate over states IN the learned generator
        for (idx_from, state_from) in enumerate(X)
            # Compute target state
            state_to_tuple = tuple((collect(Tuple(state_from)) .+ ν)...)
            state_to = CartesianIndex(state_to_tuple)
            
            # Both states must be in learned space
            if state_to ∈ X_set
                idx_to = findfirst(==(state_to), X)
                
                if idx_to !== nothing
                    rate_entry = A[idx_to, idx_from]
                    
                    if abs(rate_entry) > 1e-10
                        propensity = propensity_fn(Tuple(state_from), j)
                        
                        if propensity > 1e-10
                            k_estimate = rate_entry / propensity
                            
                            # TIGHTER OUTLIER FILTER
                            if k_estimate > 0 && k_estimate < 1.0
                                push!(rates_found, k_estimate)
                            end
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
    extract_rates_aggregated(learned_generators, propensity_fn, stoich_vecs)

Pool transitions from ALL windows to get better statistics.
"""
function extract_rates_aggregated(
    learned_generators::Vector{LearnedGenerator},
    propensity_fn::PropensityFunction,
    stoich_vecs::Vector{Vector{Int}}
)
    rate_stats = Dict{Int, NamedTuple}()
    
    for (j, ν) in enumerate(stoich_vecs)
        rates_found = Float64[]
        
        # Aggregate across ALL learned generators
        for learned_gen in learned_generators
            A = learned_gen.A
            X = learned_gen.state_space
            X_set = Set(X)
            
            for (idx_from, state_from) in enumerate(X)
                state_to_tuple = tuple((collect(Tuple(state_from)) .+ ν)...)
                state_to = CartesianIndex(state_to_tuple)
                
                if state_to ∈ X_set
                    idx_to = findfirst(==(state_to), X)
                    
                    if idx_to !== nothing
                        rate_entry = A[idx_to, idx_from]
                        
                        if abs(rate_entry) > 1e-10
                            propensity = propensity_fn(Tuple(state_from), j)
                            
                            if propensity > 1e-10
                                k_estimate = rate_entry / propensity
                                
                                # TIGHTER OUTLIER FILTER
                                if k_estimate > 0 && k_estimate < 1.0
                                    push!(rates_found, k_estimate)
                                end
                            end
                        end
                    end
                end
            end
        end
        
        # Statistics from pooled data
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
    print_rate_comparison(result, true_rates, propensity_fn, global_state_space)

Print comparison of learned vs true rates.
"""
function print_rate_comparison(
    result::OptimizationResult,
    true_rates::Vector{Float64},
    propensity_fn::PropensityFunction,
    global_state_space=nothing  # Ignored
)
    println("\n" * "="^70)
    println("AGGREGATED RATE EXTRACTION (All Windows Combined)")
    println("="^70)
    
    # Use aggregated extraction
    aggregated_stats = extract_rates_aggregated(
        result.learned_generators, 
        propensity_fn, 
        result.inferred_stoich
    )
    
    println("\nReaction | True Rate | Learned (median) | Rel. Error | Total Transitions")
    println("-"^75)
    for (j, ν) in enumerate(result.inferred_stoich)
        stats = aggregated_stats[j]
        true_k = true_rates[j]
        
        if stats.n_transitions > 0
            rel_error = abs(stats.median - true_k) / true_k * 100
            println("   R$j    |   $(round(true_k, digits=3))    |      $(round(stats.median, digits=4))      |    $(round(rel_error, digits=1))%   |     $(stats.n_transitions)")
            println("         |           |   (std=$(round(stats.std, digits=4)))    |           |")
        else
            println("   R$j    |   $(round(true_k, digits=3))    |      N/A       |    N/A   |     0")
        end
    end
    
    println("\n" * "="^70)
    println("RATE COMPARISON: LEARNED vs TRUE")
    println("="^70)
    
    println("\nTrue rates: $true_rates")
    println("True stoichiometry:")
    for (j, ν) in enumerate(result.inferred_stoich)
        println("  R$j: $ν  (k=$(true_rates[j]))")
    end
    
    println("\nExtraction method: LOCAL (from learned generators only)")
    
    println("\n" * "-"^60)
    println("Learned rates over time:")
    println("-"^60)
    
    for learned_gen in result.learned_generators
        rate_stats = extract_rates(learned_gen, propensity_fn, result.inferred_stoich)
        
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
    
    # Final window summary
    if !isempty(result.learned_generators)
        println("\n" * "="^60)
        println("FINAL WINDOW SUMMARY")
        println("="^60)
        
        final_gen = result.learned_generators[end]
        final_stats = extract_rates(final_gen, propensity_fn, result.inferred_stoich)
        
        println("\nReaction | True Rate | Learned (median) | Rel. Error | Transitions")
        println("-"^70)
        for (j, ν) in enumerate(result.inferred_stoich)
            stats = final_stats[j]
            true_k = true_rates[j]
            
            if stats.n_transitions > 0
                rel_error = abs(stats.median - true_k) / true_k * 100
                println("   R$j    |   $(round(true_k, digits=3))    |      $(round(stats.median, digits=3))       |    $(round(rel_error, digits=1))%   |     $(stats.n_transitions)")
            else
                println("   R$j    |   $(round(true_k, digits=3))    |      N/A       |    N/A   |     0")
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
    
    # Column sum errors
    col_sums = vec(sum(A, dims=1))
    println("\nColumn sum errors:")
    println("  Max: $(round(maximum(abs.(col_sums)), sigdigits=3))")
    println("  Mean: $(round(mean(abs.(col_sums)), sigdigits=3))")
    
    # Diagonal entries
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
