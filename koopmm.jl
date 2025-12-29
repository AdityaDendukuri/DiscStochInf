"""
Hybrid FSP-Koopman: Final working version
"""

using LinearAlgebra
using SparseArrays
using ExponentialUtilities
using NLopt
using Catalyst
using Statistics

include("koopman_optimization.jl")

# ============================================================================
# GENERATE EXACT MOMENTS VIA FSP
# ============================================================================

function generate_exact_moments_via_fsp(
    rn,
    u0,
    true_rates,
    tspan;
    dt_snapshot=1.0,
    polynomial_degree=2
)
    println("="^80)
    println("HYBRID FSP-KOOPMAN: Exact moments via FSP")
    println("="^80)
    
    # Get reactions and species
    rxns = Catalyst.get_rxs(rn)
    specs = Catalyst.get_species(rn)
    m = length(specs)
    n_rxns = length(rxns)
    
    println("System: $m species, $n_rxns reactions")
    
    # Extract stoichiometry
    stoichiometries = []
    for rxn in rxns
        stoich = zeros(Int, m)
        
        for (spec, coeff) in zip(rxn.substrates, rxn.substoich)
            spec_idx = findfirst(s -> isequal(s, spec), specs)
            stoich[spec_idx] -= Int(coeff)
        end
        
        for (spec, coeff) in zip(rxn.products, rxn.prodstoich)
            spec_idx = findfirst(s -> isequal(s, spec), specs)
            stoich[spec_idx] += Int(coeff)
        end
        
        push!(stoichiometries, stoich)
    end
    
    # Get reaction orders
    orders = []
    for rxn in rxns
        order = zeros(Int, m)
        for (spec, coeff) in zip(rxn.substrates, rxn.substoich)
            spec_idx = findfirst(s -> isequal(s, spec), specs)
            order[spec_idx] += Int(coeff)
        end
        push!(orders, order)
    end
    
    # Initial state
    u0_vals = [val for (sym, val) in u0]
    
    # Build state space
    println("\nExpanding state space...")
    X_cart = Set{CartesianIndex{m}}()
    push!(X_cart, CartesianIndex(u0_vals...))
    
    for expansion in 1:10
        X_new = copy(X_cart)
        
        for state in X_cart
            state_vec = [state[i] for i in 1:m]
            
            for stoich in stoichiometries
                new_state = state_vec + stoich
                if all(new_state .>= 0) && all(new_state .<= 100)
                    push!(X_new, CartesianIndex(new_state...))
                end
                
                new_state = state_vec - stoich
                if all(new_state .>= 0) && all(new_state .<= 100)
                    push!(X_new, CartesianIndex(new_state...))
                end
            end
        end
        
        if length(X_new) == length(X_cart)
            break
        end
        X_cart = X_new
    end
    
    X_vec = sort(collect(X_cart))
    N_states = length(X_vec)
    println("  State space: $N_states states")
    
    # Build CME generator
    println("\nBuilding CME generator...")
    A = zeros(N_states, N_states)
    
    for (i, state_i) in enumerate(X_vec)
        state_vec = [state_i[k] for k in 1:m]
        
        for (rxn_idx, (stoich, order)) in enumerate(zip(stoichiometries, orders))
            # Mass action propensity
            prop = true_rates[rxn_idx]
            for s in 1:m
                for k in 1:order[s]
                    prop *= (state_vec[s] - k + 1)
                end
            end
            
            # Target state
            target_vec = state_vec + stoich
            
            if all(target_vec .>= 0)
                target_ci = CartesianIndex(target_vec...)
                j = findfirst(==(target_ci), X_vec)
                
                if !isnothing(j)
                    A[j, i] += prop
                    A[i, i] -= prop
                end
            end
        end
    end
    
    println("  Matrix: $(N_states)×$(N_states)")
    
    # Polynomial basis
    basis_size, eval_basis, basis_description, get_monomial_index = 
        generate_polynomial_basis(m, polynomial_degree)
    
    println("\nPolynomial basis: $basis_size functions")
    
    # Initial distribution
    p0 = zeros(N_states)
    u0_ci = CartesianIndex(u0_vals...)
    initial_idx = findfirst(==(u0_ci), X_vec)
    p0[initial_idx] = 1.0
    
    # Time grid
    times = collect(tspan[1]:dt_snapshot:tspan[2])
    n_times = length(times)
    
    # Solve FSP
    println("\nSolving FSP over $(n_times) time points...")
    moment_data = Vector{Vector{Float64}}(undef, n_times)
    
    p_current = copy(p0)
    
    for (i, t) in enumerate(times)
        if i > 1
            dt = times[i] - times[i-1]
            expA = exp(dt * A)
            p_current = expA * p_current
        end
        
        # Extract moments
        g = zeros(basis_size)
        for (j, state_ci) in enumerate(X_vec)
            state = Float64.([state_ci[k] for k in 1:m])
            g .+= p_current[j] .* eval_basis(state)
        end
        
        moment_data[i] = g
        
        if i % 10 == 0 || i == n_times
            mass = sum(p_current)
            println("  t=$(round(t, digits=1)): mass=$(round(mass, digits=6))")
        end
    end
    
    println("✓ Generated exact moments")
    
    return moment_data, times, basis_size, eval_basis
end

# ============================================================================
# LEARN RATES
# ============================================================================

function learn_rates_from_exact_moments(
    moment_data,
    times,
    stoich_vecs,
    orders,
    eval_basis,
    m,
    d;
    λ_reg=1e-3
)
    println("\n" * "="^80)
    println("LEARNING RATES FROM EXACT MOMENTS")
    println("="^80)
    
    n_reactions = length(stoich_vecs)
    
    # Build operators
    println("\nBuilding operators...")
    
    # Fix: Create Vector{Vector{Float64}} explicitly
    states_grid = Vector{Vector{Float64}}()
    for s1 in 0:10:50, s2 in 0:2:10, s3 in 0:1:5, s4 in 0:10:50
        push!(states_grid, [Float64(s1), Float64(s2), Float64(s3), Float64(s4)])
    end
    
    println("  Grid: $(length(states_grid)) points")
    
    B_matrices = Vector{Matrix{Float64}}()
    for j in 1:n_reactions
        B_j = build_reaction_operator(
            stoich_vecs[j], orders[j], eval_basis, m, d;
            states_data=states_grid, λ_ridge=100.0
        )
        push!(B_matrices, B_j)
        println("  B_$j: norm=$(round(norm(B_j), sigdigits=3))")
    end
    
    # Scales
    scales = zeros(d)
    for g in moment_data
        scales .= max.(scales, abs.(g))
    end
    scales = max.(scales, 1.0)
    
    println("\nScales: $(round.(scales[1:min(5,d)], sigdigits=3))")
    
    # Optimize
    println("\nOptimizing...")
    θ, info = optimize_koopman_generator(
        B_matrices, moment_data, times, scales;
        λ_reg=λ_reg, θ_init=ones(n_reactions)*0.05,
        θ_lower=1e-5, θ_upper=1.0, maxeval=2000
    )
    
    return θ, info
end

# ============================================================================
# RUN
# ============================================================================

function run_hybrid()
    rn = @reaction_network begin
        k1, S + E --> SE
        k2, SE --> S + E
        k3, SE --> P + E
    end
    
    u0 = [:S => 50, :E => 10, :SE => 0, :P => 0]
    true_rates = [0.01, 0.1, 0.1]
    
    # FSP: EXACT moments (no SSA noise!)
    moments, times, d, basis = generate_exact_moments_via_fsp(
        rn, u0, true_rates, (0.0, 30.0); dt_snapshot=0.5, polynomial_degree=2
    )
    
    # Learn from EXACT moments
    θ, _ = learn_rates_from_exact_moments(
        moments, times,
        [[-1,-1,1,0], [1,1,-1,0], [0,1,-1,1]],
        [[1,1,0,0], [0,0,1,0], [0,0,1,0]],
        basis, 4, d; λ_reg=1e-3
    )
    
    # Results
    println("\n" * "="^80)
    println("HYBRID FSP-KOOPMAN RESULTS")
    println("="^80)
    println("\nRxn | True  | Found | Error")
    println("-"^35)
    
    errors = []
    for j in 1:3
        err = abs(θ[j] - true_rates[j])/true_rates[j]*100
        push!(errors, err)
        println("R$j  | $(round(true_rates[j],digits=3)) | $(round(θ[j],digits=4)) | $(round(err,digits=1))%")
    end
    
    println("\nSummary:")
    println("  Mean error: $(round(mean(errors), digits=1))%")
    println("  Best: $(round(minimum(errors), digits=1))%")
    println("  Worst: $(round(maximum(errors), digits=1))%")
    
    return θ
end

θ = run_hybrid()
