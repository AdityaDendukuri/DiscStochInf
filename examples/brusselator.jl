# File: examples/brusselator.jl
"""
Brusselator inverse problem example - Updated for adjoint method.
"""

using Catalyst
using JumpProcesses
using Random
using Plots

include("../src/basis_functions.jl")
include("../src/data_processing.jl")
include("../src/generator_construction.jl")
include("../src/optimization.jl")
include("../src/analysis.jl")

# Set plotting defaults
default(fontfamily="Computer Modern",
        linewidth=1,
        framestyle=:box,
        grid=false)

# Define Brusselator model
brusselator = @reaction_network begin
    A, ∅ --> X
    1, 2X + Y --> 3X
    B, X --> Y
    1, X --> ∅
end

# Parameters and initial condition
p = [:A => 3.0, :B => 4.0]
u0 = [:X => 0, :Y => 0]

println("="^70)
println("BRUSSELATOR CME INVERSE PROBLEM")
println("="^70)
println("\nTrue parameters:")
println("  A = 3.0 (synthesis rate)")
println("  B = 4.0 (conversion rate)")

# Generate trajectories
Random.seed!(123)
dprob = DiscreteProblem(brusselator, u0, (0.0, 50.0), p)
jprob = JumpProblem(brusselator, dprob, Direct())

println("\nGenerating trajectories...")
n_traj = 100
trajectories = [solve(jprob, SSAStepper()) for _ in 1:n_traj]
println("Generated $n_traj trajectories up to t=50")

# Compute histograms
dt = 0.5
println("\nComputing histograms (dt=$dt)...")
hists, transitions = compute_histograms(trajectories, dt, t_max=50.0)
println("Generated $(length(hists)) histogram snapshots")

# Extract stoichiometry
println("\nExtracting stoichiometry...")
stoich_basis_raw = extract_stoichiometry(transitions, min_count=100)

# Filter out null reactions
stoich_basis = filter(nu -> any(nu .!= 0), stoich_basis_raw)

println("Found $(length(stoich_basis)) non-trivial reactions:")
for (i, nu) in enumerate(stoich_basis)
    println("  Reaction $i: $nu")
end

# Use quadratic basis
basis = PolynomialBasis{2, 2}()  # [1, x, y, xy, x², y²]
n_features = get_n_features(basis)
n_params = length(stoich_basis) * n_features

println("\nBasis: Quadratic ($n_features features)")
println("Total parameters: $(length(stoich_basis)) reactions × $n_features = $n_params")

# =============================================================================
# Verify Adjoint Implementation
# =============================================================================

println("\n" * "="^70)
println("VERIFYING ADJOINT IMPLEMENTATION")
println("="^70)

# Build problem for first window
state_space = build_state_space(hists[1], hists[2],
                                method=:adaptive,
                                prob_threshold=1e-4,
                                cumulative_mass=0.99,
                                transitions=transitions)

data = build_inverse_problem_data(state_space, stoich_basis, dt, basis=basis)
window = TimeWindowData(hists[1], hists[2], state_space)

# Verify gradient
θ_test = 0.01 * randn(n_params)
verify_adjoint_gradient(θ_test, data, [window])

# =============================================================================
# Learn Generator for First Window
# =============================================================================

println("\n" * "="^70)
println("LEARNING GENERATOR: Window 1 → 2")
println("="^70)
println("State space size: $(length(state_space))")

if length(state_space) < n_params
    @warn "Underdetermined: $(length(state_space)) states < $n_params parameters"
end

println("\nOptimizing with adjoint method...")
A, θ, converged = learn_generator(data, [window], 
                                   λ1=1e-4,      # Frobenius regularization
                                   λ3=1e-7,      # L1 sparsity
                                   max_iter=1000,
                                   grad_clip=10.0,
                                   method=:adjoint,
                                   show_trace=true)

println("\n" * "="^70)
println("RESULTS")
println("="^70)
println("Converged: $converged")
println("||A||: $(round(norm(A), digits=2))")

# Check generator properties
check_generator_properties(A)

# Compute prediction error
pred_error = compute_prediction_error(A, window, dt)
println("\nPrediction error (L1): $(round(pred_error, digits=4))")

# =============================================================================
# Evaluate Learned Propensities
# =============================================================================

println("\n" * "="^70)
println("LEARNED PROPENSITIES")
println("="^70)

test_states = [[0,0], [1,0], [1,1], [2,1], [3,2], [5,3]]
props = evaluate_propensities(θ, data, test_states)

# Reaction names (based on stoichiometry)
reaction_names = []
for (k, nu) in enumerate(stoich_basis)
    if nu == [1, 0]
        push!(reaction_names, "0→X")
    elseif nu == [1, -1]
        push!(reaction_names, "2X+Y→3X")
    elseif nu == [-1, 1]
        push!(reaction_names, "X→Y")
    elseif nu == [-1, 0]
        push!(reaction_names, "X→0")
    else
        push!(reaction_names, "Reaction $k: $nu")
    end
end

println("\nPropensities at sample states:")
println("(Compare to true: 0→X ≈ A, X→Y ≈ B·X, 2X+Y→3X ≈ X²·Y, X→0 ≈ X)")
println()

for state in test_states
    x, y = state[1], state[2]
    println("State [X=$x, Y=$y]:")
    
    for (k, name) in enumerate(reaction_names)
        learned = props[state][k]
        
        # Compare to true propensity (if known)
        if name == "0→X"
            true_val = 3.0  # A
            println("  $(rpad(name, 12)): $(rpad(round(learned, digits=2), 8)) (true: $true_val)")
        elseif name == "X→Y"
            true_val = 4.0 * x  # B·X
            println("  $(rpad(name, 12)): $(rpad(round(learned, digits=2), 8)) (true: $true_val)")
        elseif name == "2X+Y→3X"
            true_val = x^2 * y  # X²·Y (rate constant = 1)
            println("  $(rpad(name, 12)): $(rpad(round(learned, digits=2), 8)) (true: $true_val)")
        elseif name == "X→0"
            true_val = Float64(x)  # X (rate constant = 1)
            println("  $(rpad(name, 12)): $(rpad(round(learned, digits=2), 8)) (true: $true_val)")
        else
            println("  $(rpad(name, 12)): $(round(learned, digits=2))")
        end
    end
    println()
end

# =============================================================================
# Multi-Window Learning
# =============================================================================

println("="^70)
println("LEARNING GENERATORS FOR MULTIPLE WINDOWS")
println("="^70)

n_windows = 6
generators = []
parameters = []
learned_data = []
state_spaces = []
window_pairs = []

for w in 1:n_windows
    println("\n" * "-"^70)
    println("WINDOW $w → $(w+1)")
    println("-"^70)
    
    # FSP-based state space
    state_space_w = build_state_space(hists[w], hists[w+1],
                                      method=:adaptive,
                                      prob_threshold=1e-4,
                                      cumulative_mass=0.99,
                                      transitions=transitions)
    
    println("State space size: $(length(state_space_w))")
    
    # Build problem
    data_w = build_inverse_problem_data(state_space_w, stoich_basis, dt, basis=basis)
    window_w = TimeWindowData(hists[w], hists[w+1], state_space_w)
    
    # Learn
    println("\nOptimizing...")
    A_w, θ_w, converged_w = learn_generator(data_w, [window_w], 
                                             λ1=1e-4,
                                             λ3=1.0e-7,
                                             max_iter=1000,
                                             grad_clip=10.0,
                                             method=:adjoint,
                                             show_trace=true)
    
    println("\nResults:")
    println("  Converged: $converged_w")
    println("  ||A||: $(round(norm(A_w), digits=2))")
    
    pred_err = compute_prediction_error(A_w, window_w, dt)
    println("  Prediction error: $(round(pred_err, digits=4))")
    
    # Store
    push!(generators, A_w)
    push!(parameters, θ_w)
    push!(learned_data, data_w)
    push!(state_spaces, state_space_w)
    push!(window_pairs, w)
end

# =============================================================================
# Summary
# =============================================================================

println("\n" * "="^70)
println("SUMMARY")
println("="^70)

for (w, A_w, ss) in zip(window_pairs, generators, state_spaces)
    println("\nWindow $w→$(w+1):")
    println("  States: $(length(ss))")
    println("  ||A||: $(round(norm(A_w), digits=2))")
    println("  Nonzeros: $(count(abs.(A_w) .> 1e-8))")
    println("  Sparsity: $(round(100*(1 - count(abs.(A_w) .> 1e-8)/length(A_w)), digits=1))%")
    
    # Column sum check
    col_sums = sum(A_w, dims=1)[:]
    println("  Max |col sum|: $(round(maximum(abs.(col_sums)), digits=10))")
end

println("\n" * "="^70)
println("Done!")
println("="^70)
