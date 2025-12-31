# File: examples/brusselator.jl

"""
Brusselator inverse problem example.
"""

using Catalyst
using JumpProcesses
using Random

include("../src/data_processing.jl")
include("../src/generator_construction.jl")
include("../src/optimization.jl")
include("../src/analysis.jl")

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

# Generate trajectories
Random.seed!(123)
dprob = DiscreteProblem(brusselator, u0, (0.0, 50.0), p)
jprob = JumpProblem(brusselator, dprob, Direct())

println("Generating trajectories...")
n_traj = 100
trajectories = [solve(jprob, SSAStepper()) for _ in 1:n_traj]

# Compute histograms
dt = 1.0
println("\nComputing histograms (dt=$dt)...")
hists, transitions = compute_histograms(trajectories, dt, t_max=50.0)

# Extract stoichiometry
println("\nExtracting stoichiometry...")
stoich_basis = extract_stoichiometry(transitions, min_count=100)
println("Found $(length(stoich_basis)) reactions:")
for (i, nu) in enumerate(stoich_basis)
    println("  Reaction $i: $nu")
end

# Learn generator for first window
println("\n" * "="^70)
println("Learning generator for window 1->2")
println("="^70)

state_space = build_state_space(hists[1], hists[2])
println("State space size: $(length(state_space))")

data = build_inverse_problem_data(state_space, stoich_basis, dt, n_features=6)
window = TimeWindowData(hists[1], hists[2], state_space)

A, θ, converged = learn_generator(data, [window], 
                                   λ1=1e-6, λ2=1e-6, 
                                   max_iter=200, show_trace=true)

println("\nResults:")
println("  Converged: $converged")
check_generator_properties(A)

# Evaluate propensities
test_states = [[0,0], [1,0], [1,1], [2,1], [3,2]]
props = evaluate_propensities(θ, data, test_states)

println("\nPropensities at sample states:")
reaction_names = ["X→Y", "2X+Y→3X", "0→X", "X→0"]
for state in test_states
    println("\nState $state:")
    for (k, name) in enumerate(reaction_names)
        println("  $name: $(round(props[state][k], digits=2))")
    end
end

# Visualizations
println("\nGenerating plots...")
p1 = plot_generator(A, state_space, title="Window 1→2 ($(length(state_space)) states)")
display(p1)

plot_histograms(hists, [1, 2, 3, 4], layout=(2,2))
