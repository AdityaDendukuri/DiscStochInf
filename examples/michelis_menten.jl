# File: examples/michaelis_menten.jl

"""
Michaelis-Menten Inverse Problem using Dictionary Learning (SINDy-CME).
"""

using Catalyst
using JumpProcesses
using Random
using Plots
using LinearAlgebra

default(fontfamily="Computer Modern", linewidth=1, framestyle=:box, grid=false)

include("../src/basis_functions.jl")
include("../src/data_processing.jl")
include("../src/generator_construction.jl")
include("../src/optimization.jl")
include("../src/analysis.jl")
include("../src/visualization.jl")

# --- 1. Ground Truth Simulation ---
mm_model = @reaction_network begin
    kB, S + E --> SE
    kD, SE --> S + E  
    kP, SE --> P + E
end

kB_true, kD_true, kP_true = 0.01, 0.1, 0.1
p = [:kB => kB_true, :kD => kD_true, :kP => kP_true]
S0, E0 = 50, 10
u0 = [:S => S0, :E => E0, :SE => 0, :P => 0]

println("Generating ground truth trajectories...")
dprob = DiscreteProblem(mm_model, u0, (0.0, 200.0), p)
jprob = JumpProblem(mm_model, dprob, Direct())
trajectories = [solve(jprob, SSAStepper()) for _ in 1:500]

dt = 1.0 
hists, transitions = compute_histograms(trajectories, dt, t_max=200.0)

# --- 2. DICTIONARY SETUP (Structure Identification) ---
println("\n" * "="^60)
println("SETTING UP DICTIONARY LEARNING")
println("="^60)

# We have 4 species: S, E, SE, P
n_species = 4

# A. Generate Candidate Reactions (The "Vague" part)
# This includes S->S+1, S+E->SE, 2S->P, etc.
candidate_library = generate_candidate_library(n_species, max_jump=1)

# Add specific complex ones if not covered (S+E -> SE is [-1, -1, 1, 0])
# The generator covers this, but let's ensure S+E->SE is in there.
# Indices: 1:S, 2:E, 3:SE, 4:P
# S(-1), E(-1), SE(+1)
push!(candidate_library, [-1, -1, 1, 0]) 
unique!(candidate_library)

println("Candidate Library Size: $(length(candidate_library)) reactions")

# B. Define Physics-Informed Basis
# Use Mass Action (x, x*y, x(x-1)/2) instead of polynomials
basis = MassActionBasis(n_species, max_order=2)
n_features = get_n_features(basis)

println("Basis Features: $(get_feature_names(basis))")
println("Total Parameters: $(length(candidate_library) * n_features)")


# --- 3. OPTIMIZATION LOOP ---
n_windows = 6
generators = []
window_pairs = []

for w in 1:n_windows
    println("\nWindow $w → $(w+1)")
    
    state_space = build_state_space_fsp(hists[w], hists[w+1])
    
    # Build data with CANDIDATE library
    data = build_inverse_problem_data(state_space, candidate_library, dt, basis=basis)
    window = TimeWindowData(hists[w], hists[w+1], state_space)
    
    # Learn
    # High λ3 enforces sparsity (selects correct reactions)
    A, θ, converged = learn_generator(data, [window], 
                                       λ1=1e-6,      # Smoothness
                                       λ3=1.0e-6,      # L1 SPARSITY (Tunable)
                                       max_iter=1000,
                                       method=:adjoint)

    # --- 4. INTERPRET RESULTS ---
    println("\nInferred Mechanism (Window $w):")
    feature_names = get_feature_names(basis)
    
    for k in 1:length(candidate_library)
        # Check total magnitude of this reaction
        idx_start = (k-1)*n_features + 1
        idx_end = k*n_features
        θ_k = θ[idx_start:idx_end]
        
        if sum(abs.(θ_k)) > 1e-3
            reaction_vec = candidate_library[k]
            print("  Reaction $k $reaction_vec: Rate = ")
            
            terms = []
            for (f, coeff) in enumerate(θ_k)
                if abs(coeff) > 1e-4
                    push!(terms, "$(round(coeff, digits=4))*[$(feature_names[f])]")
                end
            end
            println(join(terms, " + "))
        end
    end
    
    push!(generators, A)
    push!(window_pairs, w)
end
