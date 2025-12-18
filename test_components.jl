"""
Test script for debugging individual components of the inverse problem framework.

This script allows you to test each component separately without running
the full experiment.
"""

push!(LOAD_PATH, @__DIR__)

using Test
using LinearAlgebra

include("types.jl")
include("data_generation.jl")
include("state_space.jl")

println("="^60)
println("COMPONENT TESTING")
println("="^60)

# ============================================================================
# TEST 1: Configuration
# ============================================================================

println("\n--- TEST 1: Configuration ---")

config = InverseProblemConfig()
println(config)

@test config.mass_threshold > 0 && config.mass_threshold <= 1
@test config.λ_frobenius >= 0
@test config.dt_snapshot > 0
println("✓ Configuration tests passed")

# ============================================================================
# TEST 2: Histogram Merging
# ============================================================================

println("\n--- TEST 2: Histogram Merging ---")

A1 = Dict([1,2,3] => 0.5, [2,3,4] => 0.3)
A2 = Dict([1,2,3] => 0.2, [3,4,5] => 0.4)

merged = merge_hist(A1, A2)
println("Merged histogram: $merged")

@test merged[[1,2,3]] ≈ 0.7
@test merged[[2,3,4]] ≈ 0.3
@test merged[[3,4,5]] ≈ 0.4
println("✓ Histogram merging tests passed")

# ============================================================================
# TEST 3: State Space Building
# ============================================================================

println("\n--- TEST 3: State Space Building ---")

# Create mock distributions
test_dist1 = Dict(
    (1,1,0,0) => 0.4,
    (2,1,0,0) => 0.3,
    (1,2,0,0) => 0.2,
    (0,1,1,0) => 0.1
)

test_dist2 = Dict(
    (1,1,0,0) => 0.3,
    (2,1,0,0) => 0.4,
    (1,2,0,0) => 0.2,
    (0,1,1,0) => 0.1
)

test_dists = [test_dist1, test_dist2]
test_stoich = [[-1, -1, 1, 0], [1, 1, -1, 0], [1, 0, -1, 1]]

X_local = build_local_state_space_with_connectivity(test_dists, test_stoich, 0.8)

println("Built local state space with $(length(X_local)) states")
println("Sample states: $(collect(X_local)[1:min(3, length(X_local))])")

@test length(X_local) > 0
@test all(s isa Tuple for s in X_local)
println("✓ State space building tests passed")

# ============================================================================
# TEST 4: Reaction Direction Matrices
# ============================================================================

println("\n--- TEST 4: Reaction Direction Matrices ---")

# Convert state space to CartesianIndex
X_cart = Set(CartesianIndex(s...) for s in X_local)
X_vec = sort(collect(X_cart))

# Build E matrices
E_matrices = [reaction_direction(ν, X_cart) for ν in test_stoich]

println("Built $(length(E_matrices)) E matrices")
for (j, E) in enumerate(E_matrices)
    nz = count(abs.(E) .> 1e-10)
    println("  E$j: $nz non-zero entries")
end

@test length(E_matrices) == length(test_stoich)
@test all(size(E) == (length(X_vec), length(X_vec)) for E in E_matrices)
println("✓ Reaction direction tests passed")

# ============================================================================
# TEST 5: Sparsity Pattern
# ============================================================================

println("\n--- TEST 5: Sparsity Pattern ---")

N_states = length(X_vec)
sparsity, nz_off_diag = build_sparsity_pattern(E_matrices, N_states)

println("Sparsity pattern:")
println("  Total entries: $(N_states^2)")
println("  Nonzero pattern: $(count(sparsity))")
println("  Off-diagonal params: $(length(nz_off_diag))")

@test count(sparsity) > 0
@test length(nz_off_diag) > 0
@test all(idx[1] != idx[2] for idx in nz_off_diag)
println("✓ Sparsity pattern tests passed")

# ============================================================================
# TEST 6: Extract Local Distributions
# ============================================================================

println("\n--- TEST 6: Extract Local Distributions ---")

# Pad test distributions
padded = []
for dist in test_dists
    d = Dict()
    for state in X_local
        d[collect(state)] = get(dist, state, 0.0)
    end
    push!(padded, d)
end

local_dists = extract_local_distributions(padded, X_vec)

println("Extracted $(length(local_dists)) local distributions")
println("  Each has $(length(local_dists[1])) entries")
println("  First distribution sum: $(sum(local_dists[1]))")

@test length(local_dists) == length(test_dists)
@test all(length(d) == length(X_vec) for d in local_dists)
@test all(sum(d) ≈ 1.0 for d in local_dists)
println("✓ Local distribution extraction tests passed")

# ============================================================================
# TEST 7: Matrix Conversion
# ============================================================================

println("\n--- TEST 7: Matrix Conversion ---")

include("optimization.jl")

# Create a test parameter vector
v_test = randn(length(nz_off_diag))
A_test = vec_to_matrix(v_test, nz_off_diag, N_states)

println("Matrix shape: $(size(A_test))")
println("Column sums: $(vec(sum(A_test, dims=1)))")

# Check zero column sums
col_sums = vec(sum(A_test, dims=1))
@test maximum(abs.(col_sums)) < 1e-10

# Check round-trip conversion
v_recovered = matrix_to_vec(A_test, nz_off_diag)
@test v_recovered ≈ v_test

println("✓ Matrix conversion tests passed")

# ============================================================================
# TEST 8: Propensity Function
# ============================================================================

println("\n--- TEST 8: Propensity Function ---")

include("analysis.jl")

struct TestPropensity <: PropensityFunction end

function (prop::TestPropensity)(state::Tuple, reaction_idx::Int)
    if reaction_idx == 1
        return float(state[1] * state[2])
    else
        return 1.0
    end
end

prop = TestPropensity()
test_state = (5, 3, 1, 0)

p1 = prop(test_state, 1)
p2 = prop(test_state, 2)

println("Propensity for state $test_state:")
println("  Reaction 1: $p1")
println("  Reaction 2: $p2")

@test p1 == 15.0  # 5 * 3
@test p2 == 1.0
println("✓ Propensity function tests passed")

# ============================================================================
# SUMMARY
# ============================================================================

println("\n" * "="^60)
println("ALL TESTS PASSED ✓")
println("="^60)
println("\nComponents are working correctly!")
println("You can now run main_experiment.jl for the full workflow.")
