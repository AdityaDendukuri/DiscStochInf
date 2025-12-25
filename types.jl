"""
Core data structures for the inverse problem framework
"""

using LinearAlgebra

"""
    InverseProblemConfig

Configuration for inverse problem experiments.

# Fields
- `mass_threshold::Float64`: Fraction of probability mass to keep in local state space (default: 0.95)
- `λ_frobenius::Float64`: Frobenius norm regularization weight (default: 1e-6)
- `λ_prob_conservation::Float64`: Probability conservation penalty weight (default: 1e-6)
- `dt_snapshot::Float64`: Time interval between probability snapshots (default: 0.1)
- `dt_window::Float64`: Duration of each sliding window (default: 2.0)
- `snapshots_per_window::Int`: Number of snapshots in each window (default: 10)
- `max_windows::Int`: Maximum number of windows to process (default: 10)
"""
struct InverseProblemConfig
    mass_threshold::Float64
    λ_frobenius::Float64
    λ_prob_conservation::Float64
    dt_snapshot::Float64
    dt_window::Float64
    snapshots_per_window::Int
    max_windows::Int
    
    function InverseProblemConfig(;
        mass_threshold=0.95,
        λ_frobenius=1e-6,
        λ_prob_conservation=1e-6,
        dt_snapshot=0.1,
        dt_window=2.0,
        snapshots_per_window=10,
        max_windows=10
    )
        new(mass_threshold, λ_frobenius, λ_prob_conservation, 
            dt_snapshot, dt_window, snapshots_per_window, max_windows)
    end
end

"""
    WindowData

Data for a single time window in the sliding window approach.

# Fields
- `window_idx::Int`: Index of this window
- `distributions::Vector{Dict}`: Probability distributions at each snapshot
- `times::Vector{Float64}`: Time points for each snapshot
- `stoich_vecs::Vector{Vector{Int}}`: Stoichiometry vectors for reactions
"""
struct WindowData
    window_idx::Int
    distributions::Vector{Dict}
    times::Vector{Float64}
    stoich_vecs::Vector{Vector{Int}}
end

"""
    LearnedGenerator

Results from learning a generator matrix in one window.

# Fields
- `t_start::Float64`: Start time of the window
- `A::Matrix{Float64}`: Learned generator matrix
- `state_space::Vector`: Local state space (CartesianIndex or tuples)
- `convergence_info::NamedTuple`: Optimization convergence information
"""
struct LearnedGenerator
    t_start::Float64
    A::Matrix{Float64}
    state_space::Vector
    convergence_info::NamedTuple
end

"""
    OptimizationResult

Complete results from the inverse problem optimization.

# Fields
- `learned_generators::Vector{LearnedGenerator}`: Generators learned at each window
- `inferred_stoich::Vector{Vector{Int}}`: Inferred stoichiometry
- `config::InverseProblemConfig`: Configuration used
"""
struct OptimizationResult
    learned_generators::Vector{LearnedGenerator}
    inferred_stoich::Vector{Vector{Int}}
    config::InverseProblemConfig
end

"""
    extract_rates(learned_gen::LearnedGenerator, propensity_fn)

Extract reaction rates from a learned generator by dividing out propensities.

Returns a dictionary mapping reaction index to statistics (mean, median, std, etc.)
"""
function extract_rates end  # Implementation in analysis.jl

# Pretty printing
function Base.show(io::IO, config::InverseProblemConfig)
    println(io, "InverseProblemConfig:")
    println(io, "  mass_threshold = $(config.mass_threshold)")
    println(io, "  λ_frobenius = $(config.λ_frobenius)")
    println(io, "  λ_prob_conservation = $(config.λ_prob_conservation)")
    println(io, "  dt_snapshot = $(config.dt_snapshot)")
    println(io, "  dt_window = $(config.dt_window)")
    println(io, "  snapshots_per_window = $(config.snapshots_per_window)")
    print(io, "  max_windows = $(config.max_windows)")
end

function Base.show(io::IO, result::OptimizationResult)
    println(io, "OptimizationResult:")
    println(io, "  $(length(result.learned_generators)) windows")
    println(io, "  $(length(result.inferred_stoich)) reactions")
    print(io, "  Time range: [$(result.learned_generators[1].t_start), $(result.learned_generators[end].t_start)]")
end

"""
    ExperimentResult

Complete results from an inverse CME experiment including all data and metadata.

# Fields
- `reaction_network`: Original Catalyst reaction network
- `true_rates::Vector{Float64}`: True rate constants (for validation)
- `learned_generators::Vector{LearnedGenerator}`: Learned generators (one per window)
- `inferred_stoich::Vector{Vector{Int}}`: Inferred stoichiometry vectors (in Catalyst order)
- `stoich_permutation::Vector{Int}`: Permutation from raw inferred order to Catalyst order
- `global_state_space::Set`: Global state space used for rate extraction
- `config::InverseProblemConfig`: Configuration parameters
- `trajectories::Vector`: Raw SSA trajectories
- `distributions::Tuple`: (times, distributions) from histogram conversion
- `windows::Vector{WindowData}`: Window data structures
"""
struct ExperimentResult
    reaction_network  # ReactionSystem (avoid type dependency)
    true_rates::Vector{Float64}
    learned_generators::Vector{LearnedGenerator}
    inferred_stoich::Vector{Vector{Int}}
    stoich_permutation::Vector{Int}
    global_state_space::Set
    config::InverseProblemConfig
    trajectories::Vector
    distributions::Tuple
    windows::Vector{WindowData}
end
