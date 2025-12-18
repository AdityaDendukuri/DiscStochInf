# Inverse Problem Framework for CME Generator Learning

This directory contains a refactored, modular implementation of the inverse problem approach for learning Chemical Master Equation (CME) generators from trajectory data.

## Overview

The framework learns generator matrices (Q-matrices) of continuous-time Markov chains by:
1. Observing probability distributions at discrete time snapshots
2. Using sliding windows to handle time-varying systems
3. Employing Fréchet derivatives for efficient gradient computation
4. Enforcing physical constraints (zero column sums, probability conservation)

## File Structure

```
inverse_problem/
├── types.jl              # Core data structures
├── data_generation.jl    # SSA trajectory generation and distribution utilities
├── state_space.jl        # Local state space construction with connectivity
├── optimization.jl       # Objective functions and NLopt optimization
├── analysis.jl          # Rate extraction and comparison tools
├── main_experiment.jl   # Complete example workflow
└── README.md           # This file
```

## Module Descriptions

### `types.jl`
Core data structures:
- `InverseProblemConfig`: Configuration parameters for experiments
- `WindowData`: Data container for single time windows
- `LearnedGenerator`: Results from optimizing one window
- `OptimizationResult`: Complete experiment results

### `data_generation.jl`
Functions for data generation and preprocessing:
- `generate_ssa_data()`: Generate SSA trajectories
- `convert_to_distributions()`: Convert trajectories to probability distributions
- `infer_reactions_from_trajectories()`: Infer stoichiometry from observed transitions
- `pad_distributions()`: Ensure all states are represented in distributions
- `create_windows()`: Create sliding window data structures

### `state_space.jl`
State space construction utilities:
- `build_local_state_space_with_connectivity()`: Build connected local state spaces
- `reaction_direction()`: Construct reaction direction matrices (E matrices)
- `build_sparsity_pattern()`: Determine which generator entries can be non-zero
- `extract_local_distributions()`: Project distributions onto local state space
- `verify_connectivity()`: Check that reactions have valid transitions

### `optimization.jl`
Optimization routines:
- `optimize_local_generator()`: Main optimization function for one window
- `objective_with_gradient!()`: Three-term objective (data fit + regularization + probability conservation)
- `vec_to_matrix()`, `matrix_to_vec()`: Convert between parameter vectors and matrices

The objective function includes:
1. **Data fitting**: Huber loss between predicted and observed distributions
2. **Frobenius regularization**: Prevents overfitting
3. **Probability conservation**: Penalizes violations of ∑p = 1

### `analysis.jl`
Analysis and rate extraction:
- `extract_rates()`: Extract reaction rates by dividing out propensities
- `print_rate_comparison()`: Compare learned vs true rates
- `analyze_generator_properties()`: Analyze mathematical properties of learned matrices
- `PropensityFunction`: Abstract type for defining system-specific propensities

## Quick Start

### Basic Usage

```julia
# Load the framework
include("types.jl")
include("data_generation.jl")
include("state_space.jl")
include("optimization.jl")
include("analysis.jl")

# Configure experiment
config = InverseProblemConfig(
    mass_threshold = 0.95,
    λ_frobenius = 1e-6,
    λ_prob_conservation = 1e-6,
    dt_snapshot = 0.1,
    dt_window = 2.0
)

# Generate data (using your own SSA trajectories)
T, distrs = convert_to_distributions(ssa_trajs, (0.0, 150.0), 0.1)

# Infer stoichiometry
stoich_vecs = infer_reactions_from_trajectories(ssa_trajs)

# Create windows and optimize
windows = create_windows(T, padded_dists, config, stoich_vecs)
learned_generators = []

for window_data in windows
    A, X_local, conv_info = optimize_local_generator(window_data, config)
    push!(learned_generators, LearnedGenerator(window_data.times[1], A, X_local, conv_info))
end

# Extract rates
propensity_fn = YourPropensityFunction()  # Define your system's propensities
result = OptimizationResult(learned_generators, stoich_vecs, config)
print_rate_comparison(result, true_rates, propensity_fn)
```

### Running the Example

The complete example workflow is in `main_experiment.jl`:

```bash
julia main_experiment.jl
```

This runs the full pipeline on the Michaelis-Menten enzyme kinetics system:
- S + E → SE  (k₁ = 0.01)
- SE → S + E  (k₂ = 0.1)
- SE → P + E  (k₃ = 0.1)

## Key Features

### 1. Modular Design
Each component is independent and can be used/tested separately:
```julia
# Just generate data
T, distrs = convert_to_distributions(trajs, tspan, dt)

# Just build state space
X_local = build_local_state_space_with_connectivity(dists, stoich, 0.95)

# Just optimize (given prepared data)
A, X, info = optimize_local_generator(window_data, config)
```

### 2. Flexible Configuration
All parameters are centralized in `InverseProblemConfig`:
```julia
config = InverseProblemConfig(
    mass_threshold = 0.99,           # More states
    λ_frobenius = 1e-8,             # Less regularization
    dt_window = 1.0,                 # Shorter windows
    snapshots_per_window = 20        # More snapshots per window
)
```

### 3. Extensible Propensity Functions
Define custom propensities for your system:
```julia
struct MyPropensity <: PropensityFunction end

function (prop::MyPropensity)(state::Tuple, reaction_idx::Int)
    # Compute propensity for your specific reactions
    # ...
    return propensity_value
end
```

### 4. Comprehensive Analysis
Built-in tools for:
- Rate extraction with statistical summaries
- Generator property validation
- Convergence diagnostics
- Time-evolution of learned rates

## Debugging Tips

### 1. Check State Space Connectivity
```julia
X_local = build_local_state_space_with_connectivity(dists, stoich, 0.95)
E_matrices = [reaction_direction(ν, X_local) for ν in stoich]
verify_connectivity(E_matrices)  # Should show non-zero transitions
```

### 2. Monitor Objective Value
```julia
# The optimization prints:
#   Initial objective: ...
#   Return: :SUCCESS, Objective: ...
# If initial objective is NaN/Inf, check data preprocessing
```

### 3. Validate Learned Generator
```julia
analyze_generator_properties(learned_gen)
# Check:
#   - Column sums should be ~0
#   - Probability conservation should be ~1.0
#   - Sparsity should match expected pattern
```

### 4. Examine Distribution Differences
The code prints distribution differences between consecutive snapshots:
```
Distribution differences:
  Step 1->2: 0.023
  Step 2->3: 0.019
```
Very small differences (<1e-4) suggest the system isn't evolving much.

## Tuning Guide

### When rates are too high/low:
- Check propensity function implementation
- Verify stoichiometry inference
- Ensure correct state ordering

### When optimization fails to converge:
- Reduce `λ_frobenius` (less regularization)
- Increase `mass_threshold` (larger state space)
- Check for sufficient distribution variation

### When probability conservation is poor:
- Increase `λ_prob_conservation`
- Verify reaction connectivity
- Check time step sizes

### When computational cost is too high:
- Reduce `mass_threshold` (fewer states)
- Reduce `snapshots_per_window`
- Use coarser `dt_snapshot`

## Dependencies

- `LinearAlgebra`, `SparseArrays`, `Statistics` (Julia standard library)
- `Catalyst.jl`: Reaction network modeling
- `JumpProcesses.jl`: SSA trajectory generation
- `ExponentialUtilities.jl`: Matrix exponential and Fréchet derivatives
- `NLopt.jl`: Nonlinear optimization
- `StatsBase.jl`: Statistical utilities
- `ProgressLogging.jl`: Progress bars

## Future Extensions

Potential improvements and extensions:
1. Tensor train optimization for very large state spaces
2. Adaptive window sizing based on distribution changes
3. Parallel window optimization
4. Alternative loss functions (KL divergence, Wasserstein)
5. Incorporating prior information on rate constants
6. Multi-scale time stepping

## References

This implementation is based on work using:
- Fréchet derivatives for efficient gradients (Al-Mohy & Higham)
- Sliding window approaches for time-varying systems
- Stoichiometric parameterization for physical constraints

## Contact

For questions about this code, please refer to the project documentation or contact the maintainer.
