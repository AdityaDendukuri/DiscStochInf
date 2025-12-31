# File: src/basis_functions.jl

"""
Basis function definitions for propensity parameterization.
"""

using LinearAlgebra

"""
Abstract type for basis functions.
"""
abstract type PropensityBasis end

"""
    PolynomialBasis{d, p}

Polynomial basis up to degree p in d variables.
"""
struct PolynomialBasis{d, p} <: PropensityBasis
    n_features::Int
    
    function PolynomialBasis{d, p}() where {d, p}
        # Count number of monomials: binomial(d+p, p)
        n = binomial(d + p, p)
        new{d, p}(n)
    end
end

"""
    evaluate(basis::PolynomialBasis{2, 1}, state)

Evaluate linear basis: [1, x, y]
"""
function evaluate(basis::PolynomialBasis{2, 1}, state::Vector{Int})
    x, y = state[1], state[2]
    return [1.0, Float64(x), Float64(y)]
end

"""
    evaluate(basis::PolynomialBasis{2, 2}, state)

Evaluate quadratic basis: [1, x, y, xy, x², y²]
"""
function evaluate(basis::PolynomialBasis{2, 2}, state::Vector{Int})
    x, y = state[1], state[2]
    return [1.0, Float64(x), Float64(y), Float64(x*y), Float64(x^2), Float64(y^2)]
end

"""
    evaluate(basis::PolynomialBasis{2, 3}, state)

Evaluate cubic basis: [1, x, y, xy, x², y², x²y, xy², x³, y³]
"""
function evaluate(basis::PolynomialBasis{2, 3}, state::Vector{Int})
    x, y = state[1], state[2]
    return [1.0, Float64(x), Float64(y), 
            Float64(x*y), Float64(x^2), Float64(y^2),
            Float64(x^2*y), Float64(x*y^2), Float64(x^3), Float64(y^3)]
end

"""
    evaluate(basis::PolynomialBasis{3, 2}, state)

Evaluate quadratic basis for 3 species: [1, x, y, z, xy, xz, yz, x², y², z²]
"""
function evaluate(basis::PolynomialBasis{3, 2}, state::Vector{Int})
    x, y, z = state[1], state[2], state[3]
    return [1.0, Float64(x), Float64(y), Float64(z),
            Float64(x*y), Float64(x*z), Float64(y*z),
            Float64(x^2), Float64(y^2), Float64(z^2)]
end

"""
    get_n_features(basis::PropensityBasis)

Get number of features for this basis.
"""
get_n_features(basis::PropensityBasis) = basis.n_features

"""
    compute_state_features(state_space, basis::PropensityBasis)

Compute features for all states using the given basis.
"""
function compute_state_features(state_space, basis::PropensityBasis)
    n_states = length(state_space)
    n_features = get_n_features(basis)
    
    features = zeros(n_features, n_states)
    for (i, state) in enumerate(state_space)
        features[:, i] = evaluate(basis, state)
    end
    
    return features
end
