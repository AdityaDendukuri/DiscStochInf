# File: src/analysis.jl

"""
Analysis and visualization utilities.
"""

using LinearAlgebra
using Plots
using ExponentialUtilities

"""
    compute_prediction_error(A, window, dt)

Compute L1 prediction error for a window.
"""
function compute_prediction_error(A, window, dt)
    expA = exponential!(Matrix(A * dt), ExpMethodHigham2005Base())
    p_pred = expA * window.p_curr
    return sum(abs.(window.p_next - p_pred))
end

"""
    check_generator_properties(A)

Verify generator properties (column sums, etc.).
"""
function check_generator_properties(A)
    col_sums = sum(A, dims=1)[:]
    max_col_sum_error = maximum(abs.(col_sums))
    
    off_diag_negative = any(A[i,j] < -1e-10 for i in 1:size(A,1), j in 1:size(A,2) if i != j)
    
    println("Generator properties:")
    println("  Max |column sum|: $(round(max_col_sum_error, digits=10))")
    println("  Negative off-diagonal? $off_diag_negative")
    println("  ||A||: $(round(norm(A), digits=2))")
    println("  Nonzeros: $(count(abs.(A) .> 1e-10))")
end

"""
    evaluate_propensities(θ, data, states)

Evaluate learned propensities at given states.
"""
function evaluate_propensities(θ, data, states)
    n_reactions = length(data.stoich_basis)
    results = Dict()
    
    for state in states
        # Evaluate basis at this state
        features = if state in data.state_space
            idx = findfirst(==(state), data.state_space)
            data.state_features[:, idx]
        else
            # State not in training set, evaluate basis directly
            evaluate(data.basis, state)
        end
        
        props = zeros(n_reactions)
        for k in 1:n_reactions
            θ_k = θ[(k-1)*data.n_features + 1 : k*data.n_features]
            
            # Check dimension match
            if length(θ_k) != length(features)
                error("Parameter dimension ($(length(θ_k))) doesn't match basis dimension ($(length(features)))")
            end
            
            props[k] = max(0.0, dot(θ_k, features))
        end
        
        results[state] = props
    end
    
    return results
end

"""
    plot_generator(A, state_space; title="")

Visualize generator matrix.
"""
function plot_generator(A, state_space; title="Learned Generator")
    p = heatmap(A, yflip=true, c=:balance, 
               title=title, aspect_ratio=:equal,
               xlabel="State index", ylabel="State index")
    return p
end

"""
    plot_histograms(hists, state_space, indices; layout=(2,2))

Visualize probability histograms as heatmaps.
"""
function plot_histograms(hists, indices; layout=(2,2))
    plots = []
    
    for i in indices
        hist = hists[i]
        max_x = maximum(s[1] for s in keys(hist))
        max_y = maximum(s[2] for s in keys(hist))
        
        P = zeros(max_y+1, max_x+1)
        for (state, prob) in hist
            P[state[2]+1, state[1]+1] = prob
        end
        
        p = heatmap(P, c=:viridis, title="t=$(i-1)",
                   xlabel="X", ylabel="Y", yflip=false)
        push!(plots, p)
    end
    
    plot(plots..., layout=layout, size=(800, 800))
end
