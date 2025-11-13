using LinearAlgebra
using Printf

"""
Project matrix to valid CME generator space.

Enforces three constraints:
1. Off-diagonal entries ≥ 0 (A[i,j] ≥ 0 for i ≠ j)
2. Diagonal entries ≤ 0 (A[i,i] ≤ 0)
3. ROW sums = 0 (∑ⱼ A[i,j] = 0 for all i)

NOTE: CME generators have dp/dt = Ap, so ROWS must sum to zero!

# Arguments
- `A::Matrix`: Input matrix
- `mode::Symbol`: 
  - `:diagonal` - Set diagonals to enforce row sum constraint (default)
  - `:project` - Simple projection (may not preserve structure)
  - `:balanced` - Balance corrections across off-diagonals

# Returns
- `A_proj::Matrix`: Projected generator matrix

# Example
```julia
A_proj = project_to_generator(A_recovered; mode=:diagonal)
```
"""
function project_to_generator(A::AbstractMatrix; mode::Symbol=:diagonal)
    n = size(A, 1)
    A_proj = copy(A)
    
    if mode == :diagonal
        # Step 1: Ensure off-diagonals are non-negative
        for i in 1:n, j in 1:n
            if i != j
                A_proj[i, j] = max(A_proj[i, j], 0.0)
            end
        end
        
        # Step 2: Set diagonals to enforce ROW sum = 0
        for i in 1:n
            A_proj[i, i] = -sum(A_proj[i, j] for j in 1:n if j != i)
        end
        
    elseif mode == :project
        # Simple projection to constraint set
        for i in 1:n, j in 1:n
            if i != j
                A_proj[i, j] = max(A_proj[i, j], 0.0)
            else
                A_proj[i, i] = min(A_proj[i, i], 0.0)
            end
        end
        
        # Adjust to satisfy ROW sum constraint
        for i in 1:n
            row_sum = sum(A_proj[i, :])
            # Distribute error to diagonal
            A_proj[i, i] -= row_sum
        end
        
    elseif mode == :balanced
        # Balance corrections across off-diagonals
        for i in 1:n
            # Get current off-diagonal entries in this row
            off_diag_indices = [j for j in 1:n if j != i]
            off_diag_vals = [A_proj[i, j] for j in off_diag_indices]
            
            # Project to non-negative
            off_diag_vals_proj = max.(off_diag_vals, 0.0)
            
            # Update off-diagonals
            for (idx, j) in enumerate(off_diag_indices)
                A_proj[i, j] = off_diag_vals_proj[idx]
            end
            
            # Set diagonal to enforce row sum = 0
            A_proj[i, i] = -sum(off_diag_vals_proj)
        end
    end
    
    return A_proj
end


"""
Verify that a matrix satisfies CME generator constraints.

# Returns
- `valid::Bool`: True if all constraints satisfied
- `violations::Dict`: Dictionary of constraint violations

# Example
```julia
valid, violations = verify_generator_constraints(A; tol=1e-8)
if !valid
    println("Violations: ", violations)
end
```
"""
function verify_generator_constraints(A::AbstractMatrix; tol::Float64=1e-8)
    n = size(A, 1)
    violations = Dict{String, Any}()
    
    # Check off-diagonals ≥ 0
    off_diag_violations = [(i, j, A[i,j]) for i in 1:n, j in 1:n 
                           if i != j && A[i,j] < -tol]
    if !isempty(off_diag_violations)
        violations["off_diagonal_negative"] = off_diag_violations
    end
    
    # Check diagonals ≤ 0
    diag_violations = [(i, A[i,i]) for i in 1:n if A[i,i] > tol]
    if !isempty(diag_violations)
        violations["diagonal_positive"] = diag_violations
    end
    
    # Check ROW sums = 0 (NOT column sums!)
    row_sum_violations = []
    for i in 1:n
        row_sum = sum(A[i, j] for j in 1:n)
        if abs(row_sum) > tol
            push!(row_sum_violations, (i, row_sum))
        end
    end
    if !isempty(row_sum_violations)
        violations["row_sum_nonzero"] = row_sum_violations
    end
    
    valid = isempty(violations)
    
    return valid, violations
end


"""
Print constraint violation report.
"""
function print_constraint_report(A::AbstractMatrix; tol::Float64=1e-8)
    valid, violations = verify_generator_constraints(A; tol=tol)
    
    println("\n" * "="^60)
    println("Generator Constraint Report")
    println("="^60)
    
    if valid
        println("✓ All constraints satisfied!")
    else
        println("✗ Constraint violations detected:")
        
        if haskey(violations, "off_diagonal_negative")
            println("\n  Off-diagonal entries < 0:")
            for (i, j, val) in violations["off_diagonal_negative"][1:min(5, length(violations["off_diagonal_negative"]))]
                @printf("    A[%d,%d] = %.6e\n", i, j, val)
            end
            if length(violations["off_diagonal_negative"]) > 5
                println("    ... (", length(violations["off_diagonal_negative"]), " total)")
            end
        end
        
        if haskey(violations, "diagonal_positive")
            println("\n  Diagonal entries > 0:")
            for (i, val) in violations["diagonal_positive"][1:min(5, length(violations["diagonal_positive"]))]
                @printf("    A[%d,%d] = %.6e\n", i, i, val)
            end
            if length(violations["diagonal_positive"]) > 5
                println("    ... (", length(violations["diagonal_positive"]), " total)")
            end
        end
        
        if haskey(violations, "row_sum_nonzero")
            println("\n  Row sums ≠ 0:")
            for (i, sum_val) in violations["row_sum_nonzero"][1:min(5, length(violations["row_sum_nonzero"]))]
                @printf("    Row %d: sum = %.6e\n", i, sum_val)
            end
            if length(violations["row_sum_nonzero"]) > 5
                println("    ... (", length(violations["row_sum_nonzero"]), " total)")
            end
        end
    end
    
    println("="^60)
    
    return valid, violations
end


"""
Recover generator with POST-optimization projection to constraint manifold.

This is the simplest approach:
1. Run unconstrained optimization with Optim.jl
2. Project final result to generator space

# Example
```julia
A_recovered, result = recover_generator_postprojected(
    p_curr, p_prev, δt, A_init;
    method = :LBFGS,
    projection_mode = :diagonal
)
```
"""
function recover_generator_postprojected(
    p_curr::Vector{Float64},
    p_prev::Vector{Float64},
    δt::Real,
    A_init::AbstractMatrix;
    n_quad::Int = 50,
    method::Symbol = :LBFGS,
    projection_mode::Symbol = :diagonal,
    max_iter::Int = 1000,
    g_tol::Float64 = 1e-6,
    show_trace::Bool = true
)
    # Run unconstrained optimization (use existing function)
    A_unconstrained, result = recover_generator_optim(
        p_curr, p_prev, δt, A_init;
        n_quad = n_quad,
        method = method,
        max_iter = max_iter,
        g_tol = g_tol,
        show_trace = show_trace
    )
    
    # Project to generator space
    A_projected = project_to_generator(A_unconstrained; mode=projection_mode)
    
    return A_projected, result, A_unconstrained
end


"""
Test constraint projection with a simple example.
"""
function test_projection()
    println("\n" * "="^70)
    println("Testing CME Generator Projection")
    println("="^70)
    
    n = 3
    
    # Create a matrix that violates constraints
    A_bad = [
        -1.5   1.2  -0.3;   # A[1,3] is negative (bad)
         0.8  -2.0   1.5;
         0.5   0.9  -1.8
    ]
    
    println("\nOriginal matrix (violates constraints):")
    display(A_bad)
    
    # Check violations
    println("\nBefore projection:")
    print_constraint_report(A_bad)
    
    # Project with different modes
    modes = [:diagonal, :project, :balanced]
    
    for mode in modes
        println("\n" * "="^70)
        println("Projection mode: $mode")
        println("="^70)
        
        A_proj = project_to_generator(A_bad; mode=mode)
        
        println("\nProjected matrix:")
        display(A_proj)
        
        print_constraint_report(A_proj)
    end
end


"""
Test recovery with constraint enforcement.
"""
function test_recovery_with_constraints()
    println("\n" * "="^70)
    println("Testing Generator Recovery with Constraints")
    println("="^70)
    
    n = 3
    δt = 0.1
    
    # True generator (satisfies constraints)
    A_true = [-2.0  1.0  1.0;
               0.5 -1.5  1.0;
               0.5  0.5 -1.0]
    
    println("\nTrue generator:")
    display(A_true)
    print_constraint_report(A_true)
    
    # Generate data
    p_prev = [1.0, 0.0, 0.0]
    p_curr = exp(A_true * δt) * p_prev
    
    println("\nData:")
    println("  p(t-δt) = ", p_prev)
    println("  p(t)    = ", round.(p_curr, digits=6))
    
    # Initial guess (may violate constraints)
    A_init = randn(n, n) * 0.1
    
    println("\n" * "="^70)
    println("Method 1: Unconstrained optimization")
    println("="^70)
    
    A_unconstrained, result_unc = recover_generator_optim(
        p_curr, p_prev, δt, A_init;
        method = :LBFGS,
        n_quad = 50,
        show_trace = false
    )
    
    println("\nRecovered generator (unconstrained):")
    display(A_unconstrained)
    println("\nError: ", norm(A_true - A_unconstrained))
    print_constraint_report(A_unconstrained)
    
    println("\n" * "="^70)
    println("Method 2: Post-projection to constraints")
    println("="^70)
    
    A_projected, result_proj, _ = recover_generator_postprojected(
        p_curr, p_prev, δt, A_init;
        method = :LBFGS,
        projection_mode = :diagonal,
        n_quad = 50,
        show_trace = false
    )
    
    println("\nRecovered generator (post-projected):")
    display(A_projected)
    println("\nError: ", norm(A_true - A_projected))
    print_constraint_report(A_projected)
    
    # Check forward simulation
    p_pred_unc = exp(A_unconstrained * δt) * p_prev
    p_pred_proj = exp(A_projected * δt) * p_prev
    
    println("\nForward simulation errors:")
    println("  Unconstrained: ", norm(p_curr - p_pred_unc))
    println("  Post-projected: ", norm(p_curr - p_pred_proj))
    
    # Summary comparison
    println("\n" * "="^70)
    println("Summary")
    println("="^70)
    @printf("%-20s  %12s  %12s  %12s\n", "Method", "Error", "Fwd Error", "Valid?")
    println("-"^70)
    
    valid_unc, _ = verify_generator_constraints(A_unconstrained)
    valid_proj, _ = verify_generator_constraints(A_projected)
    
    @printf("%-20s  %.4e  %.4e  %s\n",
            "Unconstrained",
            norm(A_true - A_unconstrained),
            norm(p_curr - p_pred_unc),
            valid_unc ? "✓" : "✗")
    
    @printf("%-20s  %.4e  %.4e  %s\n",
            "Post-projected",
            norm(A_true - A_projected),
            norm(p_curr - p_pred_proj),
            valid_proj ? "✓" : "✗")
    
    return A_unconstrained, A_projected, result_unc, result_proj
end


"""
Run all constraint-related tests.
"""
function run_constraint_tests()
    println("\n")
    println("╔" * "="^68 * "╗")
    println("║" * " "^20 * "CME CONSTRAINT TESTS" * " "^28 * "║")
    println("╚" * "="^68 * "╝")
    
    # Test 1: Projection
    test_projection()
    
    # Test 2: Recovery with constraints
    test_recovery_with_constraints()
    
    println("\n" * "="^70)
    println("All constraint tests completed!")
    println("="^70)
end
