# Single command to do everything:
include("visualization.jl")
analyze_experiments("experiments/")

# Or step-by-step:
suite = load_experiment_suite("experiments/")
generate_all_figures(suite, output_dir="paper_figures")

# Or individual figures:
fig1 = plot_theoretical_validation(suite)
save("my_fig.pdf", fig1)
