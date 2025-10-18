import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def load_data(file_path):
    """Load training run data from JSON file"""
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def group_by_compute_budget(data):
    """Group data by compute budget"""
    budgets = {}
    for run in data:
        budget = run['compute_budget']
        if budget not in budgets:
            budgets[budget] = []
        budgets[budget].append(run)
    
    # Sort each group by parameters
    for budget in budgets:
        budgets[budget].sort(key=lambda x: x['parameters'])
    
    return budgets

def find_optimal_points_per_budget(grouped_data):
    """Find the optimal (lowest loss) point for each compute budget"""
    optimal_points = {}
    for budget, runs in grouped_data.items():
        # Find the run with minimum final loss for this budget
        best_run = min(runs, key=lambda x: x['final_loss'])
        optimal_points[budget] = best_run
    
    return optimal_points

def scaling_law_function(N, L_inf, N_c, alpha):
    """Scaling law function: L(N) = L_inf + (N_c / N)^alpha"""
    return L_inf + (N_c / N) ** alpha

def fit_scaling_law(optimal_points):
    """Fit scaling law to the optimal points"""
    # Extract parameters and losses
    N_values = np.array([point['parameters'] for point in optimal_points.values()])
    L_values = np.array([point['final_loss'] for point in optimal_points.values()])
    
    # Initial parameter guesses
    L_inf_guess = np.min(L_values) * 0.9
    N_c_guess = np.median(N_values)
    alpha_guess = 0.5
    
    # Fit the curve
    try:
        popt, pcov = curve_fit(scaling_law_function, N_values, L_values, 
                              p0=[L_inf_guess, N_c_guess, alpha_guess],
                              bounds=([0, 0, 0], [np.inf, np.inf, np.inf]))
        
        L_inf, N_c, alpha = popt
        return L_inf, N_c, alpha, pcov
    except Exception as e:
        print(f"Fitting failed: {e}")
        return None, None, None, None

def compute_optimal_model_size_scaling(optimal_points):
    """Fit scaling law for compute-optimal model size N_opt(C)"""
    # Extract compute budgets and optimal model sizes
    C_values = np.array(list(optimal_points.keys()))
    N_opt_values = np.array([point['parameters'] for point in optimal_points.values()])
    
    # Fit power law: N_opt = a * C^b
    def power_law(C, a, b):
        return a * C ** b
    
    # Initial guesses
    a_guess = N_opt_values[0] / (C_values[0] ** 0.5)
    b_guess = 0.5
    
    try:
        popt, pcov = curve_fit(power_law, C_values, N_opt_values, 
                              p0=[a_guess, b_guess],
                              bounds=([0, 0], [np.inf, np.inf]))
        a, b = popt
        return a, b, pcov
    except Exception as e:
        print(f"Model size scaling fit failed: {e}")
        return None, None, None

def compute_optimal_dataset_size_scaling(optimal_points):
    """Fit scaling law for compute-optimal dataset size D_opt(C)"""
    # Extract compute budgets and calculate dataset sizes using C ≈ 6ND
    C_values = np.array(list(optimal_points.keys()))
    D_opt_values = []
    
    for budget, point in optimal_points.items():
        # Using the approximation C ≈ 6ND
        D = budget / (6 * point['parameters'])
        D_opt_values.append(D)
    
    D_opt_values = np.array(D_opt_values)
    
    # Fit power law: D_opt = c * C^d
    def power_law(C, c, d):
        return c * C ** d
    
    # Initial guesses
    c_guess = D_opt_values[0] / (C_values[0] ** 0.5)
    d_guess = 0.5
    
    try:
        popt, pcov = curve_fit(power_law, C_values, D_opt_values, 
                              p0=[c_guess, d_guess],
                              bounds=([0, 0], [np.inf, np.inf]))
        c, d = popt
        return c, d, pcov, D_opt_values
    except Exception as e:
        print(f"Dataset size scaling fit failed: {e}")
        return None, None, None, None

def plot_model_size_scaling(data, optimal_points, a, b):
    """Plot compute-optimal model size scaling law with all data points"""
    plt.figure(figsize=(12, 8))
    
    # Extract optimal points
    C_opt_values = np.array(list(optimal_points.keys()))
    N_opt_values = np.array([point['parameters'] for point in optimal_points.values()])
    
    # Extract ALL data points from the original data
    all_C_values = []
    all_N_values = []
    for run in data:
        all_C_values.append(run['compute_budget'])
        all_N_values.append(run['parameters'])
    
    # Plot all data points with different colors for different compute budgets
    unique_budgets = sorted(set(all_C_values))
    colors = plt.cm.viridis(np.linspace(0, 1, len(unique_budgets)))
    
    budget_color_map = {}
    for i, budget in enumerate(unique_budgets):
        budget_color_map[budget] = colors[i]
    
    for run in data:
        budget = run['compute_budget']
        plt.scatter(budget, run['parameters'], 
                   color=budget_color_map[budget], alpha=0.6, s=40)
    
    # Plot optimal points with larger markers
    plt.scatter(C_opt_values, N_opt_values, color='red', marker='*', s=200, 
                label='Optimal Points', zorder=5, edgecolors='black', linewidth=1)
    
    # Plot fitted scaling law with extrapolation
    C_range = np.logspace(np.log10(min(C_opt_values)*0.1), 24, 200)  # Extrapolate to 10^24
    N_pred = a * C_range ** b
    
    plt.plot(C_range, N_pred, 'r-', linewidth=3, 
             label=f'Fit: N_opt = {a:.2e} * C^{b:.3f}')
    
    # Highlight extrapolation points
    C_extrapolate = [1e23, 1e24]
    N_extrapolate = a * np.array(C_extrapolate) ** b
    
    plt.scatter(C_extrapolate, N_extrapolate, color='green', marker='D', s=200, 
                label='Extrapolated Points', zorder=6, edgecolors='black', linewidth=1)
    
    # Add budget labels for some representative points
    for i, budget in enumerate(unique_budgets[::2]):  # Every other budget for clarity
        plt.text(budget*1.2, max(all_N_values)*0.8/(i+1), f'{budget:.0e}', 
                fontsize=8, alpha=0.7)
    
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Compute Budget C (FLOPs)')
    plt.ylabel('Model Size N (Parameters)')
    plt.title('Compute-Optimal Model Size Scaling Law (All Data Points)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("model_size_scaling_all_points.png", dpi=300)
    plt.show()
    
    return N_extrapolate

def plot_dataset_size_scaling(data, optimal_points, c, d, D_opt_values):
    """Plot compute-optimal dataset size scaling law with all data points"""
    plt.figure(figsize=(12, 8))
    
    # Extract optimal points
    C_opt_values = np.array(list(optimal_points.keys()))
    
    # Calculate dataset sizes for ALL data points using C ≈ 6ND
    all_C_values = []
    all_D_values = []
    for run in data:
        C = run['compute_budget']
        N = run['parameters']
        D = C / (6 * N)  # Using approximation C ≈ 6ND
        all_C_values.append(C)
        all_D_values.append(D)
    
    # Plot all data points with different colors for different compute budgets
    unique_budgets = sorted(set(all_C_values))
    colors = plt.cm.plasma(np.linspace(0, 1, len(unique_budgets)))
    
    budget_color_map = {}
    for i, budget in enumerate(unique_budgets):
        budget_color_map[budget] = colors[i]
    
    for i, run in enumerate(data):
        budget = run['compute_budget']
        D = all_D_values[i]
        plt.scatter(budget, D, color=budget_color_map[budget], alpha=0.6, s=40)
    
    # Plot optimal points with larger markers
    plt.scatter(C_opt_values, D_opt_values, color='red', marker='*', s=200, 
                label='Optimal Points', zorder=5, edgecolors='black', linewidth=1)
    
    # Plot fitted scaling law with extrapolation
    C_range = np.logspace(np.log10(min(C_opt_values)*0.1), 24, 200)  # Extrapolate to 10^24
    D_pred = c * C_range ** d
    
    plt.plot(C_range, D_pred, 'r-', linewidth=3, 
             label=f'Fit: D_opt = {c:.2e} * C^{d:.3f}')
    
    # Highlight extrapolation points
    C_extrapolate = [1e23, 1e24]
    D_extrapolate = c * np.array(C_extrapolate) ** d
    
    plt.scatter(C_extrapolate, D_extrapolate, color='green', marker='D', s=200, 
                label='Extrapolated Points', zorder=6, edgecolors='black', linewidth=1)
    
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Compute Budget C (FLOPs)')
    plt.ylabel('Dataset Size D (Tokens)')
    plt.title('Compute-Optimal Dataset Size Scaling Law (All Data Points)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("dataset_size_scaling_all_points.png", dpi=300)
    plt.show()
    
    return D_extrapolate

def plot_isoflops_analysis(data, grouped_data, optimal_points, L_inf, N_c, alpha):
    """Plot IsoFLOPs analysis with all data points"""
    plt.figure(figsize=(12, 8))
    
    # Plot all data points, colored by compute budget
    budgets = list(grouped_data.keys())
    colors = plt.cm.viridis(np.linspace(0, 1, len(budgets)))
    
    for i, budget in enumerate(budgets):
        runs = grouped_data[budget]
        N = [run['parameters'] for run in runs]
        L = [run['final_loss'] for run in runs]
        plt.scatter(N, L, color=colors[i], alpha=0.7, 
                   label=f'C={budget:.1e}', s=60)
    
    # Highlight optimal points
    opt_N = [point['parameters'] for point in optimal_points.values()]
    opt_L = [point['final_loss'] for point in optimal_points.values()]
    plt.scatter(opt_N, opt_L, color='red', marker='*', s=200, 
               label='Optimal Points (IsoFLOPs)', zorder=5, edgecolors='black', linewidth=1)
    
    # Plot fitted scaling law
    if L_inf is not None and N_c is not None and alpha is not None:
        N_range = np.logspace(np.log10(min(opt_N)*0.8), np.log10(max(opt_N)*1.2), 100)
        L_fit = scaling_law_function(N_range, L_inf, N_c, alpha)
        plt.plot(N_range, L_fit, 'r--', linewidth=2, 
                label=f'Fit: L_inf={L_inf:.3f}, N_c={N_c:.2e}, alpha={alpha:.3f}')
    
    plt.xscale('log')
    plt.xlabel('Parameters (N)')
    plt.ylabel('Final Loss (L)')
    plt.title('IsoFLOPs Scaling Law Analysis (All Data Points)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("isoflops_analysis_all_points.png", dpi=300)
    plt.show()

def main():
    # Load data
    file_path = 'data/isoflops_curves.json'
    data = load_data(file_path)
    
    # Group data by compute budget
    grouped_data = group_by_compute_budget(data)
    
    # Find optimal points for each compute budget
    optimal_points = find_optimal_points_per_budget(grouped_data)
    
    # Fit original scaling law
    L_inf, N_c, alpha, pcov = fit_scaling_law(optimal_points)
    
    # Plot IsoFLOPs analysis with all points
    plot_isoflops_analysis(data, grouped_data, optimal_points, L_inf, N_c, alpha)
    
    # Fit compute-optimal model size scaling
    a, b, _ = compute_optimal_model_size_scaling(optimal_points)
    
    # Fit compute-optimal dataset size scaling
    c, d, _, D_opt_values = compute_optimal_dataset_size_scaling(optimal_points)
    
    # Print results
    print("IsoFLOPs Scaling Law Results:")
    print("=" * 50)
    print(f"L_inf (asymptotic loss): {L_inf:.6f}")
    print(f"N_c (critical parameters): {N_c:.2e}")
    print(f"alpha (scaling exponent): {alpha:.6f}")
    
    print("\nCompute-Optimal Model Size Scaling:")
    print(f"N_opt(C) = {a:.6e} * C^{b:.6f}")
    
    print("\nCompute-Optimal Dataset Size Scaling:")
    print(f"D_opt(C) = {c:.6e} * C^{d:.6f}")
    
    # Predict optimal sizes for 10^23 and 10^24 FLOPs
    N_23 = a * (1e23) ** b
    N_24 = a * (1e24) ** b
    D_23 = c * (1e23) ** d
    D_24 = c * (1e24) ** d
    
    print("\nPredicted Optimal Sizes:")
    print(f"For C = 10^23 FLOPs: N_opt = {N_23:.2e} parameters, D_opt = {D_23:.2e} tokens")
    print(f"For C = 10^24 FLOPs: N_opt = {N_24:.2e} parameters, D_opt = {D_24:.2e} tokens")
    
    # Plot results with all data points
    N_extrapolate = plot_model_size_scaling(data, optimal_points, a, b)
    D_extrapolate = plot_dataset_size_scaling(data, optimal_points, c, d, D_opt_values)
    
    return L_inf, N_c, alpha, optimal_points, a, b, c, d

if __name__ == "__main__":
    L_inf, N_c, alpha, optimal_points, a, b, c, d = main()
