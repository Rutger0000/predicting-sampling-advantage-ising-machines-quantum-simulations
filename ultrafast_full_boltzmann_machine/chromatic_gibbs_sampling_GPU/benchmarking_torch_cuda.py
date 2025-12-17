# %%
import torch
import time
import numpy as np
import datetime
import cpuinfo

def gibbs_sampling_timing(device="cuda", alpha=4, nspins=484*4, nsteps=20, trials=10):
    # Define the dimensions
    M = nspins * alpha
    N = nspins

    # Define float precision
    precision = torch.float32  # or torch.float64 for double precision

    # Weights and biases
    W = torch.rand(size=(N, M), dtype=precision) - 0.5
    bias = torch.rand(size=(M,), dtype=precision) - 0.5

    # Initial visible and hidden states
    initial_hidden = torch.rand(size=(M,), dtype=precision)
    initial_visible = torch.rand(size=(N,), dtype=precision)

    # Copy to GPU
    W = W.to(device)
    bias = bias.to(device)
    initial_hidden = initial_hidden.to(device)
    initial_visible = initial_visible.to(device)

    # Define helper functions
    def calculate_hidden(visible, W, bias, M):
        activation = torch.tanh(visible @ W + bias)
        random_uniform = torch.rand(size=(M,), dtype=precision, device=device) * 2 - 1
        return torch.sign(activation - random_uniform)

    def calculate_visible(hidden, W, N):
        activation = torch.tanh(hidden @ W.T)
        random_uniform = torch.rand(size=(N,), dtype=precision, device=device) * 2 - 1
        return torch.sign(activation - random_uniform)

    def gibbs_sampling(visible, W, bias, nsteps):
        N, M = W.shape

        for i in range(nsteps):
            hidden = calculate_hidden(visible, W, bias, M)
            visible = calculate_visible(hidden, W, N)
        return visible

    # Measure the time for gibbs_sampling
    times = []
    for _ in range(trials):
        start_time = time.time()
        visible = initial_visible.clone()  # reset initial visible state
        gibbs_sampling(visible, W, bias, nsteps)
        elapsed_time = time.time() - start_time
        times.append(elapsed_time)

    mean_time = np.mean(times)
    std_time = np.std(times)
    std_error = np.std(times)/np.sqrt(trials-1)

    return mean_time, std_time, std_error


# %%
import platform

# %%
import pandas as pd
import argparse
# Parse command line arguments
parser = argparse.ArgumentParser(description='Gibbs Sampling Benchmarking Tool')
parser.add_argument('--max_nspins', type=int, required=False, help='Maximum number of spins', default=8192)
parser.add_argument('--dev', type=str, default='cuda', help='Device to run the benchmark on')
parser.add_argument('--output', type=str, required=True, help='Output file name')

args = parser.parse_args()

dev = args.dev
gpu = platform.processor()

# Define the array of nspins and alpha values to test
nspins_values = [16, 36, 64, 100, 144, 196, 256, 324, 400, 484, 576, 1024, 2048, 4096, 8192, 16384]
nspins_values = [n for n in nspins_values if n <= args.max_nspins]
alpha_values = [1, 2, 4, 8]

# Prepare a list to store the results
results = []

trials = 10
nsteps = 10

# Loop over all combinations of nspins and alpha
for nspins in nspins_values:
    for alpha in alpha_values:
        print(f"Running {nspins} {alpha}")
        mean_time, std_time, std_error = gibbs_sampling_timing(device=dev, alpha=alpha, nspins=nspins, trials=trials, nsteps=nsteps)
        results.append({
            "nspins": nspins,
            "alpha": alpha,
            "mean": mean_time,
            "std": std_time,
            "std_error": std_error,
            "trials": trials,
            "nsteps": nsteps,
            "device": gpu,
            "timestamp": datetime.datetime.now().isoformat(),
            'Platform': platform.platform(),
            'Python Version': platform.python_version(),
            'cpu': cpuinfo.get_cpu_info()['brand_raw']
        })

# Convert the results list to a pandas DataFrame
df = pd.DataFrame(results)

# Display the DataFrame
print(df)

df.to_csv(args.output)
