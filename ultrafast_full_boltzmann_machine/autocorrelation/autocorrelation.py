# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def sample_autocovariance(chain):
    """
    Calculate the autocovariance for a chain of samples.
    """
    x_shifted = chain - np.mean(chain)
    n = len(chain)
    autocov = np.array([np.dot(x_shifted[:n - t], x_shifted[t:]) / n for t in range(n)])
    return np.arange(n), autocov

def calculate_autocorrelation_time_integrated(autocovariance):
    """
    Calculate integrated autocorrelation time, see Joseph et al. 2020, doi:10.1007/978-3-030-46044-0
    
    Parameters
    ----------
    autocovariance : np.ndarray
        Normalized autocovariance values
        
    Returns
    -------
    tau_int : float
        Integrated autocorrelation time
    M_cutoff : float
        Cutoff time (where t >= 4*tau_int + 1)
    """
    assert autocovariance[0] == 1.0, "Autocovariance must be normalized to 1 at t=0"

    # find integrated autocovariance by summing from t = 1 to infty or max t
    autocovint_cumsum = 0.5 + np.cumsum(autocovariance[1:])
    M = np.arange(1, len(autocovint_cumsum)+1)

    # find the cutoff M where t >= 4*tau_int + 1
    M_cutoff = M[M >= 4*autocovint_cumsum + 1][0]
    tau_int = autocovint_cumsum[M >= 4*autocovint_cumsum + 1][0]
    
    return tau_int, M_cutoff

# %%

if __name__ == "__main__":
    # load autocovariance data which can be created by running ultrafast_full_boltzmann_machine/julia/gibbs/example.jl
    df = pd.read_csv("data/gibbs/raw/36_2/energies.csv", header=None, names=["energy"])
    autocovariance = sample_autocovariance(df["energy"].values)[1]
    autocovariance /= autocovariance[0] # normalize autocovariance

    print("Calculating integrated autocorrelation time from chromatic Gibbs sampling energy measurements...")

    tau_int, M_cutoff = calculate_autocorrelation_time_integrated(autocovariance)
    print(f"Integrated autocorrelation time: {tau_int} (≈ 2 sIM sweeps, see Fig. 2 for n = 36, alpha = 2)")
    print(f"Cutoff time C: {M_cutoff}")

