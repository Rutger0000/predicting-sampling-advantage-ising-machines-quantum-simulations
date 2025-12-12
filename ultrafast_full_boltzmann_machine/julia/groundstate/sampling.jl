using UltraFast
using BenchmarkTools
using DelimitedFiles

include("../model/ExperimentalFastTranslationInvariantRBM.jl")

Lx = 4
Ly = 4
lattice = UltraFast.LatticeConfig(Lx=Lx, Ly=Ly)

# ExperimentalFTIRBM model, with alpha=2, real weights, beta=0.5 (which is the setting used in the paper)
model = ExperimentalFTIRBM(
    lattice,
    2;
    init = UltraFast.Models.real_default_uniform(),
    beta = 0.5,
)

weights_path = "models/modified_RBM_low_model_id=0/16_2_weights.txt"

# Load weights
W_RBM = readdlm(weights_path)
UltraFast.Models.set_weights!(model, vec(W_RBM))

# Load the Models
mcset = UltraFast.MCMCSettings(nthermalization=200, nsamples=2000, sweep=lattice.nspins)

# Create MH sampler
sampler = UltraFast.Samplers.MHSampler(mcset)

# Initialize with 10 parallel chains
sampler = UltraFast.ParallelMCMCSampler(mcset, sampler, 10)

# Hamiltonian
hamiltonian = UltraFast.Hamiltonians.Heisenberg(lattice=lattice, parallel=false, marshall_sign_rule=true)

# Run the sampler to get energy estimate
t_sampling = @elapsed states = sample(sampler, model)

println("Sampling time: ", t_sampling, " seconds")

# Define the observable of interest, such as the variational energy
observable = UltraFast.Observables.VariationalEnergy()
# get the observables
energies = UltraFast.Observables.observe(observable; states=states.states, logwavefunctions=states.logwavefunctions, model=model, hamiltonian=hamiltonian)

println("Estimated energy: ", mean(energies) / (lattice.nspins*4), " per site (≈ -0.70)")