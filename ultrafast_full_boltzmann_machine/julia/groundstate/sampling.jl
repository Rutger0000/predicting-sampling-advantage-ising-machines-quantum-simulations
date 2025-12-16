using UltraFast
using BenchmarkTools
using DelimitedFiles

include("../model/ExperimentalFastTranslationInvariantRBM.jl")

Lx = 6
Ly = 6
alpha = 2
lattice = UltraFast.LatticeConfig(Lx=Lx, Ly=Ly)
nsamples = 2000

# ExperimentalFTIRBM model, with alpha=2, real weights, beta=0.5 (which is the setting used in the paper)
model = ExperimentalFTIRBM(
    lattice,
    alpha;
    init = UltraFast.Models.real_default_uniform(),
    beta = 0.5,
)

weights_path = "models/modified_RBM_low_model_id=0/$(lattice.nspins)_$(alpha)_weights.txt"

# Load weights
W_RBM = readdlm(weights_path)
UltraFast.Models.set_weights!(model, vec(W_RBM))

# Load the Models
mcset = UltraFast.MCMCSettings(nthermalization=200, nsamples=nsamples, sweep=lattice.nspins)

# Create MH sampler
sampler = UltraFast.Samplers.MHSampler(mcset)

# Initialize with 10 parallel chains
sampler = UltraFast.ParallelMCMCSampler(mcset, sampler, 10)

# Hamiltonian
hamiltonian = UltraFast.Hamiltonians.Heisenberg(lattice=lattice, parallel=false, marshall_sign_rule=true)

println("Running MH sampling... on $(lattice.nspins) spins with alpha=$(alpha) and nsamples=$(nsamples) at sweep=$(mcset.sweep) MH flips.")

# Run the sampler to get energy estimate
t_sampling = @elapsed states = sample(sampler, model)

println("Sampling time: ", t_sampling, " seconds")

# Define the observable of interest, such as the variational energy
observable = UltraFast.Observables.VariationalEnergy()
# get the observables
energies = UltraFast.Observables.observe(observable; states=states.states, logwavefunctions=states.logwavefunctions, model=model, hamiltonian=hamiltonian)

println("============================")
println("Estimated energy (MH): ", mean(energies) / (lattice.nspins*4), " per site (≈ -0.676 for n = 36 and alpha = 2)")
println("============================")