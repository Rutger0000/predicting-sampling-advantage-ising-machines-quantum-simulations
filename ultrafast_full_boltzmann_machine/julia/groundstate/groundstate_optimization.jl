using UltraFast

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

# Load the Models
mcset = UltraFast.MCMCSettings(nthermalization=200, nsamples=2000, sweep=lattice.nspins)

# Create Metropolis-Hastings sampler with the above settings                                                 
sampler = UltraFast.Samplers.MHSampler(mcset)

# Make a parallel sampler with 10 chains which uses the above MH sampler
sampler = UltraFast.ParallelMCMCSampler(mcset, sampler, 10)

# Hamiltonian
hamiltonian = UltraFast.Hamiltonians.Heisenberg(lattice=lattice, parallel=false, marshall_sign_rule=true)

# Create gs object
gs = UltraFast.Optimisation.GroundStateOptimisation(sampler=sampler, callback=UltraFast.EnergyLogger(lattice, hamiltonian, 300))

# do gs optimization
output, model = UltraFast.Optimisation.optimize!(gs, model, hamiltonian)

println("Final energy: ", output.energies[end])