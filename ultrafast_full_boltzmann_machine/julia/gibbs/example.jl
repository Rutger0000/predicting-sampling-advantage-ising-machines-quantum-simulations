include("gibbs/convenience_chromatic_gibbs.jl")
using DelimitedFiles
using UltraFast

Lx = 6
Ly = 6
alpha = 2

lattice = UltraFast.LatticeConfig(Lx=Lx, Ly=Ly)


########## Load the RBM weights and biases and perform Chromatic Gibbs sampling ##########
# Load W and b, please first run `make convert_weights`
# We load nspins = 36 and alpha = 2 of model_id = 0
W_path = "data/models/modified_RBM_low_model_id=0/W_RBM_$(lattice.nspins)_$(alpha)_ti_W.csv"
b_path = "data/models/modified_RBM_low_model_id=0/W_RBM_$(lattice.nspins)_$(alpha)_ti_b.csv"

W = readdlm(W_path, ',', Float64)
b = readdlm(b_path, ',', Float64)[:]

println("Performing Chromatic Gibbs sampling... on $(lattice.nspins) spins with alpha=$(alpha) and nsamples=32768 at 1 sIM sweep.")

# Perform Chromatic Gibbs sampling
steps = 32768
output, metadata = chromatic_rbm_sampler(nspins=lattice.nspins, 
                                        W=W, 
                                        b=b, 
                                        steps=steps, 
                                        precision=Float64, 
                                        sampling_settings=(;
                                            thermalization=16384, 
                                            sweeps=1, 
                                            save_hidden=false,
                                            mag0=true # very important to set mag0=true, as we calculate observables only over the visible spins
                                        ))

output_visible = output.visible

# Display the sampled visible spins
# display(output_visible) if wanted

######## Use UltraFast to calculate observables from the sampled states ##########

# ExperimentalFTIRBM model, with alpha=2, real weights, beta=0.5 (which is the setting used in the paper)
model = UltraFast.SymmetryRBM.TranslationInvariantRBM(
    lattice,
    alpha;
    init = UltraFast.Models.real_default_uniform(),
    beta = 0.5,
) 

# Load the weights
weights_path = "models/modified_RBM_low_model_id=0/$(lattice.nspins)_$(alpha)_weights.txt"
W_RBM = readdlm(weights_path)
UltraFast.Models.set_weights!(model, vec(W_RBM))

# Calculate observables and logwavefunctions
logwavefunctions = UltraFast.Models.wavefunction_value_parallel(model, output_visible)

# Hamiltonian
hamiltonian = UltraFast.Hamiltonians.Heisenberg(lattice=lattice, parallel=false, marshall_sign_rule=true)

# Define the observable of interest, such as the variational energy
observable = UltraFast.Observables.VariationalEnergy()
# get the observables
output_visible = Int64.(output_visible)
energies = UltraFast.Observables.observe(observable; states=output_visible, logwavefunctions=logwavefunctions, model=model, hamiltonian=hamiltonian)

println("============================")
println("Estimated energy (chromatic Gibbs): ", mean(energies) / (lattice.nspins*4), " per site (≈ -0.676 for n = 36 and alpha = 2)")
println("============================")

mkpath("data/gibbs/raw/$(lattice.nspins)_$(alpha)")

writedlm("data/gibbs/raw/$(lattice.nspins)_$(alpha)/energies.csv", energies, ',')


println("Energies saved to data/gibbs/raw/$(lattice.nspins)_$(alpha)/energies.csv")
