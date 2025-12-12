using UltraFast
using Optimisers

mutable struct ExperimentalFTIRBM{T} <: UltraFast.Models.Model{T}
    model::UltraFast.SymmetryRBM.FastTranslationInvariantRBM{T}
    beta::Float64
end

function ExperimentalFTIRBM(lattice::UltraFast.LatticeConfig, alpha::Int; init = UltraFast.Models.real_default_uniform(), beta=0.5)
    model = UltraFast.SymmetryRBM.FastTranslationInvariantRBM(lattice, alpha; init = init)
    return ExperimentalFTIRBM(model, beta)
end

# optimiser
UltraFast.Models.setup_optimiser(m::ExperimentalFTIRBM, optimiser::Optimisers.AbstractRule) = UltraFast.Models.setup_optimiser(m.model, optimiser)
UltraFast.Models.number_of_parameters(m::ExperimentalFTIRBM) = UltraFast.Models.number_of_parameters(m.model)

# states
UltraFast.Models.new_state!(m::ExperimentalFTIRBM, state::AbstractVector) = UltraFast.Models.new_state!(m.model, state)
UltraFast.Models.handle_state_update!(m::ExperimentalFTIRBM, state::AbstractVector, flips::AbstractVector) = UltraFast.Models.handle_state_update!(m.model, state, flips)

# weights
UltraFast.Models.update_weights!(opt_state, m::ExperimentalFTIRBM, gradient::AbstractVector) = UltraFast.Models.update_weights!(opt_state, m.model, gradient)
UltraFast.SymmetryRBM.independent_weights(m::ExperimentalFTIRBM) = UltraFast.SymmetryRBM.independent_weights(m.model)
UltraFast.Models.set_weights!(m::ExperimentalFTIRBM, W_RBM::AbstractVector) = UltraFast.Models.set_weights!(m.model, W_RBM)
UltraFast.Models.get_RBM_weights(m::ExperimentalFTIRBM) = UltraFast.Models.get_RBM_weights(m.model)

# gradient
UltraFast.Models.gradient(m::ExperimentalFTIRBM, state::AbstractVector) = UltraFast.Models.gradient(m.model, state) .* m.beta

# settings
UltraFast.Models.settings(m::ExperimentalFTIRBM) = Dict("type" => UltraFast.Models.identifier(m), "nspins" => m.model.lattice.nspins, "alpha" => m.model.alpha, "beta" => m.beta)

# wavefunction
UltraFast.Models.wavefunction_value(m::ExperimentalFTIRBM, state::Array{Int64}) = UltraFast.Models.wavefunction_value(m.model, state) * m.beta
UltraFast.Models.wavefunction_value(m::ExperimentalFTIRBM, state::Array{Int64}, flips::Array{Int64}) = UltraFast.Models.wavefunction_value(m.model, state, flips) .* m.beta

# nspins
UltraFast.Models.nspins(m::ExperimentalFTIRBM) = UltraFast.Models.nspins(m.model)
UltraFast.Models.hyperparameters(m::ExperimentalFTIRBM) = merge(UltraFast.Models.hyperparameters(m.model), Dict("beta" => m.beta))
UltraFast.Models.identifier(m::ExperimentalFTIRBM) = "ExperimentalFTIRBM"

UltraFast.Models.get_flattened_weights(m::ExperimentalFTIRBM) = UltraFast.Models.get_flattened_weights(m.model)

function ExperimentalFTIRBM(settings; lattice::LatticeConfig)
    alpha = settings["alpha"]
    init = settings["weight_initializer"]
    beta = settings["beta"]
    
    if init == "complex_default_uniform"
        ExperimentalFTIRBM(lattice, alpha; init = UltraFast.Models.default_uniform(), beta=beta)
    elseif init == "real_default_uniform"
        ExperimentalFTIRBM(lattice, alpha; init = UltraFast.Models.real_default_uniform(), beta=beta)
    else
        error("Unknown weight initializer: $init")
    end
end