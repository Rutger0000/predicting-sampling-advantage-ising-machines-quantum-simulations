include("chromatic_gibbs_rbm.jl")
"""
    chromatic_rbm_sampler(nspins, W, b, steps, precision, sampling_settings)

This function performs Gibbs sampling for the Ising Machine.

## Arguments
- `nspins`: The number of spins in the system.
- `W`: Part of weight matrix for the Ising model representing RBM
- `b`: Part of bias vector for the Ising model representing RBM
- `steps`: The number of steps to perform in each sweep.
- `precision`: The precision of the calculations. Default is Float32.
- `sampling_settings`: A named tuple containing the settings for the sampling.
  - `all_up`: If true, the initial state is all spins up. Default is false.
  - `mag0`: If true, the hidden spins are not sampled. Default is false.
  - `thermalization`: The number of thermalization steps. Default is 0.
  - `sweeps`: The number of sweeps to perform. Default is 1.
  - `save_hidden`: If true, the hidden spins are saved. Default is false.

## Returns
- `elapsed_time_sampling`: The elapsed time for the Gibbs sampling.
- `states`: The sampled states. A named tuple containing the sampled visible
  spins, (optionally, only mag0) the hidden spins, (optionally, only mag0) the
  actual number of thermalization and sampling steps, elapsed_time_sampling
"""
function chromatic_rbm_sampler(;nspins, W, b, steps::Int, precision=Float32, sampling_settings::NamedTuple = (;))
    all_up = get(sampling_settings, :all_up, false)
    mag0 = get(sampling_settings, :mag0, false)

    if all_up
        initial_state = ones(precision, nspins)
    else
        initial_state = sign.(rand(precision, nspins) .- 0.5) 
    end

    if mag0
        elapsed_time_sampling = @elapsed output, metadata = chromatic_rbm_gibbs_sampling_mag0(visible=initial_state, 
                                                                            W=W, 
                                                                            b=b, 
                                                                            steps=steps, 
                                                                            precision=precision,
                                                                            sampling_settings=sampling_settings)
    else
        elapsed_time_sampling = @elapsed output, metadata = chromatic_rbm_gibbs_sampling(visible=initial_state, 
                                                                            W=W, 
                                                                            b=b, 
                                                                            steps=steps, 
                                                                            precision=precision,
                                                                            sampling_settings=sampling_settings)
    end
    
    return output, (metadata..., elapsed_time_sampling=elapsed_time_sampling)
end
