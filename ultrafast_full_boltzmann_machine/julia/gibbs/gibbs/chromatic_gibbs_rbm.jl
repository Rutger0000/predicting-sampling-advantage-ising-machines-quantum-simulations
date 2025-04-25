# Implementations of chromatic Gibbs sampling for RBM.
using Random

"""
    calculate_hidden(visible, W, bias, M, precision=Float32)

Update the hidden spins using block/chromatic gibbs sampling.

# Arguments
- `visible::AbstractVector`: The visible spins
- `W::AbstractMatrix`: The weight matrix
- `bias::AbstractVector`: The bias vector for hidden spins
- `M::Int`: The number of hidden spins.
- `precision::Type`: The precision type.

# Returns
- `AbstractVector`: The updated hidden spins vector.
"""
function calculate_hidden(visible, W, bias, M, precision=Float32)
    activation = tanh.((W' * visible) .+ bias)
    random_uniform = rand(precision, M) .* 2 .- 1
    return sign.(activation .- random_uniform)
end

"""
    calculate_visible(hidden, W, N, precision=Float32)

Update the visible spins using block/chromatic gibbs sampling.

# Arguments
- `hidden::AbstractVector`: The hidden spins
- `W::AbstractMatrix`: The weight matrix
- `N::Int`: The number of visible spins.
- `precision::Type`: The precision type.

# Returns
- `AbstractVector`: The updated visible spins vector.
"""
function calculate_visible(hidden, W, N, precision=Float32)
    activation = tanh.(W * hidden)
    random_uniform = rand(precision, N) .* 2 .- 1
    return sign.(activation .- random_uniform)
end

"""
    gibbs_sampling(visible, W, bias, nsteps)

Perform Gibbs sampling for a given number of steps.

# Arguments
- `visible::AbstractVector`: The initial visible spins
- `W::AbstractMatrix`: The weight matrix
- `bias::AbstractVector`: The bias vector for hidden spins
- `nsteps::Int`: The number of Gibbs sampling steps.

# Returns
- `AbstractVector`: The final visible spins vector after sampling.
"""
function gibbs_sampling(visible, W, bias, nsteps)
    N, M = size(W)

    for _ in 1:nsteps
        hidden = calculate_hidden(visible, W, bias, M)
        visible = calculate_visible(hidden, W, N)
    end
    return visible
end

"""
    chromatic_rbm_gibbs_sampling(;visible, W, b, steps::Int, precision=Float32, sampling_settings::NamedTuple = (;))

Perform chromatic Gibbs sampling of RBM with additional settings.

# Keyword Arguments
- `visible::AbstractVector`: The initial visible spins
- `W::AbstractMatrix`: The weight matrix
- `b::AbstractVector`: The bias vector for hidden spins
- `steps::Int`: The number of Gibbs sampling steps.
- `precision::Type`: The precision type.
- `sampling_settings::NamedTuple`: Additional settings for sampling.

# Returns
- `NamedTuple`: A named tuple containing the sampled visible spins and optionally the hidden spins. 
- `NamedTuple`: A named tuple containing the metadata of the sampling.
"""
function chromatic_rbm_gibbs_sampling(;visible, W, b, steps::Int, precision=Float32, sampling_settings::NamedTuple = (;))
    N, M = size(W)

    # load settings
    sweep = get(sampling_settings, :sweep, 1)
    thermalization = get(sampling_settings, :thermalization, 0)

    # start --- saving code
    save_hidden = get(sampling_settings, :save_hidden, false)

    if save_hidden
        all_hidden = Matrix{precision}(undef, M, steps)
    end
    all_visible = Matrix{precision}(undef, N, steps)
    # end   --- saving code

    hidden = Vector{precision}(undef, M)

    for _ in 1:thermalization
        for _ in 1:sweep
            hidden = calculate_hidden(visible, W, b, M, precision)
            visible = calculate_visible(hidden, W, N, precision)
        end
    end

    for i in 1:steps
        for _ in 1:sweep
            hidden = calculate_hidden(visible, W, b, M, precision)
            visible = calculate_visible(hidden, W, N, precision)
        end

        all_visible[:, i] .= visible
        # start --- saving code
        if save_hidden
            all_hidden[:, i] .= hidden
        end
        # end   --- saving code
    end

    # start --- saving code
    if save_hidden
        return (visible=all_visible, hidden=all_hidden), (;)
    end
    return (visible=all_visible,), (;)
end

"""
    chromatic_rbm_gibbs_sampling_mag0(;visible, W, b, steps::Int, precision=Float32, sampling_settings::NamedTuple = (;))

Perform chromatic gibbs sampling of RBM with magnetization constraint (sum of visible spins equals zero).

# Keyword Arguments
- `visible::AbstractVector`: The initial visible spins
- `W::AbstractMatrix`: The weight matrix
- `b::AbstractVector`: The bias vector for hidden spins
- `steps::Int`: The number of Gibbs sampling steps.
- `precision::Type`: The precision type.
- `sampling_settings::NamedTuple`: Additional settings for sampling.

# Returns
- `NamedTuple`: A named tuple containing the sampled visible spins, optionally the hidden spins, and the actual number of thermalization and sampling steps.
- `NamedTuple`: A named tuple containing the metadata of the sampling, including the actual number of thermalization and sampling steps.
"""
function chromatic_rbm_gibbs_sampling_mag0(;visible, W, b, steps::Int, precision=Float32, sampling_settings::NamedTuple = (;))
    N, M = size(W)

    # load settings
    sweep = get(sampling_settings, :sweep, 1)
    thermalization = get(sampling_settings, :thermalization, 0)

    # start --- saving code
    save_hidden = get(sampling_settings, :save_hidden, false)

    if save_hidden
        all_hidden = Matrix{precision}(undef, M, steps)
    end
    all_visible = Matrix{precision}(undef, N, steps)
    # end   --- saving code

    actual_thermalization_steps = 0
    actual_sampling_steps = 0
    hidden = Vector{precision}(undef, M)

    for _ in 1:thermalization
        for _ in 1:sweep
            while true
                hidden = calculate_hidden(visible, W, b, M, precision)
                visible = calculate_visible(hidden, W, N, precision)
                sum_visible = sum(visible)
                actual_thermalization_steps += 1
                if sum_visible == 0
                    break
                end
            end
        end
    end

    for i in 1:steps
        for _ in 1:sweep
            while true
                hidden = calculate_hidden(visible, W, b, M, precision)
                visible = calculate_visible(hidden, W, N, precision)
                sum_visible = sum(visible)
                actual_sampling_steps += 1
                if sum_visible == 0
                    break
                end
            end
        end
        all_visible[:, i] .= visible
        # start --- saving code
        if save_hidden
            all_hidden[:, i] .= hidden
        end
        # end   --- saving code
    end
    if save_hidden
        return (visible=all_visible, hidden=all_hidden), (actual_thermalization_steps=actual_thermalization_steps, actual_sampling_steps=actual_sampling_steps)
    end
    return (visible=all_visible,), (actual_thermalization_steps=actual_thermalization_steps, actual_sampling_steps=actual_sampling_steps)
end

using BenchmarkTools
function test_gibbs_sampling(alpha, nspins, nsteps=200)
    M = nspins * alpha
    N = nspins

    W = rand(Float32, N, M) .- Float32(0.5)
    bias = rand(Float32, M) .- Float32(0.5)

    initial_visible = rand(Float32, N)

    benchmark = @benchmark gibbs_sampling($initial_visible, $W, $bias, $nsteps)

    return mean(benchmark), std(benchmark)
end

