include("common.jl")
using Dates
using Logging

@testset "gibbs-basic" begin
    nspins = 64
    alpha = 4

    M = nspins * alpha
    N = nspins

    W = rand(Float32, N, M) .- Float32(0.5)
    bias = rand(Float32, M) .- Float32(0.5)

    initial_visible = rand(Float32, N)

    @testset "gibbs_sampling" begin
        benchmark = @benchmark gibbs_sampling($initial_visible, $W, $bias, 2) seconds=1
        println(benchmark)
    end

    @testset "chromatic_rbm_gibbs_sampling" begin
        benchmark = @benchmark chromatic_rbm_gibbs_sampling(visible=$initial_visible, W=$W, b=$bias, steps=2) seconds=1 
        println(benchmark)

        output, metadata = chromatic_rbm_gibbs_sampling(visible=initial_visible, W=W, b=bias, steps=2) 
        @test length(output) == 1
        @test length(metadata) == 0
    end

    @testset "chromatic_rbm_gibbs_sampling - hidden output" begin
        benchmark = @benchmark chromatic_rbm_gibbs_sampling(visible=$initial_visible, W=$W, b=$bias, steps=2, sampling_settings=(save_hidden=true,)) seconds=1 
        
        println(benchmark)

        output, metadata = chromatic_rbm_gibbs_sampling(visible=initial_visible, W=W, b=bias, steps=2, sampling_settings=(save_hidden=true,))
        @test length(output) == 2
        @test length(metadata) == 0
    end

    @testset "chromatic_rbm_gibbs_sampling_mag0" begin
        benchmark = @benchmark chromatic_rbm_gibbs_sampling_mag0(visible=$initial_visible, W=$W, b=$bias, steps=2) seconds=1 
        
        println(benchmark)

        output, metadata = chromatic_rbm_gibbs_sampling_mag0(visible=initial_visible, W=W, b=bias, steps=2)
        @test length(output) == 1
        @test length(metadata) == 2
    end

    @testset "chromatic_rbm_gibbs_sampling_mag0 - hidden output" begin
        benchmark = @benchmark chromatic_rbm_gibbs_sampling_mag0(visible=$initial_visible, W=$W, b=$bias, steps=2, sampling_settings=(save_hidden=true,)) seconds=1 
        
        println(benchmark)

        output, metadata = chromatic_rbm_gibbs_sampling_mag0(visible=initial_visible, W=W, b=bias, steps=2, sampling_settings=(save_hidden=true,))
        @test length(output) == 2
        @test length(metadata) == 2
    end
end