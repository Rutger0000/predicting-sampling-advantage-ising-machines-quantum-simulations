include("translation_invariance/translation_invariance.jl")

using DelimitedFiles
using DataFrames
using CSV

import .TranslationInvarianceTools: WeightTranslator, weights_from

# Note the following code is only for REAL weights
weights = DataFrame(CSV.File("models/all_weights_converted.tsv", delim="\t"))

for weight in eachrow(weights)
    L = weight["L"]
    alpha = weight["alpha"]
    nonsquare = weight["nonsquare"]

    # extend the weights by doubling hidden units
    full = weight["full"]
    
    directory = weight["directory"]
    nspins = L*L

    println("Converting weights for size $L and alpha $alpha, nonsquare: $nonsquare, full: $full")

    translator = WeightTranslator(nspins, alpha, L, L)

    addition_b = full ? "_ti_b_full" : "_ti_b" 
    addition_W = full ? "_ti_W_full" : "_ti_W" 

    path_b = "data/models/$directory/W_RBM_$(nspins)_$(alpha)$(addition_b).csv"
    path_W = "data/models/$directory/W_RBM_$(nspins)_$(alpha)$(addition_W).csv"

    # check whether file already exists
    if isfile(path_b) && isfile(path_W)
        println("IGNORED Weights for size $L and alpha $alpha already exist")
        continue
    end

    mkpath("data/models/$directory")

    try 
        WRBM = readdlm("models/$directory/$(nspins)_$(alpha)_weights.txt", Float64)[:]
        
        W, b = weights_from(translator, WRBM)

        W = real.(W)
        b = real.(b)

        if full
            W = hcat(W, W)
            b = vcat(b, b)
        end

        CSV.write(path_b, Tables.table(b), writeheader=false)
        CSV.write(path_W, Tables.table(W), writeheader=false)
    catch e
        println("$(nspins)_$(alpha) not found and $e")
    end

end
