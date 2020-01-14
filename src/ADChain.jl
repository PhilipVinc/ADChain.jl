module ADChain

using ChainRulesCore
using ChainRules
using GeneralizedGenerated
using Flux

export gradient

include("lib.jl")
include("closure_pullback.jl")
include("pass.jl")

end # module
