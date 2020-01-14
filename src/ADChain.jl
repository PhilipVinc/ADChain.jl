module ADChain

using ChainRulesCore
using Flux

export gradient

include("lib.jl")
include("pass.jl")

end # module
