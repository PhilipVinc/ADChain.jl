
function ChainRulesCore.rrule(d::Flux.Dense, x)
    θ = d.W*x + d.b
    Y, ∂Ȳ = frule(broadcast, d.σ, θ)

    function dense_pullback(ȳ)
        ΔȲ = ∂Ȳ(nothing, nothing, ȳ)

        ∂W = InplaceableThunk(
            @thunk(ΔȲ*transpose(x)),
            x̄->ΔȲ*transpose(x)+x̄
        )
        ∂b = InplaceableThunk(
            @thunk(ΔȲ),
            x̄->ΔȲ+x̄
        )
        ∂σ = Zero()

        ∂dense = (W = ∂W, b = ∂b, σ = ∂σ)

        return (∂dense, @thunk(transpose(d.W)*ΔȲ))
    end
    return Y, dense_pullback
end
