
function rrule_chain_impl(ch::Type{Chain{Tl}}, x::Type{Tx}) where {Tl,Tx}
    # The number of layers
    N = length(Tl.parameters)

    # output of intermediate layers (x_n+1 = l_n(x_n))
    x_vars  = [Symbol("Y$(i-1)") for i=1:N+1]
    x_vars[1] = Symbol("x")
    # pullback of intermediate layers (pb_n = pb(l_n))
    pb_vars = [Symbol("∂Ȳ$i") for i=1:N]

    expr = Expr(:block)
    # Forward pass
    for i=1:N
        l = :(getfield(c.layers,$i))
        expr_i = :(($(x_vars[i+1]), $(pb_vars[i])) = rrule($l, $(x_vars[i])));
        push!(expr.args, expr_i)
    end

    # Work around closures not working in generated funcs
    push!(expr.args, :(chain_pb = PbClosure(tuple($(pb_vars...)))))

    push!(expr.args, :(return ($(x_vars[end]), chain_pb)))

    return expr
end

@generated function ChainRulesCore.rrule(c::Chain{C}, x::T) where {C,T}
    return rrule_chain_impl(c, x)
end
