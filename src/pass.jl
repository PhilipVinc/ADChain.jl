
function gradient_impl(ch::Type{Chain{Tl}}, x::Type{Tx}) where {Tl,Tx}
    # The number of layers
    N = length(Tl.parameters)

    # output of intermediate layers (x_n+1 = l_n(x_n))
    x_vars  = [Symbol("x$i") for i=1:N+1]
    x_vars[1] = Symbol("x")
    # pullback of intermediate layers (pb_n = pb(l_n))
    pb_vars = [Symbol("pb$i") for i=1:N]
    # sensitivity propagated by layers (ȳ_end+1 = 1)
    ȳ_vars  = [Symbol("ȳ$i") for i=1:N+1]
    # resulting gradient for layers
    ∇y_vars = [Symbol("∇y$i") for i=1:N]
    expr = Expr(:block, :($(ȳ_vars[end]) = 1))

    # Forward pass
    for i=1:N
        l = :(getfield(c.layers,$i))
        expr_i = :(($(x_vars[i+1]), $(pb_vars[i])) = rrule($l, $(x_vars[i])));
        push!(expr.args, expr_i)
    end

    # backward pass
    for i=N:-1:1
        expr_i = :(($(∇y_vars[i]), $(ȳ_vars[i])) = $(pb_vars[i])($(ȳ_vars[i+1])))
        push!(expr.args, expr_i)
    end

    # Construct the tuple holding all backward passes
    push!(expr.args, :(grad = tuple($(∇y_vars...))))
    push!(expr.args, :(return ($(x_vars[end]), grad)))

    return expr
end

@generated function gradient(c::Chain{C}, x::T) where {C,T}
    return gradient_impl(c, x)
end
