struct PbClosure{T<:Tuple}
    pbs::T
end

function PbClosure_impl(x::Type{PbClosure{T}}, y::Type{Ty}) where {T,Ty}
    # Number of layers (pullbacks)
    N = length(T.parameters)

    pb_vars = [Symbol("∂Ȳ$i") for i=1:N]
    # sensitivity propagated by layers (ȳ_end+1 = 1)
    ȳ_vars  = [Symbol("ȳ$i") for i=1:N+1]
    ȳ_vars[end] = Symbol("y")
    # resulting gradient for layers
    ∇y_vars = [Symbol("∇y$i") for i=1:N]

    expr = Expr(:block)

    for i=N:-1:1
        pb_i = :(getfield(pb.pbs, $i))
        expr_i = :(($(∇y_vars[i]), $(ȳ_vars[i])) = $(pb_i)($(ȳ_vars[i+1])))
        push!(expr.args, expr_i)
    end

    push!(expr.args, :(∂chain = tuple($(∇y_vars...))))
    push!(expr.args, :(return (∂chain, $(ȳ_vars[1]))))

    return expr
end

@generated function (pb::PbClosure{A})(y::B) where {A,B}
    return PbClosure_impl(pb, y)
end
