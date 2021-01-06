#========================================
Methods for sum nodes
========================================#

"""
Evaluates the pdf of a sum node s for the value x
"""
function pdf(s::SumNode, x::Union{Real, AbstractArray, NamedTuple})
    value = 0
    for i in eachindex(s.children)
        value = value + s.weights[i] * pdf(s.children[i], x)
    end
    value
end

"""
Set weights for sum nodes
"""
function setweights!(node::SumNode, w::Array{Float64, 1})
    #Error handling
    if !isapprox(sum(w), 1)
        println("The sum of weights should be 1")
        return
    end
    if any(isless.(w, 0.0))
        println("You need to have positive weights")
        return
    end
    push!(node.weights, w...)
end
