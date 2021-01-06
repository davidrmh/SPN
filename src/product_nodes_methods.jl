#========================================
Methods for Product nodes
========================================#

"""
Evaluates the pdf of a product node p for the value x
"""
function pdf(p::ProductNode, x::Union{Real, AbstractArray, NamedTuple})
    value = 1
    for i in eachindex(p.children)
        value = value * pdf(p.children[i], x)
    end
    value
end