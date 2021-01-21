#========================================
Methods for Product nodes
========================================#

"""
Evaluate the pdf of a product node p for the value x
"""
function pdf(p::ProductNode, x::Union{Real, AbstractArray, NamedTuple})
    value = 1
    for i in eachindex(p.children)
        value = value * pdf(p.children[i], x)
    end
    value
end

"""
    sample!(node::ProductNode, dict:Dict)
Get one sample from a product node.
The argument dict is modified in-place when reaching a leaf node
that is a descendant  of `node`.

# Arguments
- `node::ProductNode` Decomposable product node.
- `dict::Dict` A dictionary whose keys are symbols and
value the random sample associated to that symbol.
"""
function sample!(node::ProductNode, dict::Dict{Any, Any})
    for child in node.children
        sample!(child, dict)
    end
end
