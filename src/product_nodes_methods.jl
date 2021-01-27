#========================================
Methods for Product nodes
========================================#

"""
    pdf(p::ProductNode, data::Union{Real, AbstractArray, NamedTuple}, params::Dict{Any, Any})

Evaluate the pdf of a product node.

# Arguments
- `p::ProductNode` A ProductNode object.

- `data::Union{Real, AbstractArray, NamedTuple, DataFrame}` Data.

- `params::Dict{Any, Any}` Dictionary created with the function `getparameters`.
Useful por calculating the gradients with Zygote.

"""
function pdf(p::ProductNode, data::Union{Real, AbstractArray, NamedTuple, DataFrame}, params::Dict{Any, Any})
    value = 1
    for i in eachindex(p.children)
        value = value .* pdf(p.children[i], data, params)
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
