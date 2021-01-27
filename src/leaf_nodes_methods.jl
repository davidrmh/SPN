#========================================
Methods for Leaf nodes
========================================#
"""
    pdf(d::LeafNode, data::Union{Real, AbstractArray, NamedTuple}, params::Dict{Any, Any})

Evaluate the pdf of a leaf node.

# Arguments
- `d::LeafNode` A LeafNode (Distribution or Indicator node) object.

- `data::Union{Real, AbstractArray, NamedTuple}` Data.

- `params::Dict{Any, Any}` Dictionary created with the function `getparameters`.
Useful por calculating the gradients with Zygote.

"""
function pdf(d::LeafNode, data::Union{Real, AbstractArray, NamedTuple}, params::Dict{Any, Any})
    value = isa(data, NamedTuple) ? data[d.varname] : data
    Distributions.pdf(typeof(d.distribution)(params[d.id]...), value)
end

"""
    sample!(node::LeafNode, dict:Dict)
Get one sample from a leaf node.
Modify in-place the argument dict.

# Arguments
- `node::LeafNode` Leaf node with a distribution associated to it.
- `dict::Dict` A dictionary whose keys are symbols `node.varname` and
value the random sample from `node.distribution`.
"""
function sample!(node::LeafNode, dict::Dict{Any, Any})
    #Only sample when not previously
    #sampled for the random variable associated to this node
    if !haskey(dict, node.varname)
        #rand returns an array, that's why the [1]
        r = Distributions.rand(node.distribution, 1)[1]
        dict[node.varname] = r
    end
end

function getparameters(dist::Normal, logspace = true::Bool)
    par = [params(dist)...]
end
