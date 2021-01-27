#========================================
Methods for Leaf nodes
========================================#
"""
    pdf(d::LeafNode, data::Union{Real, AbstractArray, NamedTuple, DataFrame}, params::Dict{Any, Any})

Evaluate the pdf of a leaf node.

# Arguments
- `d::LeafNode` A LeafNode (Distribution or Indicator node) object.

- `data::Union{Real, AbstractArray, NamedTuple, DataFrame}` Data.

- `params::Dict{Any, Any}` Dictionary created with the function `getparameters`.
Useful por calculating the gradients with Zygote.

"""
function pdf(d::LeafNode, data::Union{Real, AbstractArray, NamedTuple, DataFrame}, params::Dict{Any, Any})
    if isa(data, DataFrame)
        vals = data[!, d.varname]
    #Array with named tuples
    elseif isa(data[1], NamedTuple)
        vals = [tup[d.varname] for tup in data]
    else isa(data, NamedTuple)
        vals = data[d.varname]
    end
    Distributions.pdf.(typeof(d.distribution)(params[d.id]...), vals)
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
