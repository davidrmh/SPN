#========================================
Methods for Leaf nodes
========================================#
function pdf(d::LeafNode, x::Union{Real, AbstractArray, NamedTuple})
    value = isa(x, NamedTuple) ? x[d.varname] : x
    Distributions.pdf(d.distribution, value)
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
