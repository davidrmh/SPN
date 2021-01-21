#========================================
Methods for Leaf nodes
========================================#
function pdf(d::LeafNode, x::Union{Real, AbstractArray, NamedTuple})
    value = isa(x, NamedTuple) ? x[d.varname] : x
    Distributions.pdf(d.distribution, value)
end
