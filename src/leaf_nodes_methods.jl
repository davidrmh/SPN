#========================================
Methods for Distribution nodes
========================================#
function pdf(d::DistributionNode, x::Union{Real, AbstractArray, NamedTuple})
    value = isa(x, NamedTuple) ? x[d.varname] : x
    Distributions.pdf(d.distribution, value)
end

#========================================
Methods for Indicator nodes
========================================#
function pdf(d::IndicatorNode, x::Union{Real, AbstractArray, NamedTuple})
    x[d.varname]
end
