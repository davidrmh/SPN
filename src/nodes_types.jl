#========================================
Definition of Nodes types
========================================#
abstract type AbstractNode end

struct ProductNode <: AbstractNode
    children::AbstractVector{Union{AbstractNode, AbstractArray}}
    parents::AbstractVector{Union{AbstractNode, AbstractArray}}
    id::Integer
end

struct SumNode <: AbstractNode
    children::AbstractVector{Union{AbstractNode, AbstractArray}}
    parents::AbstractVector{Union{AbstractNode, AbstractArray}}
    weights::Array{Float64, 1}
    id::Integer
end

struct DistributionNode <: AbstractNode
    distribution::Distribution
    parents::AbstractVector{Union{SumNode, ProductNode, AbstractArray}}
    varname::Symbol
    id::Integer
end

struct IndicatorNode <: AbstractNode
    parents::AbstractVector{Union{SumNode, ProductNode, AbstractArray}}
    varname::Symbol
    id::Integer
end


#=
Outter constructors
These constructors are the one that must be used.
They automatically keep a track of the number of
instances created for each type of node
=#
function ProductNode(children, parents)
    global _idcounter
    _idcounter = _idcounter + 1
    ProductNode(children, parents, _idcounter)
end

function SumNode(children, parents, weights)
    global _idcounter
    _idcounter = _idcounter + 1
    SumNode(children, parents, weights, _idcounter)
end

function DistributionNode(distribution, parents, varname)
    global _idcounter
    _idcounter = _idcounter + 1
    DistributionNode(distribution, parents, varname, _idcounter)
end

function IndicatorNode(parents, varname)
    global _idcounter
    _idcounter = _idcounter + 1
    IndicatorNode(parents, varname, _idcounter)
end
