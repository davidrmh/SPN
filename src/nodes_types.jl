#========================================
Definition of Nodes types
========================================#
abstract type AbstractNode end

mutable struct ProductNode <: AbstractNode
    children::AbstractVector{Union{AbstractNode,AbstractArray}}
    parents::AbstractVector{Union{AbstractNode,AbstractArray,UndefInitializer}}
    id::Int64
    copyof::Int64
end

mutable struct SumNode <: AbstractNode
    children::AbstractVector{Union{AbstractNode,AbstractArray}}
    parents::AbstractVector{Union{AbstractNode,AbstractArray,UndefInitializer}}
    weights::Array{Float64,1}
    id::Int64
    copyof::Int64
end

mutable struct DistributionNode <: AbstractNode
    distribution::Distribution
    parents::AbstractVector{Union{SumNode,ProductNode,AbstractArray}}
    varname::Symbol
    id::Int64
    copyof::Int64
end

mutable struct IndicatorNode <: AbstractNode
    parents::AbstractVector{Union{SumNode,ProductNode,AbstractArray}}
    varname::Symbol
    id::Int64
    copyof::Int64
end


#=
Outter constructors
These constructors are the ones that must be used.
They automatically keep a track of the number of
instances created for each type of node
=#
"""
    ProductNode(children, parents)

Create a sum node. With this signature, fields `id` and `copyof` are handled internally.
For root nodes, set parents = [undef].

    ProductNode(children, parents, id, copyof)
Create a product node with a given `id`.
The field `copyof` means the node is a copy of the node whose `id` is `copyof`
When a node is not a copy, `copyof` is set to `0`

# Arguments
- `children::AbstractVector{Union{AbstractNode,AbstractArray}}` Children nodes.
- `parents::AbstractVector{Union{AbstractNode,AbstractArray,UndefInitializer}}` Parent nodes.
- `id::Int64` id for unique identification.
- `copyof::Int64` id of the node that is beeing copied.

# Supertype Hierarchy
ProductNode <: AbstractNode <: Any
"""
function ProductNode(children, parents)
    global _idcounter
    _idcounter = _idcounter + 1
    ProductNode(children, parents, _idcounter, 0)
end

"""
    SumNode(children, parents)

Create a sum node. With this signature, id field is handled internally.
For root nodes, set parents = [undef].

    SumNode(children, parents, id, copyof)
Create a sum node with a given id.
The field `copyof` means the node is a copy of the node whose `id` is `copyof`
When a node is not a copy, `copyof` is set to `0`


# Arguments
- `children::AbstractVector{Union{AbstractNode,AbstractArray}}` Children nodes.
- `parents::AbstractVector{Union{AbstractNode,AbstractArray,UndefInitializer}}` Parent nodes.
- `id::Int64` id for unique identification.
- `copyof::Int64` id of the node that is beeing copied.

# Supertype Hierarchy
SumNode <: AbstractNode <: Any
"""
function SumNode(children, parents, weights)
    global _idcounter
    _idcounter = _idcounter + 1
    SumNode(children, parents, weights, _idcounter, 0)
end

"""
    DistributionNode(distribution, parents, varname)

Create a distribution node. With this signature, id field is handled internally.

    DistributionNode(distribution, parents, varname, id, copyof)

Create a distribution node with a given id.
The field `copyof` means the node is a copy of the node whose `id` is `copyof`
When a node is not a copy, `copyof` is set to `0`

# Arguments
- `distribution::Distribution` a distribution object from Distributions package.
- `parents::AbstractVector{Union{SumNode,ProductNode,AbstractArray}}` Parent nodes.
- `varname::Symbol` Variable name (see examples).
- `id::Int64` id for unique identification.
- `copyof::Int64` id of the node that is beeing copied.

# Supertype Hierarchy
DistributionNode <: AbstractNode <: Any

# Examples
```jldoctest

Create a distribution node with no parents and representing the random variable
X1 with standard normal distribution.

julia> normal = DistributionNode(Distributions.Normal(), [], :X1);
```
"""
function DistributionNode(distribution, parents, varname)
    global _idcounter
    _idcounter = _idcounter + 1
    DistributionNode(distribution, parents, varname, _idcounter, 0)
end

"""
    IndicatorNode(parents, varname)

Create an indicator node. With this signature, id field is handled internally.

    IndicatorNode(parents, varname, id, copyof)

Create an indicator node with a given id.
The field `copyof` means the node is a copy of the node whose `id` is `copyof`
When a node is not a copy, `copyof` is set to `0`


# Arguments
- `parents::AbstractVector{Union{SumNode,ProductNode,AbstractArray}}` Parent nodes.
- `varname::Symbol` Variable name (see examples).
- `id::Int64` id for unique identification.
- `copyof::Int64` id of the node that is beeing copied.

# Supertype Hierarchy
IndicatorNode <: AbstractNode <: Any

# Examples
```jldoctest

Create an indicator node with no parents and representing the indicator
for the third possible value of the variable X1.

julia>indicator = IndicatorNode([], :X1_3);
```
"""
function IndicatorNode(parents, varname)
    global _idcounter
    _idcounter = _idcounter + 1
    IndicatorNode(parents, varname, _idcounter, 0)
end
