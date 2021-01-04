#=
TO DO:
    Create a git repository
    Implement a function to set the evidence
    Implement logpdf function
    Implement likelihood function
    Implement loglikelihood function
    Implement getscope function
    Implement sample function
=#
using Distributions

#========================================
Definition of Nodes types
========================================#

#This global variable is to assign an ID to each node
_idcounter = 0

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


#========================================
Methods for Abstract nodes
========================================#
"""
Add children to a parent node.
Children is an interable with Nodes.
This function is used for building a SPN using a top to bottom approach.
"""
function addchildren!(parent::AbstractNode, children::Array{<:AbstractNode})

    #Add the children to the parent
    push!(parent.children, children...)

    #Add the parent to each child
    for child in children
        push!(child.parents, parent)
    end
end

"""
Get the scope of a non-leaf node (sum or product node)
Returns an array with symbols
"""
function scope(node::Union{SumNode, ProductNode}, neg_string = "_NEG"::String)
    sc = []
    for child in node.children
        push!(sc, scope(child, neg_string))
    end
    return vcat(unique(sc)...)
end

"""
Get the scope for leaf nodes
Returns a symbol
"""
function scope(node::Union{DistributionNode, IndicatorNode}, neg_string = "_NEG"::String)
    stringsymbol = String(node.varname)
    #Check if is a negated variable
    substringloc = findfirst(uppercase(neg_string), uppercase(stringsymbol))
    if  substringloc!= nothing
        varsymbol = Symbol(stringsymbol[1:(substringloc[1] - 1)])
    else
        varsymbol = node.varname
    end
    varsymbol
end

"""
Get the variable names that are descendants of a non-leaf node (sum or product node)
Returns an array with symbols
"""
function variablenames(node::Union{SumNode, ProductNode})
    variables = []
    for child in node.children
        push!(variables, variablenames(child))
    end
    return vcat(unique(variables)...)
end

"""
Get the variable name for a leaf node
Returns a symbol
"""
function variablenames(node::Union{DistributionNode, IndicatorNode})
    node.varname
end

"""
INTERNAL USE ONLY
Get the descendants of a node
Modifies IN-PLACE a given array that will store the descendants
"""
function _descendants!(node::AbstractNode, array::AbstractArray, memory::AbstractArray)
    #IndicatorNodes and DistributionNodes
    #Just have one descendant (themselves)
    if isa(node, Union{IndicatorNode, DistributionNode})
        if !(node.id in memory)
            push!(array, node)
            push!(memory, node.id)
            println(string("Dist with id =", node.id, " is added"))
        end
        return
    end

    #By definition, a node is its own descendant
    if !(node.id in memory)
        push!(array, node)
        push!(memory, node.id)
        println(string("Child with id =", node.id, " is added"))
    end

    for child in node.children
        _descendants!(child, array, memory)
    end
    
end

"""
Get the descendants of a node
returns an array of nodes
For the inner work see internal function _descendants!
"""
function descendants(node::AbstractNode)
    array = []
    memory = []
    _descendants!(node, array, memory)
    array
end
    

#========================================
Methods for sum nodes
========================================#

"""
Evaluates the pdf of a sum node s for the value x
"""
function pdf(s::SumNode, x::Union{Real, AbstractArray, NamedTuple})
    value = 0
    for i in eachindex(s.children)
        value = value + s.weights[i] * pdf(s.children[i], x)
    end
    value
end

"""
Set weights for sum nodes
"""
function setweights!(node::SumNode, w::Array{Float64, 1})
    #Error handling
    if !isapprox(sum(w), 1)
        println("The sum of weights should be 1")
        return
    end
    if any(isless.(w, 0.0))
        println("You need to have positive weights")
        return
    end
    push!(node.weights, w...)
end

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

#========================================
Methods for Distribution nodes
========================================#
function pdf(d::DistributionNode, x::Union{Real, AbstractArray, NamedTuple})
    Distributions.pdf(d.distribution, x)
end

#========================================
Methods for Indicator nodes
========================================#
function pdf(d::IndicatorNode, x::Union{Real, AbstractArray, NamedTuple})
    x[d.varname]
end

#========================================
Some Examples
========================================#
function normalmixture(weights = [1/3, 1/3, 1/3], mu = [-5, -2, 2], sig = [0.5, 3, 1])
    #Create the sum node
    sumnode = SumNode([], [], weights)

    #Create each normal component
    components = Array{AbstractNode, 1}(undef, length(weights))
    for i in eachindex(mu)
        varname = Symbol(string("X", i))
        components[i] = DistributionNode(Distributions.Normal(mu[i], sig[i]), [], varname)
    end

    #Connect nodes
    addchildren!(sumnode, components)
    sumnode
end

function naivebayesmixture(weights = [[0.5, 0.2, 0.3], [0.6, 0.4], [0.9, 0.1], [0.3, 0.7], [0.2, 0.8]])
    s1 = SumNode([], [], weights[1]) #root
    s2 = SumNode([], [], weights[2])
    s3 = SumNode([], [], weights[3])
    s4 = SumNode([], [], weights[4])
    s5 = SumNode([], [], weights[5])

    p1 = ProductNode([], [])
    p2 = ProductNode([], [])
    p3 = ProductNode([], [])

    x1 = IndicatorNode([], :X1)
    x1_neg = IndicatorNode([], :X1_neg)
    x2 = IndicatorNode([], :X2)
    x2_neg = IndicatorNode([], :X2_neg)

    addchildren!(s1, [p1, p2, p3])
    addchildren!(p1, [s2, s4])
    addchildren!(p2, [s2, s5])
    addchildren!(p3, [s3, s5])
    addchildren!(s2, [x1, x1_neg])
    addchildren!(s3, [x1, x1_neg])
    addchildren!(s4, [x2, x2_neg])
    addchildren!(s5, [x2, x2_neg])

    s1
end

