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

"""
    iscomplete(root)
Determine if a SPN is complete.

# Arguments
- `root::AbstractNode` root node of the SPN.
"""
function iscomplete(root::AbstractNode)
    #Get all sum nodes
    sumnodes = filter_by_type(root, SumNode)

    #All the children must have the same scope
    #So the set difference of their scope, should be empty
    for node in sumnodes
        if mapreduce(ch -> [scope(ch)], setdiff, node.children) != []
            return false
        end
    end
    true
end

"""
    sample!(node::SumNode, dict:Dict)
Get one sample from a sum node.
The argument dict is modified in-place when reaching a leaf node
that is a descendant  of `node`.

# Arguments
- `node::SumNode` Complete and locally normalized sum node.
- `dict::Dict` A dictionary whose keys are symbols and
value the random sample associated to that symbol.
"""
function sample!(node::SumNode, dict::Dict{Any, Any})
    #Select the edge to follow
    z = Distributions.Categorical(node.weights ./ sum(node.weights))
    #rand returns an array, that's why the [1]
    idx = Distributions.rand(z, 1)[1]
    sample!(node.children[idx], dict)
end
