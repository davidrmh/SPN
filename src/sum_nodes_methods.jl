#========================================
Methods for sum nodes
========================================#

"""
    pdf(s::SumNode, data::Union{Real, AbstractArray, NamedTuple},
    params::Dict{Any, Any}, logspace = false::Bool)

Evaluate the pdf of a sum node.
# Arguments
- `s::SumNode` A SumNode object.

- `data::Union{Real, AbstractArray, NamedTuple, DataFrame}` Data.

- `params::Dict{Any, Any}` Dictionary created with the function `getparameters`.
Useful por calculating the gradients with Zygote.

- `logspace::Bool` Indicates if the parameters are in the log space (true) or
in the original space (false).
"""
function pdf(s::SumNode, data::Union{Real, AbstractArray, NamedTuple, DataFrame},
    params::Dict{Any, Any}, logspace = false::Bool)
    value = 0
    for i in eachindex(s.children)
        #params[s.id] stores the weights
        if logspace
            #If parameters come from logspace
            #return them to the original space
            value = value .+ exp(params[s.id][i]) .* pdf(s.children[i], data, params, logspace)
        else
            value = value .+ params[s.id][i] .* pdf(s.children[i], data, params, logspace)
        end
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
"""
    logpdf(node::SumNode, data::Union{Real, AbstractArray, NamedTuple, DataFrame},
        params::Dict{Any, Any})

"""
function logpdf(node::SumNode, data::Union{Real, AbstractArray, NamedTuple, DataFrame},
    params::Dict{Any, Any})

    #log of each weight
    logweights = log.(params[node.id])

    #logpdf of each children
    logchildren = map(child -> logpdf(child, data, params), node.children)
    #Convert into a num_children x num_observations array
    logchildren = transpose( hcat(logchildren...) )

    #Apply (stable) logsumexp
    plus = logweights .+ logchildren
    m = maximum(plus, dims = 1)
    m .+ log.( sum( exp.(plus .- m), dims = 1 ) )
end
