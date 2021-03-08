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
    setweights!(node::SumNode, w::Array{Float64, 1})
Set weights for sum nodes. The modification is done in-place and previous
weights are deleted.

# Arguments
- `node::SumNode` A sum node.

- `w::Array{Float64, 1}` Array with the new weights.
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
    #Delete previous weights
    node.weights = []
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
    logpdf(node::SumNode, data::DataFrame, params::Dict{Any, Any},
    margvar=[:none]::Array{Symbol, 1})

Calculate the logpdf of a sum node, applying the logsumexp trick.

Return an array with dimension 1 x size(data)[1]. That is, an array of size
1 x number of observations in the dataset.

# Arguments
- `node::SumNode` A sum node.

- `data::DataFrame` Data frame with the observations. The column used is the
one that contains the header corresponding to the field `node.varname`.

- `params::Dict{Any, Any}` Dictionary with the parameters. This dictionary is
created with the function `getparameters`.

- `margvar=[:none]::Array{Symbol, 1}` Array with the symbols of the variables
to be marginalized.
"""
function logpdf(node::SumNode, data::DataFrame, params::Dict{Any, Any},
    margvar=[:none]::Array{Symbol, 1})

    #log of each weight
    logweights = log.(params[node.id])

    #logpdf of each children
    nobs = size(data)[1]
    nchildren = length(node.children)
    logchildren = zeros((nchildren, nobs))
    for i in eachindex(node.children)
        logchildren[i, :] = logpdf(node.children[i], data, params, margvar)
    end

    #Apply (stable) logsumexp
    plus = logweights .+ logchildren
    m = maximum(plus, dims = 1)
    result = m .+ log.( sum( exp.(plus .- m), dims = 1 ) )
    #This is to avoid NaN that come up when having Inf - Inf
    #This situations arise when all the children have logpdf equal to -Inf
    #equivalently pdf equal to 0.
    result[isnan.(result)] .= -Inf
    result
end

"""
    logpdf!(node::SumNode, data::DataFrame,
    params::Dict{Any, Any}, memory::Dict{Any, Any}, margvar=[:none]::Array{Symbol, 1})

Calculate the logpdf of a sum node, applying the logsumexp trick.

Return an array with dimension 1 x size(data)[1]. That is, an array of size
1 x number of observations in the dataset.

Modify `in-place` the dictionary `memory`.

This `logpdf!` is faster than `logpdf`.

# Arguments
- `node::SumNode` A sum node.

- `data::DataFrame` Data frame with the observations. The column used is the
one that contains the header corresponding to the field `node.varname`.

- `params::Dict{Any, Any}` Dictionary with the parameters. This dictionary is
created with the function `getparameters`.

- `memory::Dict{Any, Any}` Dictionary that stores the logpdf of each node.
Each key corresponds to the `id` field associated to a particular node.
The value is the logpdf of the corresponding node.

- `margvar=[:none]::Array{Symbol, 1}` Array with the symbols of the variables
to be marginalized.
"""
function logpdf!(node::SumNode, data::DataFrame,
    params::Dict{Any, Any}, memory::Dict{Any, Any}, margvar=[:none]::Array{Symbol, 1})

    #If the logpdf for `node` has been calculated
    #use the stored value
    if haskey(memory, node.id)
        return memory[node.id]
    end

    #log of each weight
    logweights = log.(params[node.id])

    #logpdf of each children
    nobs = size(data)[1]
    nchildren = length(node.children)
    logchildren = zeros((nchildren, nobs))
    for i in eachindex(node.children)
        child = node.children[i]
        #Calculate the child's logpdf if it hasn't been
        #previously calculated
        logchildren[i, :] = !haskey(memory, child.id) ?
        logpdf!(child, data, params, memory, margvar) : memory[child.id]
    end
    
    #Apply (stable) logsumexp
    plus = logweights .+ logchildren
    m = maximum(plus, dims = 1)
    nodelogpdf = m .+ log.( sum( exp.(plus .- m), dims = 1 ) )
    #This is to avoid NaN that come up when having Inf - Inf
    #These situations arise when all the children have logpdf equal to -Inf
    #equivalently pdf equal to 0.
    #Unfortunately, this line causes conflicts with Zygote
    nodelogpdf[isnan.(nodelogpdf)] .= -Inf
    #Add to memory
    memory[node.id] = nodelogpdf
    nodelogpdf
end
