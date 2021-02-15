#========================================
Methods for Product nodes
========================================#

"""
    pdf(p::ProductNode, data::Union{Real, AbstractArray, NamedTuple},
    params::Dict{Any, Any})

Evaluate the pdf of a product node.

# Arguments
- `p::ProductNode` A ProductNode object.

- `data::Union{Real, AbstractArray, NamedTuple, DataFrame}` Data.

- `params::Dict{Any, Any}` Dictionary created with the function `getparameters`.
Useful por calculating the gradients with Zygote.

- `logspace::Bool` Indicates if the parameters are in the log space (true) or
in the original space (false).
"""
function pdf(p::ProductNode, data::Union{Real, AbstractArray, NamedTuple, DataFrame},
    params::Dict{Any, Any}, logspace = false::Bool)
    value = 1
    for i in eachindex(p.children)
        value = value .* pdf(p.children[i], data, params, logspace)
    end
    value
end

"""
    sample!(node::ProductNode, dict:Dict)
Get one sample from a product node.
The argument dict is modified in-place when reaching a leaf node
that is a descendant  of `node`.

# Arguments
- `node::ProductNode` Decomposable product node.
- `dict::Dict` A dictionary whose keys are symbols and
value the random sample associated to that symbol.
"""
function sample!(node::ProductNode, dict::Dict{Any, Any})
    for child in node.children
        sample!(child, dict)
    end
end

"""
    logpdf(node::ProductNode, data::DataFrame, params::Dict{Any, Any})

Calculate the logpdf of a product node.

Return an array with dimension 1 x size(data)[1]. That is, an array of size
1 x number of observations in the dataset.

# Arguments
- `node::ProductNode` A product node.

- `data::DataFrame` Data frame with the observations. The column used is the
one that contains the header corresponding to the field `node.varname`.

- `params::Dict{Any, Any}` Dictionary with the parameters. This dictionary is
created with the function `getparameters`.
"""
function logpdf(node::ProductNode, data::DataFrame, params::Dict{Any, Any})

    #logpdf of each children
    logchildren = []
    for i in eachindex(node.children)
        childlogpdf = logpdf(node.children[i], data, params)
        push!(logchildren, [childlogpdf...])
    end
    #Array of n_children X n_obs
    logchildren = transpose(reduce(hcat, logchildren))
    #Array of 1 x n_obs
    sum(logchildren, dims = 1)
end

"""
    logpdf!(node::ProductNode, data::DataFrame,
    params::Dict{Any, Any}, memory::Dict{Any, Any})

Calculate the logpdf of a product node.

Return an array with dimension 1 x size(data)[1]. That is, an array of size
1 x number of observations in the dataset.

Modify `in-place` the dictionary `memory`.

This `logpdf!` is faster than `logpdf`.

# Arguments
- `node::ProductNode` A product node.

- `data::DataFrame` Data frame with the observations. The column used is the
one that contains the header corresponding to the field `node.varname`.

- `params::Dict{Any, Any}` Dictionary with the parameters. This dictionary is
created with the function `getparameters`.

- `memory::Dict{Any, Any}` Dictionary that stores the logpdf of each node.
Each key corresponds to the `id` field associated to a particular node.
The value is the logpdf of the corresponding node.
"""
function logpdf!(node::ProductNode, data::DataFrame,
    params::Dict{Any, Any}, memory::Dict{Any, Any})

    #If the logpdf for `node` has been calculated
    #use the stored value.
    if haskey(memory, node.id)
        return memory[node.id]
    end

    #logpdf of each children
    logchildren = []
    for i in eachindex(node.children)
        child = node.children[i]
        #Calculate the child's logpdf if it hasn't been
        #previously calculated
        childlogpdf = !haskey(memory, child.id) ? logpdf!(child, data, params, memory) : memory[child.id]
        push!(logchildren, [childlogpdf...])
    end
    #Array of n_children X n_obs
    logchildren = transpose(reduce(hcat, logchildren))
    #Array of 1 x n_obs
    nodelogpdf = sum(logchildren, dims = 1)
    #Add to memory
    memory[node.id] = nodelogpdf
    nodelogpdf
end
