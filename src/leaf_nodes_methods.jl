#========================================
Methods for Leaf nodes
========================================#
"""
    pdf(d::LeafNode, data::Union{Real, AbstractArray, NamedTuple, DataFrame},
    params::Dict{Any, Any}, logspace = false::Bool)

Evaluate the pdf of a leaf node.

# Arguments
- `d::LeafNode` A LeafNode (Distribution or Indicator node) object.

- `data::Union{Real, AbstractArray, NamedTuple, DataFrame}` Data.

- `params::Dict{Any, Any}` Dictionary created with the function `getparameters`.
Useful por calculating the gradients with Zygote.

- `logspace::Bool` Indicates if the parameters are in the log space (true) or
in the original space (false).
"""
function pdf(d::LeafNode, data::Union{Real, AbstractArray, NamedTuple, DataFrame},
    params::Dict{Any, Any}, logspace = false::Bool)
    if isa(data, DataFrame)
        vals = data[!, d.varname]
    #Array with named tuples
    elseif isa(data[1], NamedTuple)
        vals = [tup[d.varname] for tup in data]
    else isa(data, NamedTuple)
        vals = data[d.varname]
    end
    #If parameters come from the logspace
    #return them to the original space
    par = logspace ? topositivespace(d.distribution, params[d.id]) : params[d.id]
    Distributions.pdf.(typeof(d.distribution)(par...), vals)
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

"""
    tologspace(dist::Normal{Float64}, par::AbstractArray)

Change the parameter from a univariate normal distribution to the logspace.
In this case, the only parameter modified is the standard deviation.

# Arguments

- `dist::Normal{Float64}` Object of type Normal{Float64}.

- `par::AbstractArray` Parameters of the distribution.

The parameter `dist` is only used to have multiple dispatch.
"""
function tologspace(dist::Normal{Float64}, par::AbstractArray)
    #For the normal distribution the
    #second parameter in `par` is the standard deviation
    [par[1], log(par[2])]
end

"""
    topositivespace(dist::Normal{Float64}, par::AbstractArray)

Change the parameter from a univariate normal distribution to the positive (original) space.
In this case, the only parameter modified is the standard deviation.

This is the inverse function of `tologspace`.

# Arguments

- `dist::Normal{Float64}` Object of type Normal{Float64}.

- `par::AbstractArray` Parameters of the distribution.

The parameter `dist` is only used to have multiple dispatch.
"""

function topositivespace(dist::Normal{Float64}, par::AbstractArray)
    #For the normal distribution the
    #second parameter in `par` is the standard deviation
    [par[1], exp(par[2])]
end
