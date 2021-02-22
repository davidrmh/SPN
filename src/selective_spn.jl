"""
    _arrayascendants(leafnodes::Array{Any, 1})

Create an array of arrays containing the ascendants of each leaf node in
`leafnodes`. That is, the i-th inner array contains the ascendants of i-th leaf node.
INTERNAL USE ONLY

# Arguments
- `leafnodes::Array{Any, 1}` Array with leaf nodes (see `filter_by_type` function)
"""
function _arrayascendants(leafnodes::Array{Any, 1})
    arrayascendants = []
    for node in leafnodes
        #Get ascendants for `node`
        asc = ascendants(node)
        push!(arrayascendants, asc)
    end
    arrayascendants
end

"""
    _getcalctree(arrayasc::Array{Any, 1}, dictval::Dict{Any, Any}, samplenum::Integer)

Get calculation tree for a sample.
Return an array containing the id of each node that belongs to the calculation tree
for the correspondent sample.
INTERNAL USE ONLY

# Arguments
- `arrayasc::Array{Any, 1}` Array created with the funciton `_arrayascendants`.

- `dictval::Dict{Any, Any}` Dictionary with the value of each node in each sample (see `logpdf!` function).

- `samplenum::Integer` Sample number (row number in the dataset used to obtain `dictval`).
"""
function _getcalctree(arrayasc::Array{Any, 1}, dictval::Dict{Any, Any}, samplenum::Integer)
    #To store the calctree (only the id of each node)
    calctree = []

    #Ascendants for each leaf node
    for asc in arrayasc
        #Keep nodes with positive pdf (logpdf != -Inf)
        #that have not been added in previous iterations
        for node in asc
            if dictval[node.id][samplenum] != -Inf && !(node.id in calctree)
                push!(calctree, node.id)
            end
        end
    end
    calctree
end

"""
    getcalctrees(leafnodes::Array{Any, 1}, dictval::Dict{Any, Any})

Get the calculation trees for a dataset.
Return an array of arrays. Each inner array contains the id of each node
that belongs to the calculation tree for the correspondent sample.

# Arguments
- `leafnodes::Array{Any, 1}` Array containing the leaf nodes of the SPN.

- `dictval::Dict{Any, Any}` Dictionary with the value of each node in each sample (see `logpdf!` function).

# Example
```julia-repl
julia> include("spn.jl");
julia> bayes = naivebayesmixture();
julia> bayesparams = getparameters(bayes, false)[1];
julia> dictval = Dict();
julia> bayesdata = sample(bayes, 3);
julia> logpdf!(bayes, bayesdata, bayesparams, dictval);
julia> leafnodes = filter_by_type(bayes, LeafNode);
julia> calctrees = getcalctrees(leafnodes, dictval);
```
"""
function getcalctrees(leafnodes::Array{Any, 1}, dictval::Dict{Any, Any})
    #Number of samples
    nsamples = length(dictval[ [keys(dictval)...][1] ])

    #To store the calculation trees
    calctrees = []

    #Array containing the ascendants of each leaf node
    arrayasc = _arrayascendants(leafnodes)
    for n = 1:nsamples
        push!(calctrees, _getcalctree(arrayasc, dictval, n))
    end
    calctrees
end
