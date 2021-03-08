
"""
    _partial_deriv!(root::AbstractNode, des::AbstractNode, logpdfmem::Dict{Any, Any}, derivmem::Dict{Any, Any}

Calculate the partial derivative of the root node with respect
to one of its descendants.

Return an array of size 1 x `number of observations` used to calculate the
argument `logpdfmem` (see function `logpdf!`).

Modify `in-place` the argument `derivmem` which is a dictionary with keys
`(i, j)` that correspond to the partial derivative of node with id i with respect to
node with id j.

# Arguments
- root::AbstractNode` Root node.

- `des::AbstractNode` A node.

- `logpdfmem::Dict{Any, Any}` Dictionary created with the `logpdf!` function.

- `derivmem::Dict{Any, Any}` Dictionary with the partial derivatives.
"""
function _partial_deriv!(root::AbstractNode, des::AbstractNode,
    logpdfmem::Dict{Any, Any}, derivmem::Dict{Any, Any})

    #Number of observations in the dataset used to calculate logpdfmem
    nobs = length(logpdfmem[root.id])
    #if `des` is `root`, then the partial derivative is 1
    if des == root
        #Add to memory if necessary
        if !((root.id, root.id) in keys(derivmem))
            derivmem[(root.id, root.id)] = ones(nobs)
        end
        return derivmem[(root.id, root.id)]
    end

    #If des is not root, then calculate the partial derivate
    #recursively and using the local derivatives
    sum = 0
    for parent in des.parents
        key1 = (root.id, parent.id)
        key2 = (parent.id, des.id)
        #Avoid repeated calculations
        if key1 in keys(derivmem)
            if key2 in keys(derivmem)
                sum = sum .+ derivmem[key1] .* derivmem[key2]
            else
                sum = sum .+ derivmem[key1] .* _local_partial_deriv!(parent, des, logpdfmem, derivmem)
            end
        else
            if key2 in keys(derivmem)
                sum = sum .+ _partial_deriv!(root, parent, logpdfmem, derivmem) .* derivmem[key2]
            else
                sum = sum .+ _partial_deriv!(root, parent,  logpdfmem, derivmem) .* _local_partial_deriv!(parent, des, logpdfmem, derivmem)
            end
        end
    end
    #Add to memory
    derivmem[(root.id, des.id)] = sum
    return derivmem[(root.id, des.id)]
end

"""
    nodesderivatives(root::AbstractNode, logpdfmem::Dict{Any, Any}, onlyroot = true::Bool)

Calculate the partial derivatives of each node with respect to the other nodes.

Return a dictionary whose keys `(i,j)` contain the partial derivative of node
`i` with responde to node `j`. The value of this key is an array with size
`1 x number of observations` used to obtaind the argument `logpdfmem` (see `logpdf!` function).

# Arguments

- `root::AbstractNode` Root node.

- `logpdfmem::Dict{Any, Any}` Dictionary created with the `logpdf!` function.
`logpdfmem[id]` contains the logpdf function for the node with `node.id = id`.

- `onlyroot = true::Bool` Bool. If `true` (default) just keep the partial derivatives
of the root node with respect to every other node. When `false` keep the derivatives
of every node with respect to the others.
"""
function nodesderivatives(root::AbstractNode, logpdfmem::Dict{Any, Any}, onlyroot = true::Bool)
    #Get leaves
    leaves = filter_by_type(root, LeafNode)
    #To store the derivatives
    derivmem = Dict()
    for leaf in leaves
        #Calculate partial derivatives bottom up
        _partial_deriv!(root, leaf, logpdfmem, derivmem)
    end

    #Keep only the partial derivatives of the root node with respect
    #to the other nodes.
    if onlyroot
        aux = Dict()
        for k in keys(derivmem)
            if k[1] == root.id
                aux[k] = derivmem[k]
            end
        end
        return aux
    end
    derivmem
end
