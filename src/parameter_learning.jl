
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

"""
PENDING WORK
Get the partial derivative of the log-likelihood function with respect to the
weights of each sum node.

Return a dictionary with keys `(i, j)` where `i` is the id of a sum node and
`j` the index of its weight `j`.

This function implements equation (6.6) from the PhD. thesis
`Foundations of Sum-Product Networksfor Probabilistic Modeling` by Robert Peharz.
"""
function weights_loglike_deriv(root::AbstractNode, logpdfmem::Dict{Any, Any}, derivmem::Dict{Any, Any})
    #Get sum nodes
    sumnodes = filter_by_type(root, SumNode)

    #To store the results
    logderiv = Dict()

    for node in sumnodes
        for i in eachindex(node.children)
            child = node.children[i]
            factor1 = 1 ./(exp.(logpdfmem[root.id]) .+ eps())
            #Partial derivative of root with respect child
            factor2 = derivmem[(root.id, node.id)]
            factor3 = exp.(logpdfmem[child.id])
            logderiv[(node.id, i)] = sum(factor1 .* factor2 .* factor3)
        end # i
    end # node
    logderiv
end

"""
Gradient ascent step for weights.
PENDING WORK
"""
function _weights_grad_step!(sumnodes::Array{Any, 1}, gradient::Dict{Any, Any}, rate::Float64)
    for node in sumnodes
        for i in eachindex(node.weights)
            #update the weight with a gradient ascent step
            node.weights[i] = node.weights[i] + rate * gradient[(node.id, i)]
        end
    end
end

"""
Project the weights onto the probabilistic simplex
The modification is done in-place.

This function implements the algorithm in figure 1 from the paper
`Efficient Projections onto the L1-Ball for Learning in High Dimensions` by Duchi, et al.
"""
function _projection_simplex!(sumnodes)

    for node in sumnodes
        #Number of weights
        n = length(node.weights)

        #sorted weights (decreasing order)
        mu = sort(node.weights, rev = true)

        #Cummulative sum
        mu_cumsum = cumsum(mu)

        #to store the j indices
        rho_set = []
        for j in 1:n
            aux = mu[j] - (1/j) *(mu_cumsum[j] - 1)
            if aux > 0
                push!(rho_set, j)
            end
        end # j
        rho = rho_set != [] ? maximum(rho_set) : n
        theta = (1 / rho) * (mu_cumsum[rho] - 1)
        #Update weights
        for i in eachindex(node.weights)
            node.weights[i] = max(node.weights[i] - theta, 0)
        end # i
    end #node
end
"""
Learn the parameters using gradient ascent
PENDING WORK
"""
function learn_weights_gradient!(root::Union{SumNode, ProductNode}, data::DataFrame,
    niter::Int64, rate::Float64, boolinit::Bool)
    #Get sum nodes
    sumnodes = filter_by_type(root, SumNode)

    #Initialize the weights
    if boolinit
        initializeweights!(root)
    end

    parameters = getparameters(root, false)[1]
    for i in 1:niter

        logpdfmem = Dict()
        #Evalate the data with current parameters
        logpdf!(root, data, parameters, logpdfmem)

        #Nodes derivatives
        nodesderiv = nodesderivatives(root, logpdfmem, true)

        #Gradient of log-likelihood with respect to weights
        weights_grad = weights_loglike_deriv(root, logpdfmem, nodesderiv)

        #Gradient step
        _weights_grad_step!(sumnodes, weights_grad, rate)

        #Project onto the probabilistic simplex
        _projection_simplex!(sumnodes)
    end
end

"""
    learn_weights_em!(root::Union{SumNode, ProductNode}, data::DataFrame, niter::Int64, boolinit::Bool)

Optimise weights using Expectation-Maximization algorithm. The modification is done in-place.

This function implements Algorithm 7 from the PhD. thesis
`Foundations of Sum-Product Networksfor Probabilistic Modeling` by Robert Peharz.

# Arguments
- `root::Union{SumNode, ProductNode}` Root node.

- `data::DataFrame` Data used for learning.

- `niter::Int64` Number of iterations.

- `boolint::Bool` Whether to initialize the weights (true) or not (false).
This parameter can be used to continue the training (boolinit = false) of a model.
"""
function learn_weights_em!(root::Union{SumNode, ProductNode}, data::DataFrame, niter::Int64, boolinit::Bool)
    #Get sum nodes
    sumnodes = filter_by_type(root, SumNode)

    #Initialize the weights
    if boolinit
        initializeweights!(root)
    end

    #This line is outside because the content in `parameters` is pointer-like referenced
    parameters = getparameters(root, false)[1]
    for n in 1:niter
        #Evalate the data
        logpdfmem = Dict()
        logpdf!(root, data, parameters, logpdfmem)

        #Nodes derivatives
        nodesderiv = nodesderivatives(root, logpdfmem, true)

        #To store the variable n_{S,C}
        n_sc = Dict()

        #To store the sum over C of n_{S, C}
        sum_c = Dict()
        for node in sumnodes
            sum_c[node.id] = 0
            for i in eachindex(node.children)
                child = node.children[i]
                w = node.weights[i]
                factor1 = 1 ./(exp.(logpdfmem[root.id]) .+ eps())
                #Partial derivative of root with respect child
                factor2 = nodesderiv[(root.id, node.id)]
                factor3 = exp.(logpdfmem[child.id])
                aux = sum(factor1 .* factor2 .* factor3 * w)
                n_sc[(node.id, i)] = aux
                #Accumulate the sum over the children of sum node S
                sum_c[node.id] = sum_c[node.id] + aux
            end # i

            #Update weight
            for i in eachindex(node.children)
                node.weights[i] = n_sc[(node.id, i)] / (sum_c[node.id] + eps())
            end # i
        end # node
    end #niter
end
