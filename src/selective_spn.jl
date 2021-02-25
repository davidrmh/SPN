"""
    _getcalctree(root::AbstractNode, dictval::Dict{Any, Any}, samplenum::Integer)

Get calculation tree for a sample.
Return an array containing the id of each node that belongs to the calculation tree
for the correspondent sample.
INTERNAL USE ONLY

# Arguments
- `root::AbstractNode` Root node of the SPN.

- `dictval::Dict{Any, Any}` Dictionary with the value of each node in each sample (see `logpdf!` function).

- `samplenum::Integer` Sample number (row number in the dataset used to obtain `dictval`).
"""
function _getcalctree(root::AbstractNode, dictval::Dict{Any, Any}, samplenum::Integer)
    #To store the calctree (only the id of each node)
    calctree = []

    #Sample has zero density
    if dictval[root.id][samplenum] == -Inf
        return []
    end

    #To store the nodes to be explored
    explore = []
    push!(explore, root)

    #Add root to the calculation  tree
    push!(calctree, root.id)
    while explore != []
        parent = pop!(explore)
        #Case when parent is internal node
        if hasfield(typeof(parent), :children)
            for child in parent.children
                #add child to the calculation tree if it has positive density
                if dictval[child.id][samplenum] > -Inf
                    #Avoid repetitions
                    if !(child.id in calctree)
                        push!(calctree, child.id)
                    end
                    if !(child in explore)
                        push!(explore, child)
                    end
                end #dictval
            end #child
        #Case when parent is leaf node
        else
            if dictval[parent.id][samplenum] > -Inf
                #Avoid repetitions
                if !(parent.id in calctree)
                    push!(calctree, parent.id)
                end
            end
        end #hasfield
    end#while
    calctree
end

"""
    getcalctrees(root::AbstractNode, dictval::Dict{Any, Any})

Get the calculation trees for a dataset.
Return an array of arrays. Each inner array contains the id of each node
that belongs to the calculation tree for the correspondent sample.

# Arguments
- `root::AbstractNode` Array containing the leaf nodes of the SPN.

- `dictval::Dict{Any, Any}` Dictionary with the value of each node in each sample (see `logpdf!` function).
"""
function getcalctrees(root::AbstractNode, dictval::Dict{Any, Any})
    #Number of samples
    nsamples = length(dictval[ [keys(dictval)...][1] ])

    #To store the calculation trees
    calctrees = []

    for n = 1:nsamples
        push!(calctrees, _getcalctree(root, dictval, n))
    end
    calctrees
end

"""
    selective_mle!(root::AbstractNode, calctrees::Array{Any, 1})

Set the weights for sum nodes to the maximum likelihood estimate for selective SPNs.
This function implements equation (9) from the paper
`Learning Selective Sum-Product Networks by Peharz, R. et al`.

The modification is done in place for all the sum nodes in the network.

# Arguments
- `root::AbstractNode` Root node of the SPN.

- `calctrees::Array{Any, 1}` Array with the calculation trees (see function `getcalctrees`).
"""
function selective_mle!(root::AbstractNode, calctrees::Array{Any, 1})
    #Get sum nodes
    sumnodes = filter_by_type(root, SumNode)

    #Obtain the ML estimates
    for node in sumnodes
        #number of children
        nchildren = length(node.children)
        #This array stores the quantities #(S,C)
        count_sc = zeros(nchildren)
        for i = 1:nchildren
            child = node.children[i]
            for tree in calctrees
                #Increase count if the sum node and its child appear in the tree
                if node.id in tree && child.id in tree
                    count_sc[i] = count_sc[i] + 1
                end #if
            end#tree
        end #i
        #This variable is #(S)
        sum_count = sum(count_sc)
        if  sum_count == 0
            mle_weights = zeros(nchildren) .+ 1 /nchildren
        else
            mle_weights = count_sc ./ (sum_count + eps())
        end #if
        #Update weights
        setweights!(node, mle_weights)
    end #node
end
