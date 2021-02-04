#========================================
Methods related to structure learning
========================================#
"""
    disconnect!(parent, child)
Disconnect a parent and a child node.
Both arguments are modified in-place.

# Arguments
- `parent::AbstractNode` Parent node.
- `child::AbstractNode` Child node.
"""
function disconnect!(parent::AbstractNode, child::AbstractNode)
    #Check if nodes are connected
    child in parent.children ? nothing : throw("Nodes are not connected")

    #Search the index for the child in parent.children
    for i in eachindex(parent.children)
        #Disconnects
        if parent.children[i] == child
            popat!(parent.children, i);
            break
        end
    end

    #Search the index for the parent in child.parents
    for i in eachindex(child.parents)
        #Disconnects
        if child.parents[i] == parent
            popat!(child.parents, i)
            break
        end
    end
end

"""
    delete!(node)
Delete `node` from the network.

# Arguments
- `node::AbstractNode`
"""
function delete!(node::AbstractNode)

    #Disconnect parents
    if node.parents != []
        #This shallow copy is very important
        parents = copy(node.parents)
        for parent in parents
            disconnect!(parent, node)
        end
    end

    #Disconnect children
    if hasfield(typeof(node), :children) && node.children != []
        #This shallow copy is very important
        children = copy(node.children)
        for child in children
            disconnect!(node, child)
        end
    end
end

"""
    _register_copy!(root)

INTERNAL USE ONLY
Rename the nodes in a copy created with `copysubspn` function.
The modification is IN-PLACE.

# Arguments
- `root:AbstractNode` Node representing the root node from a copy.
"""
function _register_copy!(root::AbstractNode)
    #Get the descendants
    des = descendants(root)

    #Register and update fields `id` and `copyof`
    global _idcounter
    for node in des
        #Update global variable for counting
        _idcounter = _idcounter + 1
        node.copyof = node.id
        node.id = _idcounter
    end
end

"""
    copysubspn(root)

Copy a sub-SPN rooted at node `root`.
Return a node representing the root node of the sub-spn

# Arguments
- `root::Union{SumNode, ProductNode}` A node represeting the root node.
"""
function copysubspn(root::Union{SumNode, ProductNode})
    #Deep copy of the original SPN
    subspn = deepcopy(root)

    #Remove root's parents
    subspn_parents = copy(subspn.parents)
    for parent in subspn_parents
        disconnect!(parent, subspn)
    end

    #From root's descendants remove
    #parents except root
    subspn_des = descendants(subspn)
    for node in subspn_des
        #By definition a node it's its own descendant
        if node != subspn
            for parent in copy(node.parents)
                #Delete parents that are not descendants
                #of the root.
                if parent != subspn && !(parent in subspn_des)
                    disconnect!(parent, node)
                end
            end
        end
    end
    #Establish root as root node
    push!(subspn.parents, undef)

    #Register copied nodes
    _register_copy!(subspn);
    return subspn
end

"""
    reachable_x(node, varname)

Get the distribution (indicator) nodes related to `varname` that can be reached from `node`.
This function corresponds to the function I_{x}(N) in the paper
`Learning Selective Sum-Product Networks by Peharz, R. et al`.

# Arguments
- `node::AbstractNode` Node from which the search is started.
- `varname::Symbol` Variable of interest.

If we have a variable `X1` that can take on `K` distinct values, `X1_1, ... X1_K`
then varname is `:X1`.

Return an array of symbols corresponding to the names of each reachable
distribution (indicator) node.
"""
function reachable_x(node::AbstractNode, varname::Symbol)
    #Get all variables that can be reached from `node`
    variables = variablenames(node)

    #keep only the ones related to `varname`
    str_target = string(varname)
    I_X = []
    for v in variables
        occursin(str_target, string(v)) ? push!(I_X, v) : nothing
    end
    I_X
end

"""
    dismiss!(root, varname, targetlist)

Disconnect the nodes reachable from `root` that are related to `varname`
and can reach the nodes in `targetlist`.
This function is the Dismiss function (Algorithm 2) from
`Learning Selective Sum-Product Networks by Peharz, R. et al`.

Modifies `root` IN-PLACE.

# Arguments
- `root::AbstractNode` Node where the search begins.
- `varname::Symbol`
- `targetlist::Array` Array with symbols of the variables to disconnect.

If we have a variable `X1` that can take on `K` distinct values, `X1_1, ... X1_K`
then varname is `:X1`. If we want to disconnect `X1_1` and `X1_2` then
`targetlist = [:X1_1, :X1_2]`.
"""
function dismiss!(root::AbstractNode, varname::Symbol, targetlist::Array)
    #Validation
    variables_reachable = reachable_x(root, varname)
    if !(issubset(targetlist, variables_reachable))
        throw("Target list must be a subset of reachable_x(root, varname)")
    end

    #Get descendants
    des = descendants(root)

    for node in des
        if hasfield(typeof(node), :children)
            children = copy(node.children)
            for child in children
                reach_by_child = reachable_x(child, varname)
                #Empty set is subset of every set
                if issubset(reach_by_child, targetlist) && reach_by_child != []
                    disconnect!(node, child)
                end
            end
        end
    end
end

"""
    shortwire!(root::AbstractNode)
For all sum and product nodes, N, that only have one child C, connect the
parents of N as parents of C and delete N.
This function is the ShortWire function (Algorithm 3) from
`Learning Selective Sum-Product Networks by Peharz, R. et al`.

The modification is done in-place.

# Arguments
- `root::AbstractNode` Ideally the root node of the SPN.
"""
function shortwire!(root::AbstractNode)
    #Get the sum and product nodes
    nodes = filter_by_type(root, Union{SumNode, ProductNode})

    for n in nodes
        #If has only a child
        if length(n.children) == 1
            child = n.children[1]
            #add child to each parent of the node n
            for parent in n.parents
                #Check if node n is not the root node
                if parent != undef
                    addchildren!(parent, [child])
                end
            end
            #Delete node n
            delete!(n)
        end
    end
end

"""
    getchainproduct(start::ProductNode)
Get a chain of product nodes starting in the node `start`.
Return an array with the product nodes forming the chain.
The first element of this chain is the `start` node.

# Arguments
- `start::ProductNode` Product node where the search starts.
"""
function getchainproduct(start::ProductNode)
    #To store the chain
    chain = [start]
    #To store the nodes to be explored
    explore = [start]

    while explore != []
        node = pop!(explore)
        for child in node.children
            if isa(child, ProductNode)
                push!(chain, child)
                push!(explore, child)
            end
        end
    end
    chain
end

"""
    reducechain!(chain::Array{ProductNode, 1})
Reduce a chain of products node contained in the array `chain`.
This array is created with the function `getchainproduct`.

reducechain! modifies the nodes in `chain` in-place.

# Arguments
- `chain::Array{ProductNode, 1}` Array of product nodes that form the chain.

For each node `chain[k]` with `k > 1`, its children are added to
`chain[1].children`. If `chain[k]` is children of `chain[1]`, then these
nodes get disconnected.
"""
function reducechain!(chain::Array{ProductNode, 1})
    start = chain[1]
    m = length(chain)
    for i in 2:m
        node = chain[i]
        #Coonect each child (not a product node) from node to start
        for child in node.children
            if !isa(child, ProductNode) && !(child in start.children)
                addchildren!(start, [child])
            end
        end

        #Disconnect node from start if node is a child of start
        #Not using delete! since this might affect other nodes depending
        #on node
        if node in start.children
            disconnect!(start, node)
        end
    end
end

"""
    collapseproducts!(root::AbstractNode)

Combine chains of product nodes to a single product node.

This function is the CollapseProducts function (Algorithm 3) from
`Learning Selective Sum-Product Networks by Peharz, R. et al`.

The modification is done in-place.

# Arguments
- `root::AbstractNode` Root node of the SPN.
"""
function collapseproducts!(root::AbstractNode)
    #Get product nodes
    prodnodes = filter_by_type(root, ProductNode)

    #Get the chain for each productnode
    chains = map(getchainproduct, prodnodes)

    #Calculate the length of each chain
    lengthchains = map(length, chains)

    #Reduce while there is a chain that can be reduced
    while any(lengthchains .> 1)
        map(reducechain!, chains)
        prodnodes = filter_by_type(root, ProductNode)
        chains = map(getchainproduct, prodnodes)
        lengthchains = map(length, chains)
    end
end
