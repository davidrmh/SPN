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
