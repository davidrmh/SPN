#========================================
Methods related to structure learning

TO DO
    Method to delete unreachable nodes
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
    _rename_copy!(root)

INTERNAL USE ONLY
Rename the nodes in a copy created with copysubspn function.
The modificatio is IN-PLACE.

# Arguments
- `root:Union{SumNode, ProductNode}` Node representing the root node.
- `sep::String` Separator to distinguish between the original node id
and the copies id. If node.id is "1" then the first copy will have id
equal to string(node.id, sep, 1).
If this latter node has a k-th copy (k greater than 2), then this new copy will have
the id string(node_copy.id, k).

# Examples

If the original node has node.id = "1" and sep = "_", then the first
copy will have id = "1_1".
If we make a copy of this first copy then this new copy will have
id = "1_2" and similar if we continue making copies of the last copy.
"""
function _rename_copy!(root::Union{SumNode, ProductNode}, sep="_"::String)
    #Get the descendants
    des = descendants(root)

    #Rename
    for node in des
        #Is first copy
        if !contains(node.id, sep)
            node.id = string(node.id, sep, "1")
        #Is copy of a previous copy
        else
            splt = split(node.id, sep)
            #Original id
            origin = splt[1]
            #Last copy number
            n = parse(Int64, splt[2])
            node.id = string(origin, sep, n + 1)
        end
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

    #Rename nodes
    _rename_copy!(subspn, "_");
    return subspn
end
