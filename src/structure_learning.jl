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
