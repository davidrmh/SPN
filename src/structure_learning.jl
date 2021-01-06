#========================================
Methods related to structure learning

TO DO 
    Method to delete unreachable nodes
========================================#
"""
Function to disconnect a parent and a child
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
Delete a node from the network
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

