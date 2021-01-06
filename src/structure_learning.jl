#========================================
Methods related to structure learning
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

