#========================================
General methods for nodes
========================================#
"""
Add children to a parent node.
Children is an interable with Nodes.
This function is used for building a SPN using a top to bottom approach.
"""
function addchildren!(parent::AbstractNode, children::Array{<:AbstractNode})

    #Add the children to the parent
    push!(parent.children, children...)

    #Add the parent to each child
    for child in children
        push!(child.parents, parent)
    end
end

"""
Get the scope of a non-leaf node (sum or product node)
Returns an array with symbols
"""
function scope(node::Union{SumNode, ProductNode}, neg_string = "_NEG"::String)
    sc = []
    for child in node.children
        push!(sc, scope(child, neg_string))
    end
    return vcat(unique(sc)...)
end

"""
Get the scope for leaf nodes
Returns a symbol
"""
function scope(node::Union{DistributionNode, IndicatorNode}, neg_string = "_NEG"::String)
    stringsymbol = String(node.varname)
    #Check if is a negated variable
    substringloc = findfirst(uppercase(neg_string), uppercase(stringsymbol))
    if  substringloc!= nothing
        varsymbol = Symbol(stringsymbol[1:(substringloc[1] - 1)])
    else
        varsymbol = node.varname
    end
    varsymbol
end

"""
Get the variable names that are descendants of a non-leaf node (sum or product node)
Returns an array with symbols
"""
function variablenames(node::Union{SumNode, ProductNode})
    variables = []
    for child in node.children
        push!(variables, variablenames(child))
    end
    return vcat(unique(variables)...)
end

"""
Get the variable name for a leaf node
Returns a symbol
"""
function variablenames(node::Union{DistributionNode, IndicatorNode})
    node.varname
end


"""
Get the descendants of a node
returns an array with nodes
"""
function descendants(node::AbstractNode)
    #to store the descendants
    des = []
    #By definition a node is its own descendant
    push!(des, node)

    #Leaf nodes have no children
    if !hasfield(typeof(node), :children)
        return des
    end

    #nodes to explore
    #Shallow copy
    explore = copy(node.children)

    while explore !=[]
        child = pop!(explore)

        #Add the node if hasn't been added
        !(child in des) ? push!(des, child) : nothing

        #Add child's children to continue the exploration
        hasfield(typeof(child), :children) ? push!(explore, copy(child.children)...) : nothing
    end
    des
end

"""
Get a node by its id
The search is done using BFS and ideally
should start in the root node.

You can't start the search using a
leaf node.
"""
function node_by_id(root::Union{SumNode, ProductNode}, id::Integer)
    if root.id == id
        return root
    end
    #Shallow copy, so every change
    #in the returned node will change
    #the spn
    explore = copy(root.children)
    while explore != []
        #Explore a child (BFS manner)
        child = pop!(explore)
        if child.id == id
            return child
        #Add children of child node
        elseif hasfield(typeof(child), :children)
            push!(explore, copy(child.children)...)
        end
    end
    throw("No node with id:$id")
end

"""
Get the nodes of a certain type
Ideally this search starts in the root node.

The search can't initialize in a leaf node
"""
function nodes_by_type(root::Union{SumNode, ProductNode}, type::Type)
    #to store the nodes
    nodes = []
    isa(root, type) ? push!(nodes, root) : nothing

    #to store the nodes to be explored
    explore = []
    #shallow copy
    push!(explore, copy(root.children)...)
    
    while explore != []
        child = pop!(explore)
        #Add child if is the desired type and hasn't been added
        (isa(child, type) && !(child in nodes)) ? push!(nodes, child) : nothing
        #Add child's children to the exploration
        hasfield(typeof(child), :children) ? push!(explore, copy(child.children)...) : nothing
    end
    nodes
end
