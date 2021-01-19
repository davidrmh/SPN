#========================================
General methods for nodes
========================================#
"""
    addchildren!(parent, children)

Add children to a parent node.
This function is used for building an SPN using a top to bottom approach.
Both arguments are modified in-place.

# Arguments
- `parent::AbstractNode` parent node.
- `children::Array{<:AbstractNode}` children to add.
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
    scope(node, sep_string)

Get the scope of a non-leaf node (sum or product node).
Return an array with symbols.

# Arguments
- `node::Union{SumNode, ProductNode}` Node
- `sep_string = "_"::String` String separator.

The argument `sep_string` is used to consider the cases when dealing with
indicator nodes that represent distinct possible values for the same random variable.
For example, if X1 takes on three values, then we could represent it with
three indicator nodes with varname fields :X1_1, :X1_2, :X1_3 respectively.
"""
function scope(node::Union{SumNode, ProductNode}, sep_string = "_"::String)
    sc = []
    for child in node.children
        push!(sc, scope(child, sep_string))
    end
    return vcat(unique(sc)...)
end

"""
    scope(node, sep_string)

Get the scope of a leaf node (Distribution or Indicator node).
Return a symbol.

# Arguments
- `node::Union{DistributionNode, IndicatorNode}` Node
- `sep_string = "_"::String` String separator.
"""
function scope(node::Union{DistributionNode, IndicatorNode}, sep_string = "_"::String)
    stringsymbol = String(node.varname)
    #Check if varname refers to a possible value of
    #a variable (this is used for finite support variables)
    substringloc = findfirst(uppercase(sep_string), uppercase(stringsymbol))
    if  substringloc!= nothing
        varsymbol = Symbol(stringsymbol[1:(substringloc[1] - 1)])
    else
        varsymbol = node.varname
    end
    varsymbol
end

"""
    variablenames(node)

Get the variables reachable from a node. This is not the sames as `scope` function
since for indicator variables representing different values of the same
random variable, they are considered different names.

Return an array of symbols

# Arguments
- `node::AbstractNode` A node <: AbstractNode.
"""
function variablenames(node::AbstractNode)::Array{Symbol}
    #leaf node
    if !(hasfield(typeof(node), :children))
        return [node.varname]
    end
    #Get the leaves
    leaves = filter_by_type(node, Union{DistributionNode, IndicatorNode})

    #To store the symbols
    varnames = []

    for node in leaves
        !(node.varname in varnames) ? push!(varnames, node.varname) : nothing
    end
    varnames
end

"""
    descendants(node)
Get the descendants of a node. Return an array with nodes.

# Arguments
- `node::AbstractNode` A node <: AbstractNode.
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
    filter_by_id(root, id)
Get a node with id = `id` that is reachable from `root` node.
Return the node with the specified id.
The search can't start in a leaf node.

# Arguments
- `root::Union{SumNode, ProductNode}` A sum or product node. Ideally a root node.
- `id::Int64` id of the desired node.
"""
function filter_by_id(root::Union{SumNode, ProductNode}, id::Int64)
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
    filter_by_type(root, type)

Get a set of nodes with of type `type` that are reachable from `root`
Ideally this search starts in the root node.
The search can't start in a leaf node.

# Arguments
- `root::Union{SumNode, ProductNode}` Sum or product node.
- `type::Type` The type of the desired nodes.
"""
function filter_by_type(root::Union{SumNode, ProductNode}, type::Type)
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

"""
    topologicalorder(root)

Create a topological ordered list (increasing order) of the sum and product
nodes in an SPN.
The ordering is created in a bottom-up perspective.

Return a list with the ordered nodes.

# Arguments
- `root::Union{SumNode, ProductNode}` Root node of the SPN
"""
function topologicalorder(root::Union{SumNode, ProductNode})
    #Get the leaves nodes
    leaves = filter_by_type(root, Union{IndicatorNode, DistributionNode})

    #Add leaves parents
    top_order = []
    for node in leaves
        for parent in node.parents
            !(parent in top_order) ? push!(top_order, node.parents...) : nothing
        end
    end

    #Add the parents "layer wise"
    #root node has root.parents = [undef]
    for node in top_order
        #not a root node
        if node.parents != [undef]
            for parent in node.parents
                #Not previously added
                !(parent in top_order) ? push!(top_order, parent) : nothing
            end #inner for
        end #if
    end #outter for
    return top_order
end #function
