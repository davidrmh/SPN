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

"""
    normalize!(root)

Locally normalize a sum product network.

This is Algorithm 1 from `On Theoretical Properties of Sum-Product Networks`
by `Peharz, R., et al.`

# Arguments
- `root::Union{SumNode, ProductNode}` Root of the SPN.

The function modifies the nodes IN-PLACE.
"""
function normalize!(root::Union{SumNode, ProductNode})
    #Topological order
    top_order = topologicalorder(root)

    #Initialize nonnegative correction factors
    #fot product nodes
    dict_alpha = Dict()
    for node in top_order
        if isa(node, ProductNode)
            dict_alpha[node.id] = 1
        end
    end

    for node in top_order
        #Normalize sum node
        if isa(node, SumNode)
            alpha = sum(node.weights)
            @assert alpha != 0 "For node $(node.id) the sum of weights is zero!"
            node.weights = node.weights ./ alpha
        #Product Node
        elseif isa(node, ProductNode)
            alpha = dict_alpha[node.id]
            dict_alpha[node.id] = 1
        end

        #Correct parents
        for parent in node.parents
            if isa(parent, SumNode)
                #Find the corresponding weight
                #and update it
                for i in eachindex(parent.weights)
                    if parent.children[i] === node
                        parent.weights[i] = alpha * parent.weights[i]
                    end
                end
            elseif isa(parent, ProductNode)
                dict_alpha[parent.id] = alpha * dict_alpha[parent.id]
            end
        end
    end
end

"""
    isnormalized(root::AbstractNode)

Determine if a SPN is (locally) normalized.

# Arguments
- `root::AbstractNode` Ideally the root node of the SPN.
If it is not the root node, then it will consider only the sub-SPN
generated by `root`.
"""
function isnormalized(root::AbstractNode)
    #Leaf nodes are always normalized
    if isa(root, Union{DistributionNode, IndicatorNode})
        return true
    end

    #Get the sum nodes
    #and check if the weigths of each one of
    #them is one (close to one)
    sumnodes = filter_by_type(root, SumNode)
    for node in sumnodes
        if !isapprox(sum(node.weights), 1)
            return false
        end
    end
    true
end

"""
    sample(root::AbstractNode, size::Int64)
Get `size` samples from an SPN.
Return a DataFrame with the samples.

# Arguments
- `root::AbstractNode` Root node of a normalized SPN
- `size::Int64` Number of samples
"""
function sample(root::AbstractNode, size::Int64)
    @assert isnormalized(root) "You need to normalize the SPN"

    #Dictionary to store all the samples
    dict_all = Dict()
    #All symbols reachable from root
    varnames = variablenames(root)
    #initialization
    for key in varnames
        dict_all[key] = []
    end

    #Obtain one sample from the SPN
    for i in 1:size
        #dictionary to store one sample
        dict_one = Dict()
        sample!(root, dict_one)

        #Store the sample in dict_all
        for key in varnames
            #To cope with indicator nodes
            val = !haskey(dict_one, key) ? 0 : dict_one[key]
            #Convert true to 1
            val = val === true ? 1 : val
            push!(dict_all[key], val)
        end #inner for

    end #outter for
    #Convert to DataFrame
    DataFrame(dict_all)
end

"""
    getparameters(spn::AbstractNode, logspace::Bool)
Get the parameters from a SPN.

Return a dictionary with keys the id of each node in the SPN and values
the parameters of the node (only sum and leaf nodes have parameters).

Also, return a dictionary with keys the id of each node in the SPN and values
the type of node. If the node is a LeafNode, then the type of the distribution.

# Arguments
- `spn::AbstractNode` Root node of the SPN.

- `logspace::Bool` Indicates if the parameters should be transformed into
the log space (true).
"""
function getparameters(spn::AbstractNode, logspace::Bool)
    #Get sum nodes
    sumnodes = filter_by_type(spn, SumNode)
    dict_params = Dict()
    dict_types = Dict()
    #Add weights
    for node in sumnodes
        dict_params[node.id] = logspace ? log.(node.weights) : node.weights
        dict_types[node.id] = typeof(node)
    end
    #Get distribution nodes
    distnodes = filter_by_type(spn, LeafNode)
    for node in distnodes
        par = [params(node.distribution)...]
        par = logspace ? tologspace(node.distribution, par) : par
        dict_params[node.id] = par
        dict_types[node.id] = typeof(node.distribution)
    end
    dict_params, dict_types
end

"""
    ascendants(node::AbstractNode)

Get the asscendants of a node.
Return an array with nodes.
Important: This function assumes that a node is
its own ascendant.

# Arguments
- `node::AbstractNode` A node <: AbstractNode.
"""
function ascendants(node::AbstractNode)
    #To store the ascendants
    asc = []
    #A now is its own ascendant
    push!(asc, node)

    #Root nodes have no ascendants
    if node.parents == [undef]
        return asc
    end
    #Nodes to explore
    explore = copy(node.parents)

    while explore != []
        #DFS search
        parent = popfirst!(explore)

        #Add parent to `asc` if it hasn't been added
        !(parent in asc) ? push!(asc, parent) : nothing

        #Add parent's parents to continue exploration
        #Only for non-root nodes
        parent.parents != [undef] ? pushfirst!(explore, copy(parent.parents)...) : nothing
    end
    asc
end
