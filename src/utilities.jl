#========================================
Utilities
========================================#
"""
    _dotstring_labels(node)

INTERNAL FUNCTION

The function is used to create the labeling part of the DOT format string
representing the SPN. Its result is similar to the below example

digraph G {
1[label="+"];
2[label="*"];
3[label="X1"];

# Arguments
- `node::AbstractNode` A node.
"""
function _dotstring_labels(node::AbstractNode)
    #get descendants
    des = descendants(node)
    str = "digraph SPN{ \n"

    #Add to str the node label corresponding
    #to each node
    for n in des
        if isa(n, SumNode)
            str = str * string(n.id, " [label = \"+ \\n id:$(n.id) \\n cp:$(n.copyof)\"];\n")
        elseif isa(n, ProductNode)
            str = str * string(n.id, " [label = \"* \\n id:$(n.id) \\n cp:$(n.copyof)\"];\n")
        elseif isa(n, Union{IndicatorNode, DistributionNode})
            str = str * string(n.id, " [label=\"$(string(n.varname)) \\n id:$(n.id) \\n cp:$(n.copyof)\"];\n")
        else
            str = str * string(n.id, " [label = \"UNKNOW\"];\n")
        end
    end
    str
end

"""
    _dotstring_labels(node)

INTERNAL FUNCTION

The function is used to create the labeling part of the DOT format string
representing the SPN. Its result is similar to the below example

digraph G {
1[label="+"];
2[label="*"];
3[label="X1"];

# Arguments
- `node::AbstractNode` A node. Ideally the root node.
- `array::Array{String,1}`. Array containing the string from `_dotstring_labels`
- `memory::AbstractArray`

`array` is modified in place and will contain the string
with the edges in DOT notation.
"""
function _dotstring_edges!(node::AbstractNode, array::Array{String,1}, memory::AbstractArray)
    #Distribution nodes have no edges coming out
    if isa(node, Union{IndicatorNode, DistributionNode})
        return
    end
    str = pop!(array)
    #Add edges node -> child
    for i in eachindex(node.children)
        child = node.children[i]
        if isa(node, SumNode)
            w = round(node.weights[i], digits = 2)
            if !((node.id, child.id) in memory)
                str = string(str, node.id, "->", child.id, " [label = $w];\n")
                push!(memory, (node.id, child.id))
            end
        else
            if !((node.id, child.id) in memory)
                str = string(str, node.id, "->", child.id, ";\n")
                push!(memory, (node.id, child.id))
            end
        end
    end
    push!(array, str)

    #Recursion
    for child in node.children
        _dotstring_edges!(child, array, memory)
    end
end

"""
    dotfile(node, filename = "my_spn.gv")

Create a file in DOT format.

# Arguments
- `node::AbstractNode`A node. Ideally the root node.
- `filename::String` Path with the name of the gv file.
"""
function dotfile(node::AbstractNode, filename = "my_spn.gv"::String)
    str = _dotstring_labels(node)
    array = [str]
    memory = []
    _dotstring_edges!(node, array, memory)
    str = pop!(array)
    str = string(str, "}")
    open(filename, "w") do io
        write(io, str)
    end
end

"""
    createnamedtuples(data::DataFrame)

Create an array with named tuples using  the DataFrame `data`.
Each key in the tuple correspon to the name of a column in `data`.

# Arguments
- `data::DataFrame` DataFrame object.
"""
function createnamedtuples(data::DataFrame)
    keys = Symbol.(names(data))
    arraytup = []
    for row in eachrow(data)
        tup = (; zip(keys, row)...)
        push!(arraytup, tup)
    end
    arraytup
end
