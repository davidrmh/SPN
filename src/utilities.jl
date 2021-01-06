#========================================
Utilities
========================================#
"""
INTERNAL FUNCTION
This function is used for creating the
the DOT string.
The result a string with the following
pattern (IS NOT THE COMPLETE DOT STRING)

digraph G {
1[label="+"];
2[label="*"];
3[label="X1"];
"""
function _dotstring_labels(node::AbstractNode)
    #get descendants
    des = descendants(node)
    str = "digraph SPN{ \n"

    #Add to str the node label corresponding
    #to each node
    for n in des
        if isa(n, SumNode)
            str = str * string(n.id, "[label = \"+\\n id:$(n.id)\"];\n")
        elseif isa(n, ProductNode)
            str = str * string(n.id, "[label = \"*\\n id:$(n.id)\"];\n")
        elseif isa(n, Union{IndicatorNode, DistributionNode})
            str = str * string(n.id, "[label=\"$(string(n.varname))\\n id:$(n.id)\"];\n")
        else
            str = str * string(n.id, "[label = \"UNKNOW\"];\n")
        end
    end
    str
end

"""
INTERNAL FUNCTION
This function is used for creating the
the DOT string.

Ideally:
node is the root of the SPN
array is an array containing the string created with
the function _dotstring_labels.

array is modified in place and will contain the string
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
Function to create a file in
DOT format
"""
function dotfile(node, filename = "my_spn.gv")
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
