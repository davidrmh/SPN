#=
TO DO:
    Implement a function to set the evidence
    Implement logpdf function
    Implement likelihood function
    Implement loglikelihood function
    Implement sample function
=#
using Distributions
#This global variable is to assign an ID to each node
_idcounter = 0

#Node types definition and constructors
include("nodes_types.jl")

#Methods
include("general_methods.jl")
include("sum_nodes_methods.jl")
include("product_nodes_methods.jl")
include("leaf_nodes_methods.jl")
include("structure_learning.jl")

#Utilities
include("utilities.jl")

#Examples of architectures
include("examples.jl")
