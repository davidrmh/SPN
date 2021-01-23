#=================================
Evolutionary parameter learning
=================================#

"""
    initializepopultion(popsize::Int64)

Create the initial population of SPN.
Return an array containing `popsize` SPNs.

As a proof of concept, the SPN represent a mixture of Gaussians.

# Arguments
- `popsize::Int64` population size
"""
function initializepopulation(popsize = 50::Int64)
    #As a proof of concept the SPN have the same width = 3
    width = 3
    #To sample weights that sum 1
    dirichlet = Dirichlet(repeat(2:2, width))

    #To sample the mu parameter
    unifmu = Uniform(-10, 10)

    #To sample the sigma parameter
    unifsig = Uniform(0, 5)

    population = []
    for i = 1:popsize
        weights = rand(dirichlet)
        mu = rand(unifmu, width)
        sig = rand(unifsig, width)
        spn = normalmixture(weights, mu, sig)
        push!(population, spn)
    end
    population
end

"""
    createmixture(parameters::Array{Any, 1})
Create a SPN  representing a Gaussian mixture using the array `parameters`.

# Argument
- `parameters::Array{Any, 1}` Array with the parameters.
"""
function createmixture(parameters::Array{Any, 1})
    #Order is crucial
    weights = parameters[1:3]
    mu = [parameters[8], parameters[6], parameters[4]]
    sig = [parameters[9], parameters[7], parameters[5]]
    normalmixture(weights, mu, sig)
end

"""
    loglike(spn::AbstractNode, data::DataFrame)
Compute the loglikelihood of a SPN over some data

# Arguments
- `spn::AbstractNode` Root node of the SPN
- `data::DataFrame` Data
"""
function loglike(spn::AbstractNode, data::DataFrame)
    #Observations (as named tuples)
    obs = createnamedtuples(data)
    logl = 0
    for e in obs
        logl = logl + log(pdf(spn, e))
    end
    logl
end
