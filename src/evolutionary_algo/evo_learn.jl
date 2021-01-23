#=================================
Evolutionary parameter learning
=================================#

"""
    initializepopultion(popsize::Int64)

Create the initial population of SPN.
Return an array containing `popsize` SPNs.

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
