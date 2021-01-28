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
Create a mixture for the given parameters
logspace is true if the parameters are in the logspace
"""
function createmixture(params::Dict{Any, Any}, types::Dict{Any, Any}, logspace::Bool)
    weights = []
    mu = []
    sig = []

    for key in keys(params)
        if types[key] == SumNode
            #If necessary change to positive space
            w = logspace ? exp.(params[key]) : params[key]
            push!(weights, w...)
        elseif types[key] == Normal{Float64}
            m, s = params[key]
            s = logspace ? exp(s) : s
            push!(mu, m)
            push!(sig, s)
        end
    end
    spn = normalmixture(weights, mu, sig)
    #Normalize SPN
    normalize!(spn)
    spn
end

"""
Create one candidate solution by subtracting and multiplying by `stepsize`.
Return a dictionary with the same keys as `params1`.
`params1` and `params2` are expected to contain values in the logspace.
"""
function createcandidate(params1::Dict{Any, Any}, params2::Dict{Any, Any},
    types1::Dict{Any, Any}, types2::Dict{Any, Any}, stepsize::Float64)

    newparams = Dict()
    newtypes = Dict()
    ty2 = [values(types2)...]
    ky2 = [keys(types2)...]
    memory = []
    for key1 in keys(params1)
        type1 = types1[key1]
        for i in eachindex(ty2)
            #If there is a type match
            if ty2[i] == type1 && !(i in memory)
                newparams[key1] = params1[key1] .- params2[ky2[i]]
                newtypes[key1] = type1
                #save in memory to avoid using the same `params2[ky2[i]]`
                push!(memory, i)
                break
            end
        end
    end

    #Multiply by `stepsize`
    for key in keys(newparams)
        newparams[key] = stepsize .* newparams[key]
    end
    newparams, newtypes
end

"""
Create one candidate solution by summing two candidates.
Return a dictionary with the same keys as `params1`.
`params1` and `params2` are expected to contain values in the logspace.
"""
function createcandidate(params1::Dict{Any, Any}, params2::Dict{Any, Any},
    types1::Dict{Any, Any}, types2::Dict{Any, Any})

    newparams = Dict()
    newtypes = Dict()
    ty2 = [values(types2)...]
    ky2 = [keys(types2)...]
    memory = []
    for key1 in keys(params1)
        type1 = types1[key1]
        for i in eachindex(ty2)
            #If there is a type match
            if ty2[i] == type1 && !(i in memory)
                newparams[key1] = params1[key1] .+ params2[ky2[i]]
                newtypes[key1] = type1
                #save in memory to avoid using the same `params2[ky2[i]]`
                push!(memory, i)
                break
            end
        end
    end
    newparams, newtypes
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

"""
Change parameters with positive value constraint to the log-space
"""
function tologspace(parameters::Array{<:Any, 1})
    for i in [1, 2, 3, 5, 7, 9]
        parameters[i] = log(parameters[i])
    end
    parameters
end

"""
Change parameters in log-space into the positive-space
"""
function toposspace(parameters::Array{<:Any, 1})
    for i in [1, 2, 3, 5, 7, 9]
        parameters[i] = exp(parameters[i])
    end
    parameters
end

"""
Score function takes 2 parameters. We want to maximize it.
"""
function differentialevolution(scorefun::Function, data::DataFrame;
    popsize= 50::Int64, numiter = 500::Int64, tol = 1e-6::Float64,
    stepsize = 0.65::Float64, crossrate = 0.55::Float64)

    #Initialize population
    pop_spn = initializepopulation(popsize)

    #Parameters of each SPN
    pop_par = []
    for spn in pop_spn
        parameters, _ = getparameters(spn)
        push!(pop_par, parameters)
    end

    #Get the score of each SPN in the data
    scores = zeros(popsize)
    for i = 1:popsize
        spn = pop_spn[i]
        scores[i] = scorefun(spn, data)
    end
    #Best score
    agmax = argmax(scores)
    best_score = scores[agmax]
    best_prev_score = -Inf

    #Best SPN
    best_spn = pop_spn[agmax]
    best_spn_par = pop_par[agmax]

    #Counter for iterations
    count_iter = 1
    indices = [1:popsize...]
    #Dimension of each parameters vector
    n = length(pop_par[1])
    unif = Uniform(0, 1)
    while count_iter < numiter && abs(best_score - best_prev_score) > tol
        #For each individual in the population
        #Create a new one.
        new_pop_par = [] #New population
        for i in indices
            xi = tologspace(copy(pop_par[i]))
            r1 = rand(setdiff(indices, [i]), 1)[1]
            r2 = rand(setdiff(indices, [i, r1]), 1)[1]
            r3 = rand(setdiff(indices, [i, r1, r2]), 1)[1]
            #mutant vector
            xr1 = tologspace(copy(pop_par[r1]))
            xr2 = tologspace(copy(pop_par[r2]))
            xr3 = tologspace(copy(pop_par[r3]))
            vi = xr1 .+ stepsize .* (xr2 .- xr3)

            #Create new individual
            jr = rand(1:n, 1)[1]
            newind = zeros(n)
            #Modify each dimension in newind
            for j = 1:n
                rcj = rand(unif, 1)[1]
                if rcj < crossrate || j == jr
                    newind[j] = vi[j]
                else
                    newind[j] = xi[j]
                end #if
            end #for j

            #Change to positive-space and
            #add to new population
            newind = toposspace(newind)
            push!(new_pop_par, newind)
        end #for i

        #Evaluate the new population
        scores_new = zeros(popsize)
        for i in 1:popsize
            ind = new_pop_par[i]
            spn = createmixture(ind)
            scores_new[i] = scorefun(spn, data)
        end

        #Keep the best SPNs
        for i in 1:popsize
            #Maximization problem
            if scores_new[i] > scores[i]
                pop_par[i] = new_pop_par[i]
                scores[i] = scores_new[i]
            end
        end

        #Best SPN
        agmax = argmax(scores)
        best_spn_par = pop_par[agmax]
        best_spn = createmixture(best_spn_par)
        best_prev_score = best_score
        best_score = scores[agmax]

        count_iter = count_iter + 1

        if count_iter % 20 == 0
            println("Iteration: $(count_iter). Best score: $(round(best_score, digits = 7))")
        end
    end #while

    dif = abs(best_score - best_prev_score)
    if dif <= tol
        println("Required tolerance achived")
    elseif  count_iter >= numiter
        println("Reached the max number of iterations")
    end

    best_spn

end
