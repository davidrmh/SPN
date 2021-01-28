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
Combine two candidate solutions
Return a dictionary with the same keys as `params1`.
`params1` and `params2` are expected to contain values in the logspace.
"""
function combinecandidates(params1::Dict{Any, Any}, params2::Dict{Any, Any},
    types1::Dict{Any, Any}, types2::Dict{Any, Any}, crossrate::Float64)

    newparams = Dict()
    newtypes = Dict()
    ty2 = [values(types2)...]
    ky2 = [keys(types2)...]
    memory = []
    unif = Uniform(0, 1)
    for key1 in keys(params1)
        type1 = types1[key1]
        for i in eachindex(ty2)
            #If there is a type match
            if ty2[i] == type1 && !(i in memory)
                newtypes[key1] = type1
                #Combine entries
                newparams[key1] = zeros(length(params1[key1]))
                for j in eachindex(params1[key1])
                    u = rand(unif, 1)[1]
                    if u < crossrate
                        newparams[key1][j] = params2[ky2[i]][j]
                    else
                        newparams[key1][j] = params1[key1][j]
                    end
                end
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
function loglike(spn::AbstractNode, data::Union{Real, AbstractArray, NamedTuple, DataFrame},
    params::Dict{Any, Any}, logspace::Bool)
    sum(log.(pdf(spn, data, params, logspace)))
end

"""
We want to maximize the score function
"""
function differentialevolution(scorefun::Function, data::Union{Real, AbstractArray, NamedTuple, DataFrame};
    popsize= 50::Int64, numiter = 500::Int64, tol = 1e-6::Float64,
    stepsize = 0.65::Float64, crossrate = 0.55::Float64)

    #Initialize population
    pop_spn = initializepopulation(popsize)

    #Parameters of each SPN
    #The parameters will be in logspace
    #and are representd with a dictionary
    pop_par = []
    pop_types = []
    for spn in pop_spn
        parameters, types = getparameters(spn, true)
        push!(pop_par, parameters)
        push!(pop_types, types)
    end

    #Get the score of each SPN in the data
    scores = zeros(popsize)
    for i = 1:popsize
        spn = pop_spn[i]
        par = pop_par[i]
        #parameters are in logspace
        scores[i] = scorefun(spn, data, par, true)
    end

    #Best score
    agmax = argmax(scores)
    best_score = scores[agmax]
    best_prev_score = -Inf

    #Best SPN
    best_spn = pop_spn[agmax]
    best_spn_par = pop_par[agmax]
    best_spn_types = pop_types[agmax]

    #Counter for iterations
    count_iter = 1
    indices = [1:popsize...]
    while count_iter < numiter && abs(best_score - best_prev_score) > tol
        #For each individual in the population
        #Create a new one.
        new_pop_par = [] #New population of parameters
        new_pop_types = [] #New population of types
        for i in indices
            xi_par = deepcopy(pop_par[i])
            xi_types = deepcopy(pop_types[i])
            r1 = rand(setdiff(indices, [i]), 1)[1]
            r2 = rand(setdiff(indices, [i, r1]), 1)[1]
            r3 = rand(setdiff(indices, [i, r1, r2]), 1)[1]
            xr1_par = deepcopy(pop_par[r1])
            xr1_types = deepcopy(pop_types[r1])
            xr2_par = deepcopy(pop_par[r2])
            xr2_types = deepcopy(pop_types[r2])
            xr3_par = deepcopy(pop_par[r3])
            xr3_types = deepcopy(pop_types[r3])
            #Create a new candidate
            #vi = xr1 .+ stepsize .* (xr2 .- xr3)
            vi_par, vi_types = createcandidate(xr2_par, xr3_par, xr2_types, xr3_types, stepsize)
            vi_par, vi_types = createcandidate(xr1_par, vi_par, xr1_types, vi_types)
            vi_par, vi_types = combinecandidates(xi_par, vi_par, xi_types, vi_types, crossrate)

            push!(new_pop_par, vi_par)
            push!(new_pop_types, vi_types)
        end #for i

        #Evaluate the new population
        scores_new = zeros(popsize)
        for i in 1:popsize
            ind_par = new_pop_par[i]
            ind_types = new_pop_types[i]
            spn = createmixture(ind_par, ind_types, true)
            #To avoid problems with the ids of each node
            spn_par, spn_types = getparameters(spn, true)
            scores_new[i] = scorefun(spn, data, spn_par, true)
        end

        #Keep the best SPNs
        for i in 1:popsize
            #Maximization problem
            if scores_new[i] > scores[i]
                pop_par[i] = new_pop_par[i]
                pop_types[i] = new_pop_types[i]
                scores[i] = scores_new[i]
            end
        end

        #Best SPN
        agmax = argmax(scores)
        best_spn_par = pop_par[agmax]
        best_spn_types = pop_types[agmax]
        best_spn = createmixture(best_spn_par, best_spn_types, true)
        best_prev_score = best_score
        best_score = scores[agmax]

        count_iter = count_iter + 1

        if count_iter % 100 == 0
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
