#========================================
Some Examples
========================================#
function normalmixture(weights = [1/3, 1/3, 1/3], mu = [-5, -2, 2], sig = [0.5, 3, 1])
    #Create the sum node
    sumnode = SumNode([], [], weights)

    #Create each normal component
    components = Array{AbstractNode, 1}(undef, length(weights))
    for i in eachindex(mu)
        varname = Symbol(string("X", i))
        components[i] = DistributionNode(Distributions.Normal(mu[i], sig[i]), [], varname)
    end

    #Connect nodes
    addchildren!(sumnode, components)
    sumnode
end

function naivebayesmixture(weights = [[0.5, 0.2, 0.3], [0.6, 0.4], [0.9, 0.1], [0.3, 0.7], [0.2, 0.8]])
    s1 = SumNode([], [], weights[1]) #root
    s2 = SumNode([], [], weights[2])
    s3 = SumNode([], [], weights[3])
    s4 = SumNode([], [], weights[4])
    s5 = SumNode([], [], weights[5])

    p1 = ProductNode([], [])
    p2 = ProductNode([], [])
    p3 = ProductNode([], [])

    x1 = IndicatorNode([], :X1)
    x1_neg = IndicatorNode([], :X1_neg)
    x2 = IndicatorNode([], :X2)
    x2_neg = IndicatorNode([], :X2_neg)

    addchildren!(s1, [p1, p2, p3])
    addchildren!(p1, [s2, s4])
    addchildren!(p2, [s2, s5])
    addchildren!(p3, [s3, s5])
    addchildren!(s2, [x1, x1_neg])
    addchildren!(s3, [x1, x1_neg])
    addchildren!(s4, [x2, x2_neg])
    addchildren!(s5, [x2, x2_neg])

    s1
end