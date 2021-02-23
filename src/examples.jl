#========================================
Some Examples
========================================#
"""
Mixture of gaussian distributions.
"""
function normalmixture(weights = [1/3, 1/3, 1/3], mu = [-5, -2, 2], sig = [0.5, 3, 1])
    #Create the sum node
    sumnode = SumNode([], [undef], weights)

    #Create each normal component
    components = Array{AbstractNode, 1}(undef, length(weights))
    for i in eachindex(mu)
        varname = Symbol(:X)
        components[i] = DistributionNode(Distributions.Normal(mu[i], sig[i]), [], varname)
    end

    #Connect nodes
    addchildren!(sumnode, components)
    sumnode
end

"""
Naive bayes architecture
"""
function naivebayesmixture(weights = [[0.5, 0.2, 0.3], [0.6, 0.4], [0.9, 0.1], [0.3, 0.7], [0.2, 0.8]])
    s1 = SumNode([], [undef], weights[1]) #root
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

"""
Independent variables.
This example is a generalization of figure 1a) from  the paper
`Learning Selective Sum-Product Networks by Peharz, R. et al`.
"""
function independent(num_var = 3)
    #root node
    root = ProductNode([], [undef])

    for i = 1:num_var
        #each sum node has
        #associated a binary variable
        s = SumNode([], [], [0.5, 0.5])

        #children
        x_name = Symbol(string("X", i, "_1"))
        #Negation of x
        x_neg_name = Symbol(string("X", i, "_2"))
        x = IndicatorNode([], x_name)
        x_neg = IndicatorNode([], x_neg_name)

        #Connect
        addchildren!(root, [s])
        addchildren!(s, [x, x_neg])
    end
    return root
end

function example1()
    root = SumNode([], [undef], [1/3, 1/3, 1/3])
    p1 = ProductNode([], [])
    p2 = ProductNode([], [])
    p3 = ProductNode([], [])

    s1 = SumNode([], [], [0.5, 0.5])
    s2 = SumNode([], [], [0.5, 0.5])
    s3 = SumNode([], [], [0.5, 0.5])
    s4 = SumNode([], [], [0.5, 0.5])

    d1 = IndicatorNode([], :X1_1)
    d2 = IndicatorNode([], :X1_0)
    d3 = IndicatorNode([], :X2_1)
    d4 = IndicatorNode([], :X2_0)
    d5 = IndicatorNode([], :X1_1)
    d6 = IndicatorNode([], :X1_0)
    d7 = IndicatorNode([], :X2_1)
    d8 = IndicatorNode([], :X2_0)

    #Connect
    addchildren!(root, [p1, p2, p3])
    addchildren!(p1, [s1, s2])
    addchildren!(p2, [s2, s3])
    addchildren!(p3, [s3, s4])
    addchildren!(s1, [d1, d2])
    addchildren!(s2, [d3, d4])
    addchildren!(s3, [d5, d6])
    addchildren!(s4, [d7, d8])

    return root

end

"""
This example is to test the function collapseproducts!
It creates a SPN that is nor decomposable nor complete.
"""
function chainproducts()
    #Root
    n1 = SumNode([], [undef], [1/3, 1/3, 1/3])
    n2 = ProductNode([], [])
    n3 = ProductNode([], [])
    n4 = ProductNode([], [])
    n5 = ProductNode([], [])
    n6 = ProductNode([], [])
    n7 = ProductNode([], [])
    n8 = ProductNode([], [])
    n9 = ProductNode([], [])
    n10 = IndicatorNode([], :X1)
    n11 = IndicatorNode([], :X1)
    n12 = IndicatorNode([], :X1)
    n13 = IndicatorNode([], :X1)
    n14 = IndicatorNode([], :X1)
    n15 = IndicatorNode([], :X1)
    n16 = IndicatorNode([], :X1)
    n17 = IndicatorNode([], :X1)
    n18 = IndicatorNode([], :X1)

    #Connect
    addchildren!(n1, [n2, n3, n5])
    addchildren!(n2, [n4])
    addchildren!(n3, [n6, n7])
    addchildren!(n4, [n10])
    addchildren!(n5, [n4, n6])
    addchildren!(n6, [n8, n11, n12])
    addchildren!(n7, [n9, n13, n14])
    addchildren!(n8, [n15, n16])
    addchildren!(n9, [n17, n18])
    n1
end

"""
This example corresponds to figure 2c) from  the paper
`Learning Selective Sum-Product Networks by Peharz, R. et al`.
"""
function regularselective()
    #root
    n1 = SumNode([], [undef], [0.7, 0.3])
    n2 = SumNode([], [], [0.5, 0.5])
    n3 = ProductNode([], [])
    n4 = ProductNode([], [])
    n5 = ProductNode([], [])
    n6 = SumNode([], [], [0.2, 0.8])
    n7 = ProductNode([], [])
    n8 = ProductNode([], [])
    n9 = SumNode([], [], [0.3, 0.7])
    n10 = SumNode([], [], [0.1, 0.9])
    n11 = SumNode([], [], [0.2, 0.8])
    n12 = SumNode([], [], [0.5, 0.5])
    n13 = SumNode([], [], [0.6, 0.4])
    n14 = SumNode([], [], [0.55, 0.45])
    n15 = IndicatorNode([], :Z_2)
    n16 = IndicatorNode([], :Z_1)
    n17 = IndicatorNode([], :Y_2)
    n18 = IndicatorNode([], :Y_1)
    n19 = IndicatorNode([], :X_3)
    n20 = IndicatorNode([], :X_2)
    n21 = IndicatorNode([], :X_1)

    #Connect
    addchildren!(n1, [n2, n3])
    addchildren!(n2, [n4, n5])
    addchildren!(n3, [n21, n13, n12])
    addchildren!(n4, [n17, n6])
    addchildren!(n5, [n18, n14, n11])
    addchildren!(n6, [n8, n7])
    addchildren!(n7, [n9, n19])
    addchildren!(n8, [n10, n20])
    addchildren!(n9, [n16, n15])
    addchildren!(n10, [n16, n15])
    addchildren!(n11, [n16, n15])
    addchildren!(n12, [n16, n15])
    addchildren!(n13, [n17, n18])
    addchildren!(n14, [n19, n20])
    n1
end
