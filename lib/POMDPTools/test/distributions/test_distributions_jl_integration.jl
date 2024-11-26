@test POMDPDistributions.infer_variate_form(typeof([1 2; 3 4])) == Distributions.Matrixvariate
@test POMDPDistributions.infer_variate_form(typeof([1, 2])) == Distributions.Multivariate
@test POMDPDistributions.infer_variate_form(typeof(1)) == Distributions.Univariate
@test POMDPDistributions.infer_variate_form(Any) == Distributions.VariateForm

p = product_distribution([SparseCat([1, 2, 3], [0.5, 0.2, 0.3]), BoolDistribution(1.0)])
@test rand(p) isa AbstractVector
@test pdf(p, [1, 1]) == 0.5

@test_broken p = Product([SparseCat([:a,:b,:c], [0.5, 0.2, 0.3]), BoolDistribution(1.0)])
