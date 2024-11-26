module POMDPDistributions

import Distributions
import Random
using Random: AbstractRNG

# Should use Module.function directly in the code instead of doing this
import Distributions: support, pdf, mode, mean
using Distributions: DiscreteUnivariateDistribution, Distribution
using Distributions: VariateForm, Multivariate, Matrixvariate, Univariate
using Distributions: ValueSupport, Discrete
import Random: rand

using UnicodePlots: barplot

"""
Try to guess the Distributions.VariateForm for a distribution based on the sample type.
"""
function infer_variate_form(T::Type)
    if T <: AbstractVector
        return Multivariate
    elseif T <: AbstractMatrix
        return Matrixvariate
    elseif T <: Number
        return Univariate
    else
        return VariateForm
    end
end

export
    weighted_iterator
include("weighted_iteration.jl")

export
    SparseCat
include("sparse_cat.jl")

export
    BoolDistribution
include("bool.jl")

export
    Deterministic
include("deterministic.jl")

export
    Uniform,
    UnsafeUniform
include("uniform.jl")

export
    ImplicitDistribution
include("implicit.jl")

export
    showdistribution
include("pretty_printing.jl")

end
