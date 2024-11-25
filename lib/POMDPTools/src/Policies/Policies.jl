module Policies

using LinearAlgebra
using Random
using StatsBase: Weights # for Weights
using SparseArrays # for sparse vectors in alpha_vector.jl
using Parameters
using Distributions # For logpdf extenstion in playback policy

using POMDPs
import POMDPs: action, value, solve, updater

using ..BeliefUpdaters
using ..ModelTools
using ..POMDPDistributions

using Base.Iterators: take

"""
    actionvalues(p::Policy, s)

returns the values of each action at state s in a vector
"""
function actionvalues end

export
    actionvalues

export
    AlphaVectorPolicy,
    alphavectors,
    alphapairs

include("alpha_vector.jl")

export
    FunctionPolicy,
    FunctionSolver

include("function.jl")

export
    RandomPolicy,
    RandomSolver

include("random.jl")

export
    VectorPolicy,
    VectorSolver,
    ValuePolicy

include("vector.jl")

export 
    ValueDictPolicy
    
include("dict.jl")

export
    StochasticPolicy,
    UniformRandomPolicy,
    CategoricalTabularPolicy

include("stochastic.jl")

export LinearDecaySchedule,
       EpsGreedyPolicy,
       SoftmaxPolicy,
       ExplorationPolicy,
       loginfo,
       update!

include("exploration_policies.jl")

export
    PolicyWrapper,
    payload

include("utility_wrapper.jl")

export
    showpolicy

include("pretty_printing.jl")

export
    PlaybackPolicy

include("playback.jl")

export
    evaluate

include("policy_evaluation.jl")

end
