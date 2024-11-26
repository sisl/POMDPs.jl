using POMDPs
using DiscreteValueIteration: ValueIterationSolver
using QuickPOMDPs
using POMDPModels: BabyPOMDP, TigerPOMDP, SimpleGridWorld, LegacyGridWorld, RandomMDP, TMaze, Starve, FeedWhenCrying, RandomPOMDP, MiniHallway
import POMDPLinter

using POMDPTools
using Test

import Random
using Random: MersenneTwister, AbstractRNG
using StableRNGs: StableRNG
using SparseArrays: sparse

import CommonRLInterface

using Distributions: Distributions, product_distribution, Product

@testset "POMDPTools.jl" begin
    @testset "POMDPDistributions" begin
        include("distributions/test_bool.jl")
        include("distributions/test_deterministic.jl")
        include("distributions/test_implicit.jl")
        include("distributions/test_pretty_printing.jl")
        include("distributions/test_sparse_cat.jl")
        include("distributions/test_uniform.jl")
        include("distributions/test_distributions_jl_integration.jl")
    end

    @testset "ModelTools" begin
        include("model_tools/test_fully_observable_pomdp.jl")
        include("model_tools/test_generative_belief_mdp.jl")
        include("model_tools/test_info.jl")
        include("model_tools/test_matrices.jl")
        include("model_tools/test_obs_weight.jl")
        include("model_tools/test_ordered_spaces.jl")
        include("model_tools/test_reward_model.jl")
        include("model_tools/test_tabular.jl")
        include("model_tools/test_terminal_state.jl")
        include("model_tools/test_underlying_mdp.jl")
        include("model_tools/test_visualization.jl")
    end

    @testset "BeliefUpdaters" begin
        include("belief_updaters/test_belief.jl")
        include("belief_updaters/test_k_previous_observations_belief.jl")
    end

    @testset "Policies" begin
        include("policies/test_alpha_policy.jl")
        include("policies/test_evaluation.jl")
        include("policies/test_exploration_policies.jl")
        include("policies/test_function_policy.jl")
        include("policies/test_playback_policy.jl")
        include("policies/test_pretty_printing.jl")
        include("policies/test_random_solver.jl")
        include("policies/test_stochastic_policy.jl")
        include("policies/test_utility_wrapper.jl")
        include("policies/test_vector_policy.jl")
        include("policies/test_dict_policy.jl")
    end

    @testset "Simulators" begin
        include("simulators/test_display.jl")
        include("simulators/test_history_recorder.jl")
        include("simulators/test_parallel.jl")
        include("simulators/test_rollout.jl")
        include("simulators/test_sim.jl")
        include("simulators/test_stepthrough.jl")
    end

    @testset "CommonRLIntegration" begin
        include("common_rl_integration/test_common_rl.jl")
    end

    @testset "Testing" begin
        include("testing/runtests.jl")
    end
end
