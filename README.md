# POMDPs

This package provides a basic interface for working with partially observable Markov decision processes (POMDPs).

Installation:
```julia
Pkg.clone("https://github.com/sisl/POMDPs.jl.git")
```

## Supported Solvers

The following MDP solvers support this interface:
* [Value Iteration](https://github.com/sisl/DiscreteValueIteration.jl)
* [Monte Carlo Tree Search](https://github.com/sisl/MCTS.jl)

The following POMDP solvers support this interface:
* [QMDP](https://github.com/sisl/QMDP.jl)
* [SARSOP](https://github.com/sisl/SARSOP.jl)

To get started, follow the tutorial in [this](http://nbviewer.ipython.org/github/sisl/POMDPs.jl/blob/master/examples/GridWorld.ipynb) notebook.

## Basic Types

The basic types are

- `POMDP`
- `AbstractDistribution`
- `AbstractSpace`
- `Belief`
- `Solver`
- `Policy`
- `Simulator`

## Model functions

- `discount(pomdp::POMDP)` returns the discount
- `states(pomdp::POMDP)` returns the complete state space 
- `actions(pomdp::POMDP)` returns the complete action space
- `actions(pomdp::POMDP, state::Any, aspace::AbstractSpace=actions(pomdp))` modifies `aspace` to the action space accessible from the given state and returns it
- `observations(pomdp::POMDP)` returns the complete observation space
- `observations(pomdp::POMDP, state::Any, ospace::AbstractSpace)` modifies `ospace` to the observation space accessible from the given state and returns it
- `reward(pomdp::POMDP, state::Any, action::Any)` returns the immediate reward for the state-action pair
- `reward(pomdp::POMDP, state::Any, action::Any, statep::Any)` returns the immediate reward for the s-a-s' triple
- `transition(pomdp::POMDP, state, action, distribution=create_transition_distribution(pomdp))` modifies `distribution` to the transition distribution from the current state-action pair and returns it
- `observation(pomdp::POMDP, state, action, distribution=create_observation_distribution(pomdp))` modifies `distribution` to the observation distribution from the current state and *previous* action and returns it
- `isterminal(pomdp::POMDP, state::Any)` checks if a state is terminal
- `create_state(pomdp::POMDP)` creates a single state object (for preallocation purposes)
- `create_observation(pomdp::POMDP)` creates a single observation object (for preallocation purposes)
- `index(pomdp::POMDP, state::State)` returns the index of the given state for a discrete POMDP 


## Distribution Functions

- `rand!(rng::AbstractRNG, sample, d::AbstractDistribution)` fill with random sample from distribution
- `pdf(d::AbstractDistribution, x)` value of probability distribution function at x
- `create_transition_distribution(pomdp::POMDP)` returns a transition distribution
- `create_observation_distribution(pomdp::POMDP)` returns an observation distribution


## Space Functions
- `domain(space::AbstractSpace)` returns an iterator over a space


## Solver functions

- `create_policy(solver::Solver, pomdp::POMDP)` creates a policy object (for preallocation purposes)
- `solve(solver::Solver, pomdp::POMDP, policy::Policy=create_policy(solver, pomdp))` solves the POMDP and modifies `policy` to be the solution of `pomdp` and returns it


## Policy Functions
- `action(pomdp::POMDP, policy::Policy, belief::Belief, action=create_action(pomdp))` returns an action for the current belief given the policy
- `action(pomdp::POMDP, policy::Policy, state::Any, action=create_action(pomdp))` returns an action for the current state given the policy
- `value(policy::Policy, belief::Belief)` returns the expected value for the current belief given the policy
- `value(policy::Policy, state::Any)` returns the expected value for the current state given the policy
- `create_action(pomdp::POMDP)` returns an action (for preallocation purposes)


## Belief Functions
- `create_belief(pomdp::POMDP)` creates a belief object (for preallocation purposes)
- `belief(pomdp::POMDP, belief_old::Belief, action::Any, obs::Any, belief_new::Belief=create_belief(pomdp))` modifies `belief_new` to the belief given the old belief and the latest action and observation and returns the updated belief. `belief_old` and `belief_new` should *not* be references to the same object

## Simulation Functions
- `simulate(simulator::Simulator, pomdp::POMDP, policy::Policy)` runs a simulation using the specified policy and returns the accumulated reward
