"""
    LinearDecaySchedule
A schedule that linearly decreases a value from `start` to `stop` in `steps` steps.
if the value is greater or equal to `stop`, it stays constant.

# Constructor 

`LinearDecaySchedule(;start, stop, steps)`
"""
@with_kw struct LinearDecaySchedule{R<:Real} <: Function
    start::R
    stop::R
    steps::Int
end

function (schedule::LinearDecaySchedule)(k)
    rate = (schedule.start - schedule.stop) / schedule.steps
    val = schedule.start - k*rate 
    val = max(schedule.stop, val)
end


"""
    ExplorationPolicy <: Policy
An abstract type for exploration policies.
Sampling from an exploration policy is done using `action(exploration_policy, on_policy, k, state)`.
`k` is a value that is used to determine the exploration parameter. It is usually a training step in a TD-learning algorithm.
"""
abstract type ExplorationPolicy <: Policy end

"""
    loginfo(::ExplorationPolicy, k)
returns information about an exploration policy, e.g. epsilon for e-greedy or temperature for softmax.
It is expected to return a namedtuple (e.g. (temperature=0.5)). `k` is the current training step that is used to compute the exploration parameter.
"""
function loginfo end

"""
    EpsGreedyPolicy <: ExplorationPolicy

represents an epsilon greedy policy, sampling a random action with a probability `eps` or returning an action from a given policy otherwise.
The evolution of epsilon can be controlled using a schedule. This feature is useful for using those policies in reinforcement learning algorithms. 

# Constructor:

`EpsGreedyPolicy(problem::Union{MDP, POMDP}, eps::Union{Function, Float64}; rng=Random.default_rng(), schedule=ConstantSchedule)`

If a function is passed for `eps`, `eps(k)` is called to compute the value of epsilon when calling `action(exploration_policy, on_policy, k, s)`.


# Fields

- `eps::Function`
- `rng::AbstractRNG`
- `m::M` POMDPs or MDPs problem
- `on_policy::P` a policy to use for the greedy part
- `k::Int` the current training step to use for computing eps(k)
"""
mutable struct EpsGreedyPolicy{P<:Union{Nothing,Policy},T<:Function,R<:AbstractRNG,M<:Union{MDP,POMDP}} <: ExplorationPolicy
    on_policy::P
    k::Int
    eps::T
    rng::R
    m::M
end

function EpsGreedyPolicy(problem::Union{MDP,POMDP}, eps::Function;
                         rng::AbstractRNG=Random.default_rng())
    return EpsGreedyPolicy(nothing, 1, eps, rng, problem)
end
function EpsGreedyPolicy(problem::Union{MDP,POMDP}, eps::Real;
                         rng::AbstractRNG=Random.default_rng())
    return EpsGreedyPolicy(problem, x -> eps, rng=rng)
end
function EpsGreedyPolicy(problem::Union{MDP,POMDP}, on_policy::Policy, eps::Function;
    k::Int=1, rng::AbstractRNG=Random.default_rng())
    return EpsGreedyPolicy(on_policy, k, eps, rng, problem)
end
function EpsGreedyPolicy(problem::Union{MDP,POMDP}, on_policy::Policy, eps::Real;
    k::Int=1, rng::AbstractRNG=Random.default_rng())
    return EpsGreedyPolicy(problem, on_policy, x -> eps, k=k, rng=rng)
end

function POMDPs.action(p::EpsGreedyPolicy, on_policy::Policy, k, s)
    if rand(p.rng) < p.eps(k)
        return rand(p.rng, actions(p.m,s))
    else
        return action(on_policy, s)
    end
end
POMDPs.action(p::EpsGreedyPolicy{<:Policy}, s) = action(p, p.on_policy, p.k, s)

loginfo(p::EpsGreedyPolicy, k) = (eps=p.eps(k),)
loginfo(p::EpsGreedyPolicy) = loginfo(p, p.k)

function update!(p::EpsGreedyPolicy, k::Int)
    p.k = k
    return p
end
function update!(p::EpsGreedyPolicy{P}, on_policy::P) where {P<:Policy}
    p.on_policy = on_policy
    return p
end

# softmax 
"""
    SoftmaxPolicy <: ExplorationPolicy

represents a softmax policy, sampling a random action according to a softmax function. 
The softmax function converts the action values of the on policy into probabilities that are used for sampling. 
A temperature parameter or function can be used to make the resulting distribution more or less wide.

# Constructor

`SoftmaxPolicy(problem, temperature::Union{Function, Float64}; rng=Random.default_rng())`

If a function is passed for `temperature`, `temperature(k)` is called to compute the value of the temperature when calling `action(exploration_policy, on_policy, k, s)`

# Fields 

- `temperature::Function`
- `rng::AbstractRNG`
- `actions::A` an indexable list of action

"""
struct SoftmaxPolicy{T<:Function, R<:AbstractRNG, A} <: ExplorationPolicy
    temperature::T
    rng::R
    actions::A
end

function SoftmaxPolicy(problem, temperature::Function; 
                       rng::AbstractRNG=Random.default_rng())
    return SoftmaxPolicy(temperature, rng, actions(problem))
end
function SoftmaxPolicy(problem, temperature::Real; 
                       rng::AbstractRNG=Random.default_rng())
    return SoftmaxPolicy(x->temperature, rng, actions(problem))
end

function POMDPs.action(p::SoftmaxPolicy, on_policy::Policy, k, s)
    vals = actionvalues(on_policy, s)
    vals ./= p.temperature(k)
    maxval = maximum(vals)
    exp_vals = exp.(vals .- maxval)
    exp_vals /= sum(exp_vals)
    return p.actions[sample(p.rng, Weights(exp_vals))]
end

loginfo(p::SoftmaxPolicy, k) = (temperature=p.temperature(k),)
