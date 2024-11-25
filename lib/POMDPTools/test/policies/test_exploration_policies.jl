using POMDPModels

problem =  SimpleGridWorld()
# e greedy
policy = EpsGreedyPolicy(problem, 0.5)
a = first(actions(problem))
@inferred action(policy, FunctionPolicy(s->a::Symbol), 1, GWPos(1,1))
policy = EpsGreedyPolicy(problem, 0.0)
@test action(policy, FunctionPolicy(s->a), 1, GWPos(1,1)) == a
policy = EpsGreedyPolicy(problem, FunctionPolicy(s->a), 0.0)
@test action(policy, GWPos(1,1)) == a

# softmax
policy = SoftmaxPolicy(problem, 0.5)
@test loginfo(policy, 1).temperature == 0.5
on_policy = ValuePolicy(problem)
@inferred action(policy, on_policy, 1, GWPos(1,1))

# test linear schedule
schedule = LinearDecaySchedule(start=1.0, stop=0.0, steps=10)
policy = EpsGreedyPolicy(problem, FunctionPolicy(s->a), schedule)
for i=1:11
    action(policy, FunctionPolicy(s->a), i, GWPos(1,1))
    @test policy.eps(i) < 1.0
    @test loginfo(policy, i).eps == policy.eps(i)
end
@test policy.eps(11) ≈ 0.0
update!(policy, 11)
@test policy.eps(policy.k) ≈ 0.0
@test action(policy, FunctionPolicy(s->a), 11, GWPos(1,1)) == action(policy, GWPos(1,1))
