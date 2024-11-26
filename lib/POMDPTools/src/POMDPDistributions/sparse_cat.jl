"""
    SparseCat(values, probabilities)

Create a sparse categorical distribution.

`values` is an iterable object containing the possible values (can be of any type) in the distribution that have nonzero probability. `probabilities` is an iterable object that contains the associated probabilities.

This is optimized for value iteration with a fast implementation of `weighted_iterator`. Both `pdf` and `rand` are order n.
"""
struct SparseCat{V, P, F} <: Distribution{F, Discrete}
    vals::V
    probs::P
end

# handle cases where probs is an array of something other than numbers (issue #35)
function SparseCat(v, p::AbstractArray)
    cp = try
        convert(AbstractArray{Float64}, p)
    catch
        @error("Couldn't convert all probabilities to Float64 when creating a SparseCat distribution. Did you get the arguments in the right order?", values=v, probabilities=p)
        rethrow()
    end
    SparseCat(v, cp)
end
# the method above gets all arrays *except* ones that have a numeric eltype, which are handled below
SparseCat(v, p::AbstractArray{<:Number}) = SparseCat{typeof(v), typeof(p), infer_variate_form(eltype(v))}(v, p)

SparseCat(v, p) = SparseCat{typeof(v), typeof(p), infer_variate_form(eltype(v))}(v, p)

rand(rng::AbstractRNG, s::Random.SamplerTrivial{<:SparseCat}) = rand(rng, s[])

function rand(rng::AbstractRNG, d::SparseCat)
    r = sum(d.probs)*rand(rng)
    tot = zero(eltype(d.probs))
    for (v, p) in weighted_iterator(d)
        tot += p
        if r < tot
            return v
        end
    end
    if sum(d.probs) <= 0.0
        error("""
              Tried to sample from a SparseCat distribution with probabilities that sum to $(sum(d.probs)).

              vals = $(d.vals)

              probs = $(d.probs)
              """)
    end
    error("Error sampling from SparseCat distribution with vals $(d.vals) and probs $(d.probs)") # try to help with type stability
end

Distributions.sampler(d::SparseCat) = Random.SamplerTrivial(d)
Random.Sampler(::AbstractRNG, d::SparseCat, repetition::Union{Val{1}, Val{Inf}}) = Random.SamplerTrivial(d)

# to resolve ambiguity between pdf(::UnivariateDistribution, ::Real) and pdf(::SparseCat, ::Any)
pdf(d::SparseCat, s) = _pdf(d, s)
pdf(d::SparseCat, s::Real) = _pdf(d, s)

Distributions.logpdf(d::SparseCat, x) = log(pdf(d, x))

# slow linear search :(
function _pdf(d::SparseCat, s)
    for (v, p) in weighted_iterator(d)
        if v == s
            return p
        end
    end
    return zero(eltype(d.probs))
end

function _pdf(d::SparseCat{V,P}, s) where {V<:AbstractArray, P<:AbstractArray}
    for (i,v) in enumerate(d.vals)
        if v == s
            return d.probs[i]
        end
    end
    return zero(eltype(d.probs))
end


support(d::SparseCat) = d.vals

struct SparseCatIterator{D<:SparseCat}
    d::D
end


weighted_iterator(d::SparseCat) = SparseCatIterator(d)

# iterator for general SparseCat
# this has some type stability problems
function Base.iterate(i::SparseCatIterator)
    d = i.d
    val, vstate = iterate(d.vals)
    prob, pstate = iterate(d.probs)
    return ((val=>prob), (vstate, pstate))
end
function Base.iterate(i::SparseCatIterator, dstate::Tuple)
    d = i.d
    vstate, pstate = dstate
    vnext = iterate(d.vals, vstate)
    pnext = iterate(d.probs, pstate)
    if vnext === nothing || pnext === nothing
        return nothing 
    end
    val, vstate_next = vnext
    prob, pstate_next = pnext
    return ((val=>prob), (vstate_next, pstate_next))
end

# iterator for SparseCat with indexed members
const Indexed = Union{AbstractArray, Tuple, NamedTuple}

function Base.iterate(i::SparseCatIterator{<:SparseCat{<:Indexed,<:Indexed}}, state::Integer=1)
    if state > length(i)
        return nothing 
    end
    return (i.d.vals[state]=>i.d.probs[state], state+1)
end

Base.length(i::SparseCatIterator) = min(length(i.d.vals), length(i.d.probs))
Base.eltype(D::Type{SparseCatIterator{SparseCat{V,P,F}}}) where {V,P,F} = Pair{eltype(V), eltype(P)}

sampletype(D::Type{SparseCat{V,P,F}}) where {V,P,F} = eltype(V)
Random.gentype(D::Type{SparseCat{V,P,F}}) where {V,P,F} = eltype(V)

function mean(d::SparseCat)
    vsum = zero(eltype(d.vals))
    for (v, p) in weighted_iterator(d)
        vsum += v*p
    end
    return vsum/sum(d.probs)
end

function mode(d::SparseCat)
    bestp = zero(eltype(d.probs))
    bestv = first(d.vals)
    for (v, p) in weighted_iterator(d)
        if p >= bestp
            bestp = p
            bestv = v
        end 
    end
    return bestv
end

Base.show(io::IO, m::MIME"text/plain", d::SparseCat) = showdistribution(io, m, d, title="SparseCat distribution")
