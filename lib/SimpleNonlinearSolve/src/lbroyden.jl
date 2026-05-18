"""
    SimpleLimitedMemoryBroyden(;
        threshold::Union{Val, Int} = Val(27), linesearch = Val(false), alpha = nothing
    )

A limited memory implementation of Broyden. This method applies the L-BFGS scheme to
Broyden's method.

If the threshold is larger than the problem size, then this method will use `SimpleBroyden`.

### Keyword Arguments:

  - `linesearch`: `Val(true)` uses `LiFukushimaLineSearch`, `Val(false)` disables
    line search, or pass any `LineSearch.AbstractLineSearchAlgorithm` for a custom one.
    For `StaticArray`/GPU problems the line search must have a static (non-allocating)
    dispatch — currently `LiFukushimaLineSearch(; nan_maxiters = nothing)` and
    `StrongWolfeLineSearch()` qualify. `StrongWolfeLineSearch` additionally requires
    a `grad_f` keyword forwarded through `solve`. Other algorithms (`BackTracking`,
    `GoldenSection`, …) work on `Array` inputs but allocate on `StaticArray`.
  - `alpha`: Scale the initial jacobian initialization with `alpha`. If it is `nothing`, we
    will compute the scaling using `2 * norm(fu) / max(norm(u), true)`.

!!! warning

    Currently `alpha` is only used for StaticArray problems. This will be fixed in the
    future.
"""
@concrete struct SimpleLimitedMemoryBroyden <: AbstractSimpleNonlinearSolveAlgorithm
    linesearch
    threshold <: Val
    alpha
end

function SimpleLimitedMemoryBroyden(;
        threshold::Union{Val, Int} = Val(27),
        linesearch::Union{Bool, Val{true}, Val{false}, AbstractLineSearchAlgorithm} = Val(false),
        alpha = nothing
    )
    linesearch = linesearch isa Bool ? Val(linesearch) : linesearch
    threshold = threshold isa Int ? Val(threshold) : threshold
    return SimpleLimitedMemoryBroyden(linesearch, threshold, alpha)
end

function SciMLBase.__solve(
        prob::ImmutableNonlinearProblem, alg::SimpleLimitedMemoryBroyden,
        args...; termination_condition = nothing, verbose = NonlinearVerbosity(), kwargs...
    )
    if prob.u0 isa SArray
        if termination_condition === nothing ||
                termination_condition isa NonlinearSolveBase.AbsNormTerminationMode
            return internal_static_solve(
                prob, alg, args...; termination_condition, kwargs...
            )
        end

        if verbose isa Bool
            if verbose
                verbose = NonlinearVerbosity()
            else
                verbose = NonlinearVerbosity(None())
            end
        elseif verbose isa AbstractVerbosityPreset
            verbose = NonlinearVerbosity(verbose)
        end

        @SciMLMessage("Specifying `termination_condition = $(termination_condition)` for \
               `SimpleLimitedMemoryBroyden` with `SArray` is not non-allocating. Use \
               either `termination_condition = AbsNormTerminationMode(Base.Fix2(norm, Inf))` \
               or `termination_condition = nothing`.", verbose, :termination_condition)
    end
    return internal_generic_solve(prob, alg, args...; termination_condition, kwargs...)
end

@views function internal_generic_solve(
        prob::ImmutableNonlinearProblem, alg::SimpleLimitedMemoryBroyden,
        args...; abstol = nothing, reltol = nothing, maxiters = 1000,
        alias::Union{Nothing, SciMLBase.NonlinearAliasSpecifier} = nothing,
        alias_u0 = false,
        termination_condition = nothing, kwargs...
    )
    # Extract alias_u0: if alias struct provided, use it; otherwise use alias_u0 kwarg
    _alias_u0 = alias === nothing ? alias_u0 : Utils.get_alias_u0(alias, alias_u0)
    x = NLBUtils.maybe_unaliased(prob.u0, _alias_u0)
    η = min(NLBUtils.unwrap_val(alg.threshold), maxiters)

    # For scalar problems / if the threshold is larger than problem size just use Broyden
    if x isa Number || length(x) ≤ η
        sol = SciMLBase.__solve(
            prob, SimpleBroyden(; alg.linesearch), args...;
            abstol, reltol, maxiters, termination_condition, kwargs...
        )
        return Utils.nonlinear_solution_new_alg(sol, alg)
    end

    fx = NLBUtils.evaluate_f(prob, x)

    U, Vᵀ = init_low_rank_jacobian(x, fx, x isa StaticArray ? alg.threshold : Val(η))

    abstol, reltol,
        tc_cache = NonlinearSolveBase.init_termination_cache(
        prob, abstol, reltol, fx, x, termination_condition, Val(:simple)
    )

    @bb xo = copy(x)
    @bb δx = copy(fx)
    @bb δx .*= -1
    @bb fo = copy(fx)
    @bb δf = copy(fx)

    @bb vᵀ_cache = copy(x)
    Tcache = lbroyden_threshold_cache(x, x isa StaticArray ? alg.threshold : Val(η))
    @bb mat_cache = copy(x)

    ls_cache = if alg.linesearch isa Val{true}
        init(prob, LiFukushimaLineSearch(; nan_maxiters = nothing), fx, x)
    elseif alg.linesearch isa Val{false}
        nothing
    else
        init(prob, alg.linesearch, fx, x; kwargs...)
    end

    for i in 1:maxiters
        if ls_cache === nothing
            α = true
        else
            ls_sol = solve!(ls_cache, xo, δx)
            α = ls_sol.step_size # Ignores the return code for now
        end

        @bb @. x = xo + α * δx
        fx = NLBUtils.evaluate_f!!(prob, fx, x)
        @bb @. δf = fx - fo

        # Termination Checks
        solved, retcode, fx_sol, x_sol = Utils.check_termination(tc_cache, fx, x, xo, prob)
        solved && return SciMLBase.build_solution(prob, alg, x_sol, fx_sol; retcode)

        Uₚ = selectdim(U, 2, 1:min(η, i - 1))
        Vᵀₚ = selectdim(Vᵀ, 1, 1:min(η, i - 1))

        vᵀ = rmatvec!!(vᵀ_cache, Tcache, Uₚ, Vᵀₚ, δx)
        mvec = matvec!!(mat_cache, Tcache, Uₚ, Vᵀₚ, δf)
        d = dot(vᵀ, δf)
        @bb @. δx = (δx - mvec) / d

        selectdim(U, 2, mod1(i, η)) .= NLBUtils.safe_vec(δx)
        selectdim(Vᵀ, 1, mod1(i, η)) .= NLBUtils.safe_vec(vᵀ)

        Uₚ = selectdim(U, 2, 1:min(η, i))
        Vᵀₚ = selectdim(Vᵀ, 1, 1:min(η, i))
        δx = matvec!!(δx, Tcache, Uₚ, Vᵀₚ, fx)
        @bb @. δx *= -1

        @bb copyto!(xo, x)
        @bb copyto!(fo, fx)
    end

    return SciMLBase.build_solution(prob, alg, x, fx; retcode = ReturnCode.MaxIters)
end

# Non-allocating StaticArrays version of SimpleLimitedMemoryBroyden is actually quite
# finicky, so we'll implement it separately from the generic version
# Ignore termination_condition. Don't pass things into internal functions
function internal_static_solve(
        prob::ImmutableNonlinearProblem{<:SArray}, alg::SimpleLimitedMemoryBroyden, args...;
        abstol = nothing, maxiters = 1000, kwargs...
    )
    x = prob.u0
    fx = NLBUtils.evaluate_f(prob, x)

    U, Vᵀ = init_low_rank_jacobian(vec(x), vec(fx), alg.threshold)

    abstol = NonlinearSolveBase.get_tolerance(x, abstol, eltype(x))

    xo, δx, fo, δf = x, -fx, fx, fx

    ls_cache = if alg.linesearch === Val(true)
        init(prob, LiFukushimaLineSearch(; nan_maxiters = nothing), fx, x)
    elseif alg.linesearch === Val(false)
        nothing
    else
        init(prob, alg.linesearch, fx, x; kwargs...)
    end

    T = promote_type(eltype(x), eltype(fx))
    if alg.alpha === nothing
        fx_norm = L2_NORM(fx)
        x_norm = L2_NORM(x)
        init_α = ifelse(fx_norm ≥ 1.0e-5, max(x_norm, T(true)) / (2 * fx_norm), T(true))
    else
        init_α = inv(alg.alpha)
    end

    converged,
        res = internal_unrolled_lbroyden_initial_iterations(
        prob, xo, fo, δx, abstol, U, Vᵀ, alg.threshold, ls_cache, init_α
    )

    converged && return SciMLBase.build_solution(
        prob, alg, res.x, res.fx; retcode = ReturnCode.Success
    )

    xo, fo, δx = res.x, res.fx, res.δx

    for i in 1:(maxiters - NLBUtils.unwrap_val(alg.threshold))
        if ls_cache === nothing
            α = true
        else
            ls_sol = solve!(ls_cache, xo, δx)
            α = ls_sol.step_size # Ignores the return code for now
        end

        x = xo + α * δx
        fx = NLBUtils.evaluate_f!!(prob, fx, x)
        δf = fx - fo

        maximum(abs, fx) ≤ abstol &&
            return SciMLBase.build_solution(prob, alg, x, fx; retcode = ReturnCode.Success)

        vᵀ = NLBUtils.restructure(x, rmatvec!!(U, Vᵀ, vec(δx), init_α))
        mvec = NLBUtils.restructure(x, matvec!!(U, Vᵀ, vec(δf), init_α))

        d = dot(vᵀ, δf)
        δx = @. (δx - mvec) / d

        U = Base.setindex(U, vec(δx), mod1(i, NLBUtils.unwrap_val(alg.threshold)))
        Vᵀ = Base.setindex(Vᵀ, vec(vᵀ), mod1(i, NLBUtils.unwrap_val(alg.threshold)))

        δx = -NLBUtils.restructure(fx, matvec!!(U, Vᵀ, vec(fx), init_α))

        xo, fo = x, fx
    end

    return SciMLBase.build_solution(prob, alg, xo, fo; retcode = ReturnCode.MaxIters)
end

@generated function internal_unrolled_lbroyden_initial_iterations(
        prob, xo, fo, δx, abstol, U, Vᵀ, ::Val{threshold}, ls_cache, init_α
    ) where {threshold}
    calls = []
    for i in 1:threshold
        static_idx, static_idx_p1 = Val(i - 1), Val(i)
        push!(
            calls, quote
                if ls_cache === nothing
                    α = true
                else
                    ls_sol = solve!(ls_cache, xo, δx)
                    α = ls_sol.step_size # Ignores the return code for now
                end
                x = xo .+ α .* δx
                fx = prob.f(x, prob.p)
                δf = fx - fo

                maximum(abs, fx) ≤ abstol && return true, (; x, fx, δx)

                Uₚ = first_n_getindex(U, $(static_idx))
                Vᵀₚ = first_n_getindex(Vᵀ, $(static_idx))

                vᵀ = NLBUtils.restructure(x, rmatvec!!(Uₚ, Vᵀₚ, vec(δx), init_α))
                mvec = NLBUtils.restructure(x, matvec!!(Uₚ, Vᵀₚ, vec(δf), init_α))

                d = dot(vᵀ, δf)
                δx = @. (δx - mvec) / d

                U = Base.setindex(U, vec(δx), $(i))
                Vᵀ = Base.setindex(Vᵀ, vec(vᵀ), $(i))

                Uₚ = first_n_getindex(U, $(static_idx_p1))
                Vᵀₚ = first_n_getindex(Vᵀ, $(static_idx_p1))
                δx = -NLBUtils.restructure(fx, matvec!!(Uₚ, Vᵀₚ, vec(fx), init_α))

                x0, fo = x, fx
            end
        )
    end
    push!(
        calls, quote
            # Termination Check
            maximum(abs, fx) ≤ abstol && return true, (; x, fx, δx)

            return false, (; x, fx, δx)
        end
    )
    return Expr(:block, calls...)
end

function rmatvec!!(y, xᵀU, U, Vᵀ, x)
    # xᵀ × (-I + UVᵀ)
    η = size(U, 2)
    if η == 0
        @bb @. y = -x
        return y
    end
    x_ = vec(x)
    xᵀU_ = view(xᵀU, 1:η)
    @bb xᵀU_ = transpose(U) × x_
    @bb y = transpose(Vᵀ) × vec(xᵀU_)
    @bb @. y -= x
    return y
end

rmatvec!!(::Nothing, Vᵀ, x, init_α) = -x .* init_α
rmatvec!!(U, Vᵀ, x, init_α) = fast_mapTdot(fast_mapdot(x, U), Vᵀ) .- x .* init_α

function matvec!!(y, Vᵀx, U, Vᵀ, x)
    # (-I + UVᵀ) × x
    η = size(U, 2)
    if η == 0
        @bb @. y = -x
        return y
    end
    x_ = vec(x)
    Vᵀx_ = view(Vᵀx, 1:η)
    @bb Vᵀx_ = Vᵀ × x_
    @bb y = U × vec(Vᵀx_)
    @bb @. y -= x
    return y
end

@inline matvec!!(::Nothing, Vᵀ, x, init_α) = -x .* init_α
@inline matvec!!(U, Vᵀ, x, init_α) = fast_mapTdot(fast_mapdot(x, Vᵀ), U) .- x .* init_α

function fast_mapdot(x::SVector{S1}, Y::SVector{S2, <:SVector{S1}}) where {S1, S2}
    return map(Base.Fix1(dot, x), Y)
end
@generated function fast_mapTdot(
        x::SVector{S1}, Y::SVector{S1, <:SVector{S2}}
    ) where {S1, S2}
    calls = []
    syms = [gensym("m$(i)") for i in 1:S1]
    for i in 1:S1
        push!(calls, :($(syms[i]) = x[$(i)] .* Y[$i]))
    end
    push!(calls, :(return .+($(syms...))))
    return Expr(:block, calls...)
end

@generated function first_n_getindex(x::SVector{L, T}, ::Val{N}) where {L, T, N}
    @assert N ≤ L
    getcalls = ntuple(i -> :(x[$i]), N)
    N == 0 && return :(return nothing)
    return :(return SVector{$N, $T}(($(getcalls...))))
end

function lbroyden_threshold_cache(x, ::Val{threshold}) where {threshold}
    return NLBUtils.safe_similar(x, threshold)
end
function lbroyden_threshold_cache(x::StaticArray, ::Val{threshold}) where {threshold}
    return zeros(MArray{Tuple{threshold}, eltype(x)})
end
lbroyden_threshold_cache(::SArray, ::Val{threshold}) where {threshold} = nothing

function init_low_rank_jacobian(
        u::StaticArray{S1, T1}, fu::StaticArray{S2, T2}, ::Val{threshold}
    ) where {S1, S2, T1, T2, threshold}
    T = promote_type(T1, T2)
    fuSize, uSize = Size(fu), Size(u)
    Vᵀ = MArray{Tuple{threshold, prod(uSize)}, T}(undef)
    U = MArray{Tuple{prod(fuSize), threshold}, T}(undef)
    return U, Vᵀ
end
@generated function init_low_rank_jacobian(
        u::SVector{Lu, T1}, fu::SVector{Lfu, T2}, ::Val{threshold}
    ) where {Lu, Lfu, T1, T2, threshold}
    T = promote_type(T1, T2)
    inner_inits_Vᵀ = [:(zeros(SVector{$Lu, $T})) for i in 1:threshold]
    inner_inits_U = [:(zeros(SVector{$Lfu, $T})) for i in 1:threshold]
    return quote
        Vᵀ = SVector($(inner_inits_Vᵀ...))
        U = SVector($(inner_inits_U...))
        return U, Vᵀ
    end
end
function init_low_rank_jacobian(u, fu, ::Val{threshold}) where {threshold}
    Vᵀ = NLBUtils.safe_similar(u, threshold, length(u))
    U = NLBUtils.safe_similar(u, length(fu), threshold)
    return U, Vᵀ
end
