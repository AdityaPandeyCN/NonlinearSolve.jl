"""
    SimpleBroyden(; linesearch = Val(false), alpha = nothing)

A low-overhead implementation of Broyden. This method is non-allocating on scalar and static
array problems.

### Keyword Arguments

  - `linesearch`: `Val(true)` uses `LiFukushimaLineSearch`, `Val(false)` disables
    line search, or pass any `LineSearch.AbstractLineSearchAlgorithm` for a custom one.
    For `StaticArray`/GPU problems the line search must have a static (non-allocating)
    dispatch — currently `LiFukushimaLineSearch(; nan_maxiters = nothing)` and
    `StrongWolfeLineSearch()` qualify. `StrongWolfeLineSearch` additionally requires
    a `grad_f` keyword forwarded through `solve`. Other algorithms (`BackTracking`,
    `GoldenSection`, …) work on `Array` inputs but allocate on `StaticArray`.
  - `alpha`: Scale the initial jacobian initialization with `alpha`. If it is `nothing`, we
    will compute the scaling using `2 * norm(fu) / max(norm(u), true)`.
"""
@concrete struct SimpleBroyden <: AbstractSimpleNonlinearSolveAlgorithm
    linesearch
    alpha
end

function SimpleBroyden(;
        linesearch::Union{Bool, Val{true}, Val{false}, AbstractLineSearchAlgorithm} = Val(false),
        alpha = nothing
    )
    linesearch = linesearch isa Bool ? Val(linesearch) : linesearch
    return SimpleBroyden(linesearch, alpha)
end

function SciMLBase.__solve(
        prob::ImmutableNonlinearProblem, alg::SimpleBroyden, args...;
        abstol = nothing, reltol = nothing, maxiters = 1000,
        alias::Union{Nothing, SciMLBase.NonlinearAliasSpecifier} = nothing,
        alias_u0 = false,
        termination_condition = nothing, kwargs...
    )
    # Extract alias_u0: if alias struct provided, use it; otherwise use alias_u0 kwarg
    _alias_u0 = alias === nothing ? alias_u0 : Utils.get_alias_u0(alias, alias_u0)
    x = NLBUtils.maybe_unaliased(prob.u0, _alias_u0)
    fx = NLBUtils.evaluate_f(prob, x)
    T = promote_type(eltype(fx), eltype(x))

    iszero(fx) &&
        return SciMLBase.build_solution(prob, alg, x, fx; retcode = ReturnCode.Success)

    @bb xo = copy(x)
    @bb δx = similar(x)
    @bb δf = copy(fx)
    @bb fprev = copy(fx)

    if alg.alpha === nothing
        fx_norm = L2_NORM(fx)
        x_norm = L2_NORM(x)
        init_α = ifelse(fx_norm ≥ 1.0e-5, max(x_norm, T(true)) / (2 * fx_norm), T(true))
    else
        init_α = inv(alg.alpha)
    end

    J⁻¹ = Utils.identity_jacobian(fx, x, init_α)
    @bb J⁻¹δf = copy(x)
    @bb xᵀJ⁻¹ = copy(x)
    @bb δJ⁻¹n = copy(x)
    @bb δJ⁻¹ = copy(J⁻¹)

    abstol, reltol,
        tc_cache = NonlinearSolveBase.init_termination_cache(
        prob, abstol, reltol, fx, x, termination_condition, Val(:simple)
    )

    ls_cache = if alg.linesearch isa Val{true}
        init(prob, LiFukushimaLineSearch(; nan_maxiters = nothing), fx, x)
    elseif alg.linesearch isa Val{false}
        nothing
    else
        init(prob, alg.linesearch, fx, x; kwargs...)
    end

    for _ in 1:maxiters
        @bb δx = J⁻¹ × vec(fprev)
        @bb δx .*= -1

        if ls_cache === nothing
            α = true
        else
            ls_sol = solve!(ls_cache, xo, δx)
            α = ls_sol.step_size # Ignores the return code for now
        end

        @bb @. x = xo + α * δx
        fx = NLBUtils.evaluate_f!!(prob, fx, x)
        @bb @. δf = fx - fprev

        # Termination Checks
        solved, retcode, fx_sol, x_sol = Utils.check_termination(tc_cache, fx, x, xo, prob)
        solved && return SciMLBase.build_solution(prob, alg, x_sol, fx_sol; retcode)

        @bb J⁻¹δf = J⁻¹ × vec(δf)
        d = dot(δx, J⁻¹δf)
        @bb xᵀJ⁻¹ = transpose(J⁻¹) × vec(δx)

        @bb @. δJ⁻¹n = (δx - J⁻¹δf) / d

        δJ⁻¹n_ = NLBUtils.safe_vec(δJ⁻¹n)
        xᵀJ⁻¹_ = NLBUtils.safe_vec(xᵀJ⁻¹)
        @bb δJ⁻¹ = δJ⁻¹n_ × transpose(xᵀJ⁻¹_)
        @bb J⁻¹ .+= δJ⁻¹

        @bb copyto!(xo, x)
        @bb copyto!(fprev, fx)
    end

    return SciMLBase.build_solution(prob, alg, x, fx; retcode = ReturnCode.MaxIters)
end
