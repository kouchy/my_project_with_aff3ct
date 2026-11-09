# Generic Modular LDPC Belief Propagation Decoder in Julia for StreamPU (`Stateful_Julia`)
# Tier 2 (`ldpc_bp_std.jl`): Idiomatic, Zero-Allocation High-Performance Julia (~330 lines).
#
# Uses standard Julia performance idioms without any low-level compiler intrinsics:
#   - Zero `Base.llvmcall`, zero `@generated` functions, zero `VecElement`, and zero raw `Ptr` arithmetic.
#   - Parametric types and multiple dispatch over 4 orthogonal axes (`Schedule`, `Rule`, `T`, `Mode`).
#   - Pre-allocated per-thread `LDPC_Workspace` buffers (zero GC allocations on the hot path).
#   - Single unified kernel family parameterized by `Val(W)` (`Val(1)` for `ScalarMode`, `Val(W)` for `SimdInterMode`)
#     using immutable `ntuple` + dot-broadcasting (`.+`, `min.`, `ifelse.`) so LLVM keeps accumulators in registers.

abstract type AbstractSchedule end
struct ScheduleFlooding <: AbstractSchedule end
struct ScheduleLayered  <: AbstractSchedule end

abstract type AbstractUpdateRule end
struct RuleSPA <: AbstractUpdateRule end
struct RuleOMS{T<:Union{Float32, Int16}} <: AbstractUpdateRule
    offset::T
end

abstract type AbstractExecMode end
struct ScalarMode       <: AbstractExecMode end
struct SimdInterMode{W} <: AbstractExecMode end
SimdInterMode(W::Integer = 1) = SimdInterMode{Int(W)}()

@inline _max_val(::Type{Float32}) = floatmax(Float32)
@inline _max_val(::Type{Int16})   = typemax(Int16)

@inline _saturate(x::Float32, ::Float32)::Float32 = x
@inline _saturate(x::Int16,   sat::Int16)::Int16  = clamp(x, -sat, sat)
@inline _saturate(x::Int32,   sat::Int16)::Int16  = Int16(clamp(x, Int32(-sat), Int32(sat)))

@inline _add_sat(a::Float32, b::Float32, ::Float32 = 0.0f0)::Float32 = a + b
@inline _add_sat(a::Int16,   b::Int16,   sat::Int16 = typemax(Int16))::Int16 =
    _saturate(Int32(a) + Int32(b), sat)

@inline _sub_sat(a::Float32, b::Float32)::Float32 = a - b
@inline _sub_sat(a::Int16,   b::Int16)::Int16     = _saturate(Int32(a) - Int32(b), typemax(Int16))

struct LDPC_Graph{S<:AbstractSchedule, U<:AbstractUpdateRule, T<:Union{Float32, Int16}}
    schedule::S
    rule::U
    sat_val::T
    n_ite::Int32
    enable_syndrome::Bool
    syndrome_depth::Int32
    var_offsets::Vector{Int32}
    chk_offsets::Vector{Int32}
    transpose::Vector{Int32}
    chk_to_var::Vector{Int32}
    info_bits_pos::Vector{Int32}
end

function LDPC_Graph(
    sched_sym::Symbol, rule_sym::Symbol, ::Type{T},
    offset_val::Real, sat_val_in::Real,
    n_ite::Int32, enable_syndrome::Bool, syndrome_depth::Int32,
    n_var_nodes::Int32, n_chk_nodes::Int32,
    h_cols_addr::UInt, h_cols_len::Int32,
    info_bits_addr::UInt, K::Int32
) where {T<:Union{Float32, Int16}}
    h_cols        = unsafe_wrap(Array, Ptr{UInt32}(h_cols_addr),    Int(h_cols_len))
    info_bits_raw = unsafe_wrap(Array, Ptr{UInt32}(info_bits_addr), Int(K))

    nv, nc      = Int(n_var_nodes), Int(n_chk_nodes)
    var_degrees = zeros(Int32, nv)
    chk_offsets = Vector{Int32}(undef, nc + 1)
    chk_offsets[1] = Int32(1)
    idx = 1
    for c in 1:nc
        deg = Int32(h_cols[idx]); idx += 1
        chk_offsets[c + 1] = chk_offsets[c] + deg
        for _ in 1:deg
            v = Int(h_cols[idx]) + 1; idx += 1
            var_degrees[v] += Int32(1)
        end
    end

    var_offsets = Vector{Int32}(undef, nv + 1)
    var_offsets[1] = Int32(1)
    for v in 1:nv
        var_offsets[v + 1] = var_offsets[v] + var_degrees[v]
    end

    n_branches  = Int(chk_offsets[end] - 1)
    transpose   = Vector{Int32}(undef, n_branches)
    chk_to_var  = Vector{Int32}(undef, n_branches)
    connections = zeros(Int32, nv)
    idx, k      = 1, 1
    for c in 1:nc
        deg = Int(h_cols[idx]); idx += 1
        for _ in 1:deg
            v = Int(h_cols[idx]) + 1; idx += 1
            transpose[k]  = var_offsets[v] + connections[v]
            chk_to_var[k] = Int32(v)
            connections[v] += Int32(1)
            k += 1
        end
    end

    info_bits_pos = [Int32(info_bits_raw[i]) + Int32(1) for i in 1:Int(K)]
    sched = (sched_sym === :BP_HORIZONTAL_LAYERED || sched_sym === :BP_HL) ? ScheduleLayered() : ScheduleFlooding()
    rule  = (rule_sym === :OMS) ? RuleOMS{T}(T(offset_val)) : RuleSPA()
    return LDPC_Graph{typeof(sched), typeof(rule), T}(
        sched, rule, T(sat_val_in), n_ite, enable_syndrome, syndrome_depth,
        var_offsets, chk_offsets, transpose, chk_to_var, info_bits_pos
    )
end

function LDPC_Graph(
    n_ite::Int32, enable_syndrome::Bool, syndrome_depth::Int32,
    n_var_nodes::Int32, n_chk_nodes::Int32,
    h_cols_addr::UInt, h_cols_len::Int32, info_bits_addr::UInt, K::Int32
)
    return LDPC_Graph(:BP_FLOODING, :SPA, Float32, 0.0f0, 0.0f0,
        n_ite, enable_syndrome, syndrome_depth, n_var_nodes, n_chk_nodes,
        h_cols_addr, h_cols_len, info_bits_addr, K)
end

mutable struct LDPC_Workspace{M<:AbstractExecMode, T<:Union{Float32, Int16}}
    mode::M
    msg_chk_to_var::Matrix{T}
    msg_var_to_chk::Matrix{T}
    post::Matrix{T}
    Y_N::Matrix{T}
    values::Matrix{Float32}
    contrib::Matrix{T}
end

function LDPC_Workspace(
    mode_sym::Symbol, ::Type{T},
    n_var_nodes::Int32, n_branches::Int32, max_chk_degree::Int32,
    n_frames_per_wave::Int32 = Int32(1)
) where {T<:Union{Float32, Int16}}
    nv, nb, mc = Int(n_var_nodes), Int(n_branches), Int(max_chk_degree)
    is_inter   = (mode_sym === :INTER || mode_sym === :SIMD_INTER)
    W          = is_inter ? max(1, Int(n_frames_per_wave)) : 1
    mode       = is_inter ? SimdInterMode(W) : ScalarMode()
    return LDPC_Workspace{typeof(mode), T}(
        mode,
        zeros(T, W, nb), zeros(T, W, nb), zeros(T, W, nv), zeros(T, W, nv),
        zeros(Float32, W, mc), zeros(T, W, mc)
    )
end

LDPC_Workspace(nv::Int32, nb::Int32, mc::Int32) = LDPC_Workspace(:SCALAR, Float32, nv, nb, mc)

# ====================================================================================================
# UNIFIED BP KERNELS (parameterized by `Val(W)`: `W = 1` for ScalarMode, `W = n_frames` for SimdInterMode)
# ====================================================================================================

@inline _load_vec(A::Matrix{T}, col::Integer, ::Val{W}) where {T, W} =
    ntuple(w -> @inbounds(A[w, col]), Val(W))

@inline function _store_vec!(A::Matrix{T}, col::Integer, v::NTuple{W, T}) where {T, W}
    @inbounds for w in 1:W
        A[w, col] = v[w]
    end
end

@inline function _update_var_nodes!(
    var_offsets::Vector{Int32}, Y_N::Matrix{T},
    c2v::Matrix{T}, v2c::Matrix{T}, post::Matrix{T}, vw::Val{W}, write_v2c::Bool
) where {T<:Union{Float32, Int16}, W}
    @inbounds for v in 1:(length(var_offsets) - 1)
        e_start, e_stop = Int(var_offsets[v]), Int(var_offsets[v + 1]) - 1
        acc = _load_vec(Y_N, v, vw)
        for e in e_start:e_stop
            acc = _add_sat.(acc, _load_vec(c2v, e, vw))
        end
        if write_v2c
            for e in e_start:e_stop
                _store_vec!(v2c, e, _sub_sat.(acc, _load_vec(c2v, e, vw)))
            end
        else
            _store_vec!(post, v, acc)
        end
    end
end

@inline function _check_syndrome_soft(
    chk_offsets::Vector{Int32}, chk_to_var::Vector{Int32}, post::Matrix{T}, vw::Val{W}
)::Bool where {T<:Union{Float32, Int16}, W}
    z = zero(T)
    @inbounds for c in 1:(length(chk_offsets) - 1)
        synd = ntuple(_ -> false, vw)
        for k in Int(chk_offsets[c]):(Int(chk_offsets[c + 1]) - 1)
            synd = synd .⊻ (_load_vec(post, chk_to_var[k], vw) .< z)
        end
        any(synd) && return false
    end
    return true
end

# 1. Flooding + SPA
@inline function _step_bp!(
    ::ScheduleFlooding, ::RuleSPA, ::Float32,
    graph::LDPC_Graph, work::LDPC_Workspace{M, Float32}, vw::Val{W}
) where {M, W}
    max_tanh = 1.0f0 - eps(Float32)
    v2c, c2v, vals = work.msg_var_to_chk, work.msg_chk_to_var, work.values
    @inbounds for c in 1:(length(graph.chk_offsets) - 1)
        k_start, k_stop = Int(graph.chk_offsets[c]), Int(graph.chk_offsets[c + 1]) - 1
        sng, prd, nz, idx = ntuple(_ -> false, vw), ntuple(_ -> 1.0f0, vw), ntuple(_ -> Int32(0), vw), 1
        for k in k_start:k_stop
            v_val = _load_vec(v2c, graph.transpose[k], vw)
            res   = tanh.(abs.(v_val) .* 0.5f0)
            isz   = res .== 0.0f0
            sng   = ifelse.(isz, sng, sng .⊻ signbit.(v_val))
            prd   = ifelse.(isz, prd, prd .* res)
            nz    = ifelse.(isz, nz .+ Int32(1), nz)
            _store_vec!(vals, idx, res); idx += 1
        end
        idx = 1
        for k in k_start:k_stop
            t_k   = graph.transpose[k]
            v_val = _load_vec(v2c, t_k, vw)
            val_v = _load_vec(vals, idx, vw); idx += 1
            isz   = val_v .== 0.0f0
            rt    = min.(ifelse.(isz, prd, prd ./ val_v), max_tanh)
            mag   = 2.0f0 .* atanh.(rt)
            s_out = sng .⊻ ifelse.(isz, false, signbit.(v_val))
            keep  = (nz .== Int32(0)) .| ((nz .== Int32(1)) .& isz)
            _store_vec!(c2v, t_k, ifelse.(keep, ifelse.(s_out, .-mag, mag), 0.0f0))
        end
    end
end

# 2. Flooding + OMS
@inline function _step_bp!(
    ::ScheduleFlooding, rule::RuleOMS{T}, sat_val::T,
    graph::LDPC_Graph, work::LDPC_Workspace{M, T}, vw::Val{W}
) where {M, T<:Union{Float32, Int16}, W}
    z, m, off = zero(T), _max_val(T), rule.offset
    v2c, c2v  = work.msg_var_to_chk, work.msg_chk_to_var
    @inbounds for c in 1:(length(graph.chk_offsets) - 1)
        k_start, k_stop = Int(graph.chk_offsets[c]), Int(graph.chk_offsets[c + 1]) - 1
        sng, min1, min2 = ntuple(_ -> false, vw), ntuple(_ -> m, vw), ntuple(_ -> m, vw)
        for k in k_start:k_stop
            v_val = _load_vec(v2c, graph.transpose[k], vw)
            v_abs = abs.(v_val)
            sng   = sng .⊻ (v_val .< z)
            min2  = min.(min2, max.(v_abs, min1))
            min1  = min.(min1, v_abs)
        end
        cst1 = max.(z, _sub_sat.(min2, off))
        cst2 = max.(z, _sub_sat.(min1, off))
        for k in k_start:k_stop
            t_k   = graph.transpose[k]
            v_val = _load_vec(v2c, t_k, vw)
            mag   = ifelse.(abs.(v_val) .== min1, cst1, cst2)
            s_out = sng .⊻ (v_val .< z)
            _store_vec!(c2v, t_k, _saturate.(ifelse.(s_out, .-mag, mag), sat_val))
        end
    end
end

# 3. Horizontal Layered + SPA
@inline function _step_bp!(
    ::ScheduleLayered, ::RuleSPA, ::Float32,
    graph::LDPC_Graph, work::LDPC_Workspace{M, Float32}, vw::Val{W}
) where {M, W}
    max_tanh = 1.0f0 - eps(Float32)
    vn, msg, vals, contrib = work.post, work.msg_chk_to_var, work.values, work.contrib
    @inbounds for c in 1:(length(graph.chk_offsets) - 1)
        k_start, k_stop = Int(graph.chk_offsets[c]), Int(graph.chk_offsets[c + 1]) - 1
        sng, prd, nz, idx = ntuple(_ -> false, vw), ntuple(_ -> 1.0f0, vw), ntuple(_ -> Int32(0), vw), 1
        for k in k_start:k_stop
            u     = graph.chk_to_var[k]
            v_val = _load_vec(vn, u, vw) .- _load_vec(msg, k, vw)
            _store_vec!(contrib, idx, v_val)
            res = tanh.(abs.(v_val) .* 0.5f0)
            isz = res .== 0.0f0
            sng = ifelse.(isz, sng, sng .⊻ signbit.(v_val))
            prd = ifelse.(isz, prd, prd .* res)
            nz  = ifelse.(isz, nz .+ Int32(1), nz)
            _store_vec!(vals, idx, res); idx += 1
        end
        idx = 1
        for k in k_start:k_stop
            u       = graph.chk_to_var[k]
            v_val   = _load_vec(contrib, idx, vw)
            val_v   = _load_vec(vals, idx, vw); idx += 1
            isz     = val_v .== 0.0f0
            rt      = min.(ifelse.(isz, prd, prd ./ val_v), max_tanh)
            mag     = 2.0f0 .* atanh.(rt)
            s_out   = sng .⊻ ifelse.(isz, false, signbit.(v_val))
            keep    = (nz .== Int32(0)) .| ((nz .== Int32(1)) .& isz)
            new_msg = ifelse.(keep, ifelse.(s_out, .-mag, mag), 0.0f0)
            _store_vec!(msg, k, new_msg)
            _store_vec!(vn,  u, v_val .+ new_msg)
        end
    end
end

# 4. Horizontal Layered + OMS
@inline function _step_bp!(
    ::ScheduleLayered, rule::RuleOMS{T}, sat_val::T,
    graph::LDPC_Graph, work::LDPC_Workspace{M, T}, vw::Val{W}
) where {M, T<:Union{Float32, Int16}, W}
    z, m, off = zero(T), _max_val(T), rule.offset
    vn, msg, contrib = work.post, work.msg_chk_to_var, work.contrib
    @inbounds for c in 1:(length(graph.chk_offsets) - 1)
        k_start, k_stop = Int(graph.chk_offsets[c]), Int(graph.chk_offsets[c + 1]) - 1
        sng, min1, min2, idx = ntuple(_ -> false, vw), ntuple(_ -> m, vw), ntuple(_ -> m, vw), 1
        for k in k_start:k_stop
            u     = graph.chk_to_var[k]
            v_val = _sub_sat.(_load_vec(vn, u, vw), _load_vec(msg, k, vw))
            _store_vec!(contrib, idx, v_val); idx += 1
            v_abs = abs.(v_val)
            sng   = sng .⊻ (v_val .< z)
            min2  = min.(min2, max.(v_abs, min1))
            min1  = min.(min1, v_abs)
        end
        cst1 = max.(z, _sub_sat.(min2, off))
        cst2 = max.(z, _sub_sat.(min1, off))
        idx  = 1
        for k in k_start:k_stop
            u       = graph.chk_to_var[k]
            v_val   = _load_vec(contrib, idx, vw); idx += 1
            mag     = ifelse.(abs.(v_val) .== min1, cst1, cst2)
            s_out   = sng .⊻ (v_val .< z)
            new_msg = _saturate.(ifelse.(s_out, .-mag, mag), sat_val)
            _store_vec!(msg, k, new_msg)
            _store_vec!(vn,  u, _add_sat.(v_val, new_msg, sat_val))
        end
    end
end

# ====================================================================================================
# TOP-LEVEL ENTRY POINTS (`decode_siho_julia`)
# ====================================================================================================

@inline function _run_bp!(
    graph::LDPC_Graph{S, U, T}, work::LDPC_Workspace{M, T},
    sck_Y_N::AbstractVector{T}, sck_V_K::AbstractVector{B}, vw::Val{W}
)::Int32 where {S<:AbstractSchedule, U<:AbstractUpdateRule, M<:AbstractExecMode, T<:Union{Float32, Int16}, B<:Union{Int32, Int16}, W}
    K, N = length(graph.info_bits_pos), length(graph.var_offsets) - 1
    n_ite, s_depth = Int(graph.n_ite), Int(graph.syndrome_depth)
    target_in = (S === ScheduleLayered) ? work.post : work.Y_N
    @inbounds for w in 1:W
        off = (w - 1) * N
        for v in 1:N
            target_in[w, v] = sck_Y_N[off + v]
        end
    end
    fill!(work.msg_chk_to_var, zero(T))

    cur_depth, valid, ite_done = 0, true, 0
    for ite in 1:n_ite
        if S === ScheduleFlooding
            _update_var_nodes!(graph.var_offsets, work.Y_N, work.msg_chk_to_var, work.msg_var_to_chk, work.post, vw, true)
        end
        _step_bp!(graph.schedule, graph.rule, graph.sat_val, graph, work, vw)
        if graph.enable_syndrome && (S === ScheduleLayered || ite != n_ite)
            if S === ScheduleFlooding
                _update_var_nodes!(graph.var_offsets, work.Y_N, work.msg_chk_to_var, work.msg_var_to_chk, work.post, vw, false)
            end
            synd      = _check_syndrome_soft(graph.chk_offsets, graph.chk_to_var, work.post, vw)
            cur_depth = synd ? ((cur_depth + 1) % s_depth) : 0
            valid     = synd && (cur_depth == 0)
            if valid
                ite_done = ite
                break
            end
        end
        ite == n_ite && (ite_done = n_ite)
    end
    if S === ScheduleFlooding && ite_done == n_ite
        _update_var_nodes!(graph.var_offsets, work.Y_N, work.msg_chk_to_var, work.msg_var_to_chk, work.post, vw, false)
    end
    @inbounds for w in 1:W
        off = (w - 1) * K
        for i in 1:K
            sck_V_K[off + i] = ifelse(work.post[w, graph.info_bits_pos[i]] < zero(T), one(B), zero(B))
        end
    end
    return ifelse(valid, Int32(0), Int32(1))
end

function decode_siho_julia(
    graph::LDPC_Graph{S, U, T}, work::LDPC_Workspace{ScalarMode, T},
    sck_Y_N::AbstractVector{T}, sck_V_K::AbstractVector{B},
    ::UInt32 = UInt32(0), ::UInt32 = UInt32(1)
)::Int32 where {S<:AbstractSchedule, U<:AbstractUpdateRule, T<:Union{Float32, Int16}, B<:Union{Int32, Int16}}
    return _run_bp!(graph, work, sck_Y_N, sck_V_K, Val(1))
end

function decode_siho_julia(
    graph::LDPC_Graph{S, U, T}, work::LDPC_Workspace{SimdInterMode{W}, T},
    sck_Y_N::AbstractVector{T}, sck_V_K::AbstractVector{B},
    ::UInt32 = UInt32(0), n_frames_per_wave::UInt32 = UInt32(W)
)::Int32 where {S<:AbstractSchedule, U<:AbstractUpdateRule, W, T<:Union{Float32, Int16}, B<:Union{Int32, Int16}}
    Int(n_frames_per_wave) == W && return _run_bp!(graph, work, sck_Y_N, sck_V_K, Val(W))
    return _run_bp!(graph, work, sck_Y_N, sck_V_K, Val(Int(n_frames_per_wave)))
end

# Ahead-of-execution compilation for ScalarMode and SimdInterMode kernels
for S in (ScheduleFlooding, ScheduleLayered)
    for U in (RuleSPA, RuleOMS{Float32})
        precompile(decode_siho_julia, (LDPC_Graph{S, U, Float32}, LDPC_Workspace{ScalarMode, Float32}, Vector{Float32}, Vector{Int32}, UInt32, UInt32))
        for W in (4, 8)
            precompile(decode_siho_julia, (LDPC_Graph{S, U, Float32}, LDPC_Workspace{SimdInterMode{W}, Float32}, Vector{Float32}, Vector{Int32}, UInt32, UInt32))
        end
    end
    precompile(decode_siho_julia, (LDPC_Graph{S, RuleOMS{Int16}, Int16}, LDPC_Workspace{ScalarMode, Int16}, Vector{Int16}, Vector{Int16}, UInt32, UInt32))
    for W in (8, 16)
        precompile(decode_siho_julia, (LDPC_Graph{S, RuleOMS{Int16}, Int16}, LDPC_Workspace{SimdInterMode{W}, Int16}, Vector{Int16}, Vector{Int16}, UInt32, UInt32))
    end
end