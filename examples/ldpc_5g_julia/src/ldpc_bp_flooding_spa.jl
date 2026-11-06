# Generic Modular LDPC Belief Propagation Decoder in Julia for StreamPU (`Stateful_Julia`)
#
# Mirrors AFF3CT's template-based C++ architecture by decoupling four orthogonal axes via
# Julia's parametric types and multiple dispatch (zero runtime branching overhead):
#   1. Scheduling (`S <: AbstractSchedule`):
#        - `ScheduleFlooding` (`BP_FLOODING` / BP-F)
#        - `ScheduleLayered`  (`BP_HORIZONTAL_LAYERED` / BP-HL)
#   2. Check-Node Update Rule (`U <: AbstractUpdateRule`):
#        - `RuleSPA`          (Sum-Product Algorithm, `Float32`)
#        - `RuleOMS{T}`       (Offset Min-Sum, `Float32` or `Int16`)
#   3. Representation (`T <: Union{Float32, Int16}`):
#        - `Float32` (`f32` floating-point LLRs)
#        - `Int16`   (`i16` fixed-point quantized LLRs with symmetric saturation `[-sat_val, +sat_val]`)
#   4. Vectorization / Execution Mode (`M <: AbstractExecMode`):
#        - `ScalarMode`       (intra-frame, 1 frame per wave)
#        - `SimdInterMode`    (inter-frame SIMD across `W = n_frames_per_wave` frames simultaneously)

abstract type AbstractSchedule end
struct ScheduleFlooding <: AbstractSchedule end
struct ScheduleLayered  <: AbstractSchedule end

abstract type AbstractUpdateRule end
struct RuleSPA <: AbstractUpdateRule end
struct RuleOMS{T<:Union{Float32, Int16}} <: AbstractUpdateRule
    offset::T
end

abstract type AbstractExecMode end
struct ScalarMode    <: AbstractExecMode end
struct SimdInterMode <: AbstractExecMode end

@inline _max_val(::Type{Float32}) = floatmax(Float32)
@inline _max_val(::Type{Int16})   = typemax(Int16)

@inline _saturate(x::Float32, ::Float32)::Float32 = x
@inline function _saturate(x::Int16, sat_val::Int16)::Int16
    return clamp(x, -sat_val, sat_val)
end
@inline function _saturate(x::Int32, sat_val::Int16)::Int16
    return Int16(clamp(x, Int32(-sat_val), Int32(sat_val)))
end

@inline _add_sat(a::Float32, b::Float32, ::Float32)::Float32 = a + b
@inline _add_sat(a::Int16, b::Int16, ::Int16)::Int16 = Int16(clamp(Int32(a) + Int32(b), Int32(typemin(Int16)), Int32(typemax(Int16))))

@inline _sub_sat(a::Float32, b::Float32)::Float32 = a - b
@inline _sub_sat(a::Int16, b::Int16)::Int16 = Int16(clamp(Int32(a) - Int32(b), Int32(typemin(Int16)), Int32(typemax(Int16))))

# Immutable read-only Tanner graph representation of the LDPC parity-check matrix H
# and decoder configuration parameters. Shared across all worker threads (`shared = true`).
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

function _build_tanner_graph_arrays(
    n_var_nodes::Int32,
    n_chk_nodes::Int32,
    h_cols_addr::UInt,    h_cols_len::Int32,
    info_bits_addr::UInt, K::Int32
)
    h_cols        = unsafe_wrap(Array, Ptr{UInt32}(h_cols_addr),    Int(h_cols_len))
    info_bits_raw = unsafe_wrap(Array, Ptr{UInt32}(info_bits_addr), Int(K))

    # 1. Count variable node degrees and build chk_offsets (1-based CSR/CSC offsets)
    var_degrees = zeros(Int32, Int(n_var_nodes))
    chk_offsets = Vector{Int32}(undef, Int(n_chk_nodes) + 1)
    chk_offsets[1] = Int32(1)

    idx = 1
    for c in 1:Int(n_chk_nodes)
        deg = Int32(h_cols[idx])
        idx += 1
        chk_offsets[c + 1] = chk_offsets[c] + deg
        for _ in 1:deg
            v = Int(h_cols[idx]) + 1
            var_degrees[v] += Int32(1)
            idx += 1
        end
    end

    var_offsets = Vector{Int32}(undef, Int(n_var_nodes) + 1)
    var_offsets[1] = Int32(1)
    for v in 1:Int(n_var_nodes)
        var_offsets[v + 1] = var_offsets[v] + var_degrees[v]
    end

    # 2. Build transpose and chk_to_var (1-based indices)
    n_branches  = Int(chk_offsets[end] - 1)
    transpose   = Vector{Int32}(undef, n_branches)
    chk_to_var  = Vector{Int32}(undef, n_branches)
    connections = zeros(Int32, Int(n_var_nodes))

    idx = 1
    k   = 1
    for c in 1:Int(n_chk_nodes)
        deg = Int(h_cols[idx])
        idx += 1
        for _ in 1:deg
            v = Int(h_cols[idx]) + 1
            idx += 1
            branch_id = var_offsets[v] + connections[v]
            connections[v] += Int32(1)
            transpose[k]  = branch_id
            chk_to_var[k] = Int32(v)
            k += 1
        end
    end

    # 3. Convert 0-based C++ info_bits_pos to 1-based Julia indices
    info_bits_pos = Vector{Int32}(undef, Int(K))
    for i in 1:Int(K)
        info_bits_pos[i] = Int32(info_bits_raw[i]) + Int32(1)
    end

    return var_offsets, chk_offsets, transpose, chk_to_var, info_bits_pos
end

# Generic constructor parameterized by schedule Symbol (`:BP_FLOODING` or `:BP_HORIZONTAL_LAYERED`),
# update rule Symbol (`:SPA` or `:OMS`), data type `T` (`Float32` or `Int16`), OMS `offset`, and `sat_val`.
function LDPC_Graph(
    sched_sym::Symbol,
    rule_sym::Symbol,
    ::Type{T},
    offset_val::Real,
    sat_val_in::Real,
    n_ite::Int32,
    enable_syndrome::Bool,
    syndrome_depth::Int32,
    n_var_nodes::Int32,
    n_chk_nodes::Int32,
    h_cols_addr::UInt,    h_cols_len::Int32,
    info_bits_addr::UInt, K::Int32
) where {T<:Union{Float32, Int16}}
    var_offsets, chk_offsets, transpose, chk_to_var, info_bits_pos = _build_tanner_graph_arrays(
        n_var_nodes, n_chk_nodes, h_cols_addr, h_cols_len, info_bits_addr, K
    )
    sched = (sched_sym === :BP_HORIZONTAL_LAYERED || sched_sym === :BP_HL) ? ScheduleLayered() : ScheduleFlooding()
    rule  = (rule_sym === :OMS) ? RuleOMS{T}(T(offset_val)) : RuleSPA()
    sat_v = T(sat_val_in)
    return LDPC_Graph{typeof(sched), typeof(rule), T}(
        sched,
        rule,
        sat_v,
        n_ite,
        enable_syndrome,
        syndrome_depth,
        var_offsets,
        chk_offsets,
        transpose,
        chk_to_var,
        info_bits_pos
    )
end

# Backward-compatible constructor defaulting to `(ScheduleFlooding, RuleSPA, Float32)`.
function LDPC_Graph(
    n_ite::Int32,
    enable_syndrome::Bool,
    syndrome_depth::Int32,
    n_var_nodes::Int32,
    n_chk_nodes::Int32,
    h_cols_addr::UInt,    h_cols_len::Int32,
    info_bits_addr::UInt, K::Int32
)
    return LDPC_Graph(
        :BP_FLOODING,
        :SPA,
        Float32,
        0.0f0,
        0.0f0,
        n_ite,
        enable_syndrome,
        syndrome_depth,
        n_var_nodes,
        n_chk_nodes,
        h_cols_addr, h_cols_len,
        info_bits_addr, K
    )
end

# Per-thread mutable working buffers (`shared = false`) parameterized by execution mode `M`
# (`ScalarMode` or `SimdInterMode`) and LLR representation `T` (`Float32` or `Int16`).
mutable struct LDPC_Workspace{M<:AbstractExecMode, T<:Union{Float32, Int16}}
    mode::M
    # Scalar mode 1D buffers
    msg_chk_to_var::Vector{T}
    msg_var_to_chk::Vector{T}
    post::Vector{T}
    values::Vector{Float32}
    contrib::Vector{T}
    # Inter-frame SIMD 2D buffers of shape (W, ...) where W = n_frames_per_wave
    msg_chk_to_var_inter::Matrix{T}
    msg_var_to_chk_inter::Matrix{T}
    post_inter::Matrix{T}
    Y_N_inter::Matrix{T}
    values_inter::Matrix{Float32}
    contrib_inter::Matrix{T}
    # Per-wave SIMD reduction scratch vectors of length W
    sign_w::Vector{UInt8}
    min1_w::Vector{T}
    min2_w::Vector{T}
    cst1_w::Vector{T}
    cst2_w::Vector{T}
    prod_w::Vector{Float32}
    nzeros_w::Vector{Int32}
    synd_w::Vector{UInt8}
end

function LDPC_Workspace(
    mode_sym::Symbol,
    ::Type{T},
    n_var_nodes::Int32,
    n_branches::Int32,
    max_chk_degree::Int32,
    n_frames_per_wave::Int32 = Int32(1)
) where {T<:Union{Float32, Int16}}
    nv = Int(n_var_nodes)
    nb = Int(n_branches)
    mc = Int(max_chk_degree)
    W  = max(1, Int(n_frames_per_wave))

    if mode_sym === :INTER || mode_sym === :SIMD_INTER
        return LDPC_Workspace{SimdInterMode, T}(
            SimdInterMode(),
            Vector{T}(undef, 0),
            Vector{T}(undef, 0),
            Vector{T}(undef, 0),
            Vector{Float32}(undef, 0),
            Vector{T}(undef, 0),
            zeros(T, W, nb),
            zeros(T, W, nb),
            zeros(T, W, nv),
            zeros(T, W, nv),
            zeros(Float32, W, mc),
            zeros(T, W, mc),
            zeros(UInt8, W),
            zeros(T, W),
            zeros(T, W),
            zeros(T, W),
            zeros(T, W),
            zeros(Float32, W),
            zeros(Int32, W),
            zeros(UInt8, W)
        )
    else
        return LDPC_Workspace{ScalarMode, T}(
            ScalarMode(),
            zeros(T, nb),
            zeros(T, nb),
            zeros(T, nv),
            zeros(Float32, mc),
            zeros(T, mc),
            Matrix{T}(undef, 0, 0),
            Matrix{T}(undef, 0, 0),
            Matrix{T}(undef, 0, 0),
            Matrix{T}(undef, 0, 0),
            Matrix{Float32}(undef, 0, 0),
            Matrix{T}(undef, 0, 0),
            Vector{UInt8}(undef, 0),
            Vector{T}(undef, 0),
            Vector{T}(undef, 0),
            Vector{T}(undef, 0),
            Vector{T}(undef, 0),
            Vector{Float32}(undef, 0),
            Vector{Int32}(undef, 0),
            Vector{UInt8}(undef, 0)
        )
    end
end

# Backward-compatible constructor defaulting to `(ScalarMode, Float32)`.
function LDPC_Workspace(n_var_nodes::Int32, n_branches::Int32, max_chk_degree::Int32)
    return LDPC_Workspace(:SCALAR, Float32, n_var_nodes, n_branches, max_chk_degree, Int32(1))
end

# Ensures `SimdInterMode` buffers match the runtime `W = n_frames_per_wave` if `set_n_frames` was called after constructor.
@inline function _ensure_inter_capacity!(
    work::LDPC_Workspace{SimdInterMode, T},
    W::Int,
    n_var_nodes::Int,
    n_branches::Int,
    max_chk_degree::Int
) where {T<:Union{Float32, Int16}}
    if size(work.post_inter, 1) != W
        work.msg_chk_to_var_inter = zeros(T, W, n_branches)
        work.msg_var_to_chk_inter = zeros(T, W, n_branches)
        work.post_inter           = zeros(T, W, n_var_nodes)
        work.Y_N_inter            = zeros(T, W, n_var_nodes)
        work.values_inter         = zeros(Float32, W, max_chk_degree)
        work.contrib_inter        = zeros(T, W, max_chk_degree)
        work.sign_w               = zeros(UInt8, W)
        work.min1_w               = zeros(T, W)
        work.min2_w               = zeros(T, W)
        work.cst1_w               = zeros(T, W)
        work.cst2_w               = zeros(T, W)
        work.prod_w               = zeros(Float32, W)
        work.nzeros_w             = zeros(Int32, W)
        work.synd_w               = zeros(UInt8, W)
    end
    return nothing
end

# ====================================================================================================
# 1. SCALAR KERNELS (`ScalarMode`)
# ====================================================================================================

# Variable-Node (VN) update step for Flooding schedule (Scalar, `Float32` or `Int16`).
@inline function _initialize_var_to_chk!(
    var_offsets::Vector{Int32},
    Y_N::AbstractVector{T},
    msg_chk_to_var::Vector{T},
    msg_var_to_chk::Vector{T}
) where {T<:Union{Float32, Int16}}
    n_var_nodes = length(var_offsets) - 1
    @inbounds for v in 1:n_var_nodes
        e_start = Int(var_offsets[v])
        e_stop  = Int(var_offsets[v + 1]) - 1
        if T === Float32
            sum_msg = 0.0f0
            for e in e_start:e_stop
                sum_msg += msg_chk_to_var[e]
            end
            tmp = Y_N[v] + sum_msg
            for e in e_start:e_stop
                msg_var_to_chk[e] = tmp - msg_chk_to_var[e]
            end
        else
            sum_msg = Int32(0)
            for e in e_start:e_stop
                sum_msg += Int32(msg_chk_to_var[e])
            end
            tmp = Int32(Y_N[v]) + sum_msg
            for e in e_start:e_stop
                msg_var_to_chk[e] = _saturate(tmp - Int32(msg_chk_to_var[e]), typemax(Int16))
            end
        end
    end
    return nothing
end

# Flooding + SPA (Scalar, `Float32`)
@inline function _decode_single_ite_flooding!(
    ::RuleSPA,
    ::Float32,
    chk_offsets::Vector{Int32},
    transpose::Vector{Int32},
    values::Vector{Float32},
    msg_var_to_chk::Vector{Float32},
    msg_chk_to_var::Vector{Float32}
)
    n_chk_nodes = length(chk_offsets) - 1
    max_tanh    = 1.0f0 - eps(Float32)

    @inbounds for c in 1:n_chk_nodes
        k_start = Int(chk_offsets[c])
        k_stop  = Int(chk_offsets[c + 1]) - 1

        sign    = false
        product = 1.0f0
        n_zeros = 0
        idx     = 1

        for k in k_start:k_stop
            var_val = msg_var_to_chk[transpose[k]]
            var_abs = abs(var_val)
            res     = tanh(var_abs * 0.5f0)
            is_zero = (res == 0.0f0)

            sign    = ifelse(is_zero, sign, sign ⊻ signbit(var_val))
            product = ifelse(is_zero, product, product * res)
            n_zeros = ifelse(is_zero, n_zeros + 1, n_zeros)
            values[idx] = res
            idx += 1
        end

        if n_zeros == 0
            idx = 1
            for k in k_start:k_stop
                t_k     = transpose[k]
                var_val = msg_var_to_chk[t_k]
                val_v   = values[idx]
                res_tmp = product / val_v
                res_tmp = ifelse(res_tmp < 1.0f0, res_tmp, max_tanh)
                res_abs = 2.0f0 * atanh(res_tmp)
                res_sng = sign ⊻ signbit(var_val)
                msg_chk_to_var[t_k] = ifelse(res_sng, -res_abs, res_abs)
                idx += 1
            end
        elseif n_zeros == 1
            res_tmp = ifelse(product < 1.0f0, product, max_tanh)
            res_abs = 2.0f0 * atanh(res_tmp)
            res_val = ifelse(sign, -res_abs, res_abs)
            idx = 1
            for k in k_start:k_stop
                t_k   = transpose[k]
                val_v = values[idx]
                msg_chk_to_var[t_k] = ifelse(val_v == 0.0f0, res_val, 0.0f0)
                idx += 1
            end
        else
            for k in k_start:k_stop
                msg_chk_to_var[transpose[k]] = 0.0f0
            end
        end
    end
    return nothing
end

# Flooding + OMS (Scalar, `Float32` or `Int16`)
@inline function _decode_single_ite_flooding!(
    rule::RuleOMS{T},
    sat_val::T,
    chk_offsets::Vector{Int32},
    transpose::Vector{Int32},
    ::Vector{Float32},
    msg_var_to_chk::Vector{T},
    msg_chk_to_var::Vector{T}
) where {T<:Union{Float32, Int16}}
    n_chk_nodes = length(chk_offsets) - 1
    zero_T      = zero(T)
    max_T       = _max_val(T)
    off         = rule.offset

    @inbounds for c in 1:n_chk_nodes
        k_start = Int(chk_offsets[c])
        k_stop  = Int(chk_offsets[c + 1]) - 1

        sign = false
        min1 = max_T
        min2 = max_T

        for k in k_start:k_stop
            var_val = msg_var_to_chk[transpose[k]]
            var_abs = abs(var_val)
            sign  ⊻= (var_val < zero_T)
            min2    = min(min2, max(var_abs, min1))
            min1    = min(min1, var_abs)
        end

        cst1 = max(zero_T, T(min2 - off))
        cst2 = max(zero_T, T(min1 - off))

        for k in k_start:k_stop
            t_k     = transpose[k]
            var_val = msg_var_to_chk[t_k]
            var_abs = abs(var_val)
            res_abs = ifelse(var_abs == min1, cst1, cst2)
            res_sng = sign ⊻ (var_val < zero_T)
            res     = ifelse(res_sng, -res_abs, res_abs)
            msg_chk_to_var[t_k] = _saturate(res, sat_val)
        end
    end
    return nothing
end

# Horizontal Layered + SPA (Scalar, `Float32`)
@inline function _decode_single_ite_layered!(
    ::RuleSPA,
    ::Float32,
    chk_offsets::Vector{Int32},
    chk_to_var::Vector{Int32},
    values::Vector{Float32},
    contrib::Vector{Float32},
    var_nodes::Vector{Float32},
    messages::Vector{Float32}
)
    n_chk_nodes = length(chk_offsets) - 1
    max_tanh    = 1.0f0 - eps(Float32)

    @inbounds for c in 1:n_chk_nodes
        k_start = Int(chk_offsets[c])
        k_stop  = Int(chk_offsets[c + 1]) - 1

        sign    = false
        product = 1.0f0
        n_zeros = 0
        idx     = 1

        for k in k_start:k_stop
            u       = chk_to_var[k]
            var_val = var_nodes[u] - messages[k]
            contrib[idx] = var_val
            var_abs = abs(var_val)
            res     = tanh(var_abs * 0.5f0)
            is_zero = (res == 0.0f0)

            sign    = ifelse(is_zero, sign, sign ⊻ signbit(var_val))
            product = ifelse(is_zero, product, product * res)
            n_zeros = ifelse(is_zero, n_zeros + 1, n_zeros)
            values[idx] = res
            idx += 1
        end

        if n_zeros == 0
            idx = 1
            for k in k_start:k_stop
                u       = chk_to_var[k]
                var_val = contrib[idx]
                val_v   = values[idx]
                res_tmp = product / val_v
                res_tmp = ifelse(res_tmp < 1.0f0, res_tmp, max_tanh)
                res_abs = 2.0f0 * atanh(res_tmp)
                res_sng = sign ⊻ signbit(var_val)
                new_msg = ifelse(res_sng, -res_abs, res_abs)
                messages[k]  = new_msg
                var_nodes[u] = var_val + new_msg
                idx += 1
            end
        elseif n_zeros == 1
            res_tmp = ifelse(product < 1.0f0, product, max_tanh)
            res_abs = 2.0f0 * atanh(res_tmp)
            res_val = ifelse(sign, -res_abs, res_abs)
            idx = 1
            for k in k_start:k_stop
                u       = chk_to_var[k]
                var_val = contrib[idx]
                val_v   = values[idx]
                new_msg = ifelse(val_v == 0.0f0, res_val, 0.0f0)
                messages[k]  = new_msg
                var_nodes[u] = var_val + new_msg
                idx += 1
            end
        else
            idx = 1
            for k in k_start:k_stop
                u = chk_to_var[k]
                messages[k]  = 0.0f0
                var_nodes[u] = contrib[idx]
                idx += 1
            end
        end
    end
    return nothing
end

# Horizontal Layered + OMS (Scalar, `Float32` or `Int16`)
@inline function _decode_single_ite_layered!(
    rule::RuleOMS{T},
    sat_val::T,
    chk_offsets::Vector{Int32},
    chk_to_var::Vector{Int32},
    ::Vector{Float32},
    contrib::Vector{T},
    var_nodes::Vector{T},
    messages::Vector{T}
) where {T<:Union{Float32, Int16}}
    n_chk_nodes = length(chk_offsets) - 1
    zero_T      = zero(T)
    max_T       = _max_val(T)
    off         = rule.offset

    @inbounds for c in 1:n_chk_nodes
        k_start = Int(chk_offsets[c])
        k_stop  = Int(chk_offsets[c + 1]) - 1

        sign = false
        min1 = max_T
        min2 = max_T
        idx  = 1

        for k in k_start:k_stop
            u       = chk_to_var[k]
            var_val = _sub_sat(var_nodes[u], messages[k])
            contrib[idx] = var_val
            var_abs = abs(var_val)
            sign  ⊻= (var_val < zero_T)
            min2    = min(min2, max(var_abs, min1))
            min1    = min(min1, var_abs)
            idx    += 1
        end

        cst1 = max(zero_T, T(min2 - off))
        cst2 = max(zero_T, T(min1 - off))

        idx = 1
        for k in k_start:k_stop
            u       = chk_to_var[k]
            var_val = contrib[idx]
            var_abs = abs(var_val)
            res_abs = ifelse(var_abs == min1, cst1, cst2)
            res_sng = sign ⊻ (var_val < zero_T)
            new_msg = _saturate(ifelse(res_sng, -res_abs, res_abs), sat_val)
            messages[k]  = new_msg
            var_nodes[u] = _add_sat(var_val, new_msg, sat_val)
            idx += 1
        end
    end
    return nothing
end

@inline function _compute_post!(
    var_offsets::Vector{Int32},
    Y_N::AbstractVector{T},
    msg_chk_to_var::Vector{T},
    post::Vector{T}
) where {T<:Union{Float32, Int16}}
    n_var_nodes = length(var_offsets) - 1
    @inbounds for v in 1:n_var_nodes
        e_start = Int(var_offsets[v])
        e_stop  = Int(var_offsets[v + 1]) - 1
        if T === Float32
            sum_msg = 0.0f0
            for e in e_start:e_stop
                sum_msg += msg_chk_to_var[e]
            end
            post[v] = Y_N[v] + sum_msg
        else
            sum_msg = Int32(0)
            for e in e_start:e_stop
                sum_msg += Int32(msg_chk_to_var[e])
            end
            post[v] = _saturate(Int32(Y_N[v]) + sum_msg, typemax(Int16))
        end
    end
    return nothing
end

@inline function _check_syndrome_soft(
    chk_offsets::Vector{Int32},
    chk_to_var::Vector{Int32},
    post::Vector{T}
)::Bool where {T<:Union{Float32, Int16}}
    n_chk_nodes = length(chk_offsets) - 1
    zero_T      = zero(T)
    @inbounds for c in 1:n_chk_nodes
        k_start = Int(chk_offsets[c])
        k_stop  = Int(chk_offsets[c + 1]) - 1
        sign    = false
        for k in k_start:k_stop
            llr = post[chk_to_var[k]]
            sign ⊻= (llr < zero_T)
        end
        if sign
            return false
        end
    end
    return true
end

# ====================================================================================================
# 2. INTER-FRAME SIMD KERNELS (`SimdInterMode`)
# ====================================================================================================

# Hardware SIMD register width in bytes (auto-detected across ARM64 NEON = 16B, x86 AVX2 = 32B, AVX-512 = 64B)
const _SIMD_REG_BYTES = if Sys.ARCH in (:x86_64, :i686)
    try
        ll = lowercase(unsafe_string(ccall(:jl_get_cpu_name, Cstring, ())))
        occursin("avx512", ll) || occursin("skylake-avx512", ll) || occursin("icelake", ll) || occursin("sapphirerapids", ll) || occursin("zen4", ll) || occursin("zen5", ll) ? 64 : 32
    catch
        32
    end
else
    16 # ARM64 NEON (128-bit)
end

_llvm_type(::Type{Float32}) = "float"
_llvm_type(::Type{Int32})   = "i32"
_llvm_type(::Type{Int16})   = "i16"

function _build_shuffle_ir(::Type{T}, W::Int, Mask::Tuple) where {T}
    llvm_t = _llvm_type(T)
    mask_str = ""
    for i in 1:length(Mask)
        if i > 1
            mask_str *= ", "
        end
        mask_str *= "i32 $(Mask[i])"
    end
    return "%res = shufflevector <$W x $llvm_t> %0, <$W x $llvm_t> %1, <$W x i32> <$mask_str>\nret <$W x $llvm_t> %res"
end

@generated function _simd_shuffle(
    a::NTuple{W, Base.VecElement{T}},
    b::NTuple{W, Base.VecElement{T}},
    ::Val{Mask}
) where {W, T, Mask}
    ir = _build_shuffle_ir(T, W, Mask)
    VecT = NTuple{W, Base.VecElement{T}}
    return quote
        $(Expr(:meta, :inline))
        Base.llvmcall($ir, $VecT, Tuple{$VecT, $VecT}, a, b)
    end
end

function _build_hard_dec_ir(::Type{T}, ::Type{B}, W::Int) where {T, B}
    llvm_t = _llvm_type(T)
    llvm_b = _llvm_type(B)
    cmp_op = (T === Float32) ? "fcmp olt" : "icmp slt"
    return "%cmp = $cmp_op <$W x $llvm_t> %0, zeroinitializer\n%res = zext <$W x i1> %cmp to <$W x $llvm_b>\nret <$W x $llvm_b> %res"
end

@generated function _hard_dec_vec(ptr::Ptr{T}, ::Type{B}, ::Val{W}) where {T, B, W}
    ir   = _build_hard_dec_ir(T, B, W)
    VecT = NTuple{W, Base.VecElement{T}}
    VecB = NTuple{W, Base.VecElement{B}}
    return quote
        $(Expr(:meta, :inline))
        v = unsafe_load(Ptr{$VecT}(ptr))
        Base.llvmcall($ir, $VecB, Tuple{$VecT}, v)
    end
end

# Generic log2(W)-stage radix-2 SIMD butterfly transpose for any power-of-2 tile W in {2, 4, 8, 16, 32}
function _build_butterfly_ast(W::Int)
    @assert ispow2(W) && W >= 2
    n_stages = trailing_zeros(W)
    stmts = Expr[]
    cur_vars = Vector{Symbol}(undef, W)
    for i in 0:(W - 1)
        s_var = Symbol("r_0_$i")
        cur_vars[i + 1] = s_var
        push!(stmts, :($s_var = vecs[$(i + 1)]))
    end

    for s in 0:(n_stages - 1)
        step = 1 << s
        next_vars = Vector{Symbol}(undef, W)
        for i in 0:(W - 1)
            next_vars[i + 1] = Symbol("r_$(s + 1)_$i")
        end
        mask_lo = Int[]
        mask_hi = Int[]
        for k in 0:step:((W ÷ 2) - 1)
            for j in 0:(step - 1)
                push!(mask_lo, k + j)
            end
            for j in 0:(step - 1)
                push!(mask_lo, W + k + j)
            end
        end
        for k in (W ÷ 2):step:(W - 1)
            for j in 0:(step - 1)
                push!(mask_hi, k + j)
            end
            for j in 0:(step - 1)
                push!(mask_hi, W + k + j)
            end
        end
        t_lo = Tuple(mask_lo)
        t_hi = Tuple(mask_hi)

        for block in 0:(2 * step):(W - 1)
            for j in 0:(step - 1)
                i0     = block + j
                i1     = block + j + step
                out_lo = block + 2 * j
                out_hi = block + 2 * j + 1
                push!(stmts, :($(next_vars[out_lo + 1]) = _simd_shuffle($(cur_vars[i0 + 1]), $(cur_vars[i1 + 1]), Val($t_lo))))
                push!(stmts, :($(next_vars[out_hi + 1]) = _simd_shuffle($(cur_vars[i0 + 1]), $(cur_vars[i1 + 1]), Val($t_hi))))
            end
        end
        cur_vars = next_vars
    end

    ret_tuple = Expr(:tuple, cur_vars...)
    return quote
        $(Expr(:meta, :inline))
        @inbounds begin
            $(stmts...)
            return $ret_tuple
        end
    end
end

@generated function _butterfly_transpose(vecs::NTuple{W, NTuple{W, Base.VecElement{T}}}) where {W, T}
    return _build_butterfly_ast(W)
end

function _build_transpose_in_tiled_ast(::Type{T}, W_tile::Int) where {T}
    VecT    = NTuple{W_tile, Base.VecElement{T}}
    elem_sz = sizeof(T)
    loads   = Expr[]
    r_syms  = Vector{Symbol}(undef, W_tile)
    c_syms  = Vector{Symbol}(undef, W_tile)
    stores  = Expr[]
    for w in 0:(W_tile - 1)
        rs = Symbol("r_$w")
        cs = Symbol("c_$w")
        r_syms[w + 1] = rs
        c_syms[w + 1] = cs
        push!(loads, :($rs = unsafe_load(Ptr{$VecT}(pY + (((w0 + $w) * N_cw + v0) * $elem_sz)))))
        push!(stores, :(unsafe_store!(Ptr{$VecT}(pS + (((v0 + $w) * W + w0) * $elem_sz)), $cs)))
    end
    in_tup  = Expr(:tuple, r_syms...)
    out_lhs = Expr(:tuple, c_syms...)
    return quote
        $(Expr(:meta, :inline))
        pY = pointer(sck_Y_N)
        pS = pointer(Y_N_inter)
        @inbounds for w0 in 0:$W_tile:(W - $W_tile)
            v0 = 0
            while v0 + $W_tile <= N_cw
                $(loads...)
                $out_lhs = _butterfly_transpose($in_tup)
                $(stores...)
                v0 += $W_tile
            end
        end
        return nothing
    end
end

@generated function _transpose_in_tiled!(
    Y_N_inter::Matrix{T},
    sck_Y_N::AbstractVector{T},
    N_cw::Int,
    W::Int,
    ::Val{W_tile}
) where {T, W_tile}
    return _build_transpose_in_tiled_ast(T, W_tile)
end

function _build_extract_tiled_ast(::Type{B}, ::Type{T}, W_tile::Int) where {B, T}
    VecB   = NTuple{W_tile, Base.VecElement{B}}
    sz_T   = sizeof(T)
    sz_B   = sizeof(B)
    loads  = Expr[]
    r_syms = Vector{Symbol}(undef, W_tile)
    c_syms = Vector{Symbol}(undef, W_tile)
    stores = Expr[]
    for j in 0:(W_tile - 1)
        vs = Symbol("v_$j")
        rs = Symbol("r_$j")
        cs = Symbol("c_$j")
        r_syms[j + 1] = rs
        c_syms[j + 1] = cs
        push!(loads, :($vs = Int(info_bits_pos[i0 + $(j + 1)]) - 1))
        push!(loads, :($rs = _hard_dec_vec(pS + (($vs * W + w0) * $sz_T), $B, Val($W_tile))))
        push!(stores, :(unsafe_store!(Ptr{$VecB}(pV + (((w0 + $j) * K + i0) * $sz_B)), $cs)))
    end
    in_tup  = Expr(:tuple, r_syms...)
    out_lhs = Expr(:tuple, c_syms...)
    return quote
        $(Expr(:meta, :inline))
        pS = pointer(post_inter)
        pV = pointer(sck_V_K)
        @inbounds for w0 in 0:$W_tile:(W - $W_tile)
            i0 = 0
            while i0 + $W_tile <= K
                $(loads...)
                $out_lhs = _butterfly_transpose($in_tup)
                $(stores...)
                i0 += $W_tile
            end
        end
        return nothing
    end
end

@generated function _extract_hard_decision_tiled!(
    sck_V_K::AbstractVector{B},
    post_inter::Matrix{T},
    info_bits_pos::Vector{Int32},
    K::Int,
    W::Int,
    ::Val{W_tile}
) where {B, T, W_tile}
    return _build_extract_tiled_ast(B, T, W_tile)
end

@inline function _transpose_in!(
    Y_N_inter::Matrix{T},
    sck_Y_N::AbstractVector{T},
    N_cw::Int,
    W::Int
) where {T<:Union{Float32, Int16}}
    # Optimal hardware SIMD lane count for T:
    # - ARM64 NEON (16B): 4 for Float32, 8 for Int16
    # - x86 AVX2   (32B): 8 for Float32, 16 for Int16
    # - x86 AVX512 (64B): 16 for Float32, 32 for Int16
    hw_lanes = _SIMD_REG_BYTES ÷ sizeof(T)
    if hw_lanes >= 32 && W % 32 == 0 && N_cw % 32 == 0
        _transpose_in_tiled!(Y_N_inter, sck_Y_N, N_cw, W, Val(32))
    elseif hw_lanes >= 16 && W % 16 == 0 && N_cw % 16 == 0
        _transpose_in_tiled!(Y_N_inter, sck_Y_N, N_cw, W, Val(16))
    elseif hw_lanes >= 8 && W % 8 == 0 && N_cw % 8 == 0
        _transpose_in_tiled!(Y_N_inter, sck_Y_N, N_cw, W, Val(8))
    elseif W % 4 == 0 && N_cw % 4 == 0
        _transpose_in_tiled!(Y_N_inter, sck_Y_N, N_cw, W, Val(4))
    else
        @inbounds for w in 1:W
            offset = (w - 1) * N_cw
            for v in 1:N_cw
                Y_N_inter[w, v] = sck_Y_N[offset + v]
            end
        end
    end
    return nothing
end

@inline function _extract_hard_decision_inter!(
    sck_V_K::AbstractVector{B},
    post_inter::Matrix{T},
    info_bits_pos::Vector{Int32},
    K::Int,
    W::Int
) where {B<:Union{Int32, Int16}, T<:Union{Float32, Int16}}
    hw_lanes = _SIMD_REG_BYTES ÷ max(sizeof(B), sizeof(T))
    if hw_lanes >= 32 && W % 32 == 0 && K % 32 == 0
        _extract_hard_decision_tiled!(sck_V_K, post_inter, info_bits_pos, K, W, Val(32))
    elseif hw_lanes >= 16 && W % 16 == 0 && K % 16 == 0
        _extract_hard_decision_tiled!(sck_V_K, post_inter, info_bits_pos, K, W, Val(16))
    elseif hw_lanes >= 8 && W % 8 == 0 && K % 8 == 0
        _extract_hard_decision_tiled!(sck_V_K, post_inter, info_bits_pos, K, W, Val(8))
    elseif W % 4 == 0 && K % 4 == 0
        _extract_hard_decision_tiled!(sck_V_K, post_inter, info_bits_pos, K, W, Val(4))
    else
        zero_T = zero(T)
        zero_B = zero(B)
        one_B  = one(B)
        @inbounds for w in 1:W
            offset = (w - 1) * K
            for v in 1:K
                k = info_bits_pos[v]
                sck_V_K[offset + v] = ifelse(post_inter[w, k] < zero_T, one_B, zero_B)
            end
        end
    end
    return nothing
end

@inline function _initialize_var_to_chk_inter!(
    var_offsets::Vector{Int32},
    Y_N_inter::Matrix{T},
    msg_chk_to_var_inter::Matrix{T},
    msg_var_to_chk_inter::Matrix{T},
    sat_val::T,
    W::Int
) where {T<:Union{Float32, Int16}}
    n_var_nodes = length(var_offsets) - 1
    zero_T      = zero(T)
    @inbounds for v in 1:n_var_nodes
        e_start = Int(var_offsets[v])
        e_stop  = Int(var_offsets[v + 1]) - 1
        for e in e_start:e_stop
            @simd ivdep for w in 1:W
                msg_var_to_chk_inter[w, e] = zero_T
            end
        end
        for e in e_start:e_stop
            @simd ivdep for w in 1:W
                # accumulate into first slot or via post_inter
            end
        end
    end
    return nothing
end

# Using `post_inter` as the accumulator for `_initialize_var_to_chk_inter!` (zero temporary allocations, pure unit-stride SIMD):
@inline function _initialize_var_to_chk_inter!(
    var_offsets::Vector{Int32},
    Y_N_inter::Matrix{T},
    msg_chk_to_var_inter::Matrix{T},
    msg_var_to_chk_inter::Matrix{T},
    post_inter::Matrix{T},
    W::Int
) where {T<:Union{Float32, Int16}}
    n_var_nodes = length(var_offsets) - 1
    @inbounds for v in 1:n_var_nodes
        e_start = Int(var_offsets[v])
        e_stop  = Int(var_offsets[v + 1]) - 1
        @simd ivdep for w in 1:W
            post_inter[w, v] = Y_N_inter[w, v]
        end
        for e in e_start:e_stop
            @simd ivdep for w in 1:W
                post_inter[w, v] = _add_sat(post_inter[w, v], msg_chk_to_var_inter[w, e], typemax(T))
            end
        end
        for e in e_start:e_stop
            @simd ivdep for w in 1:W
                msg_var_to_chk_inter[w, e] = _sub_sat(post_inter[w, v], msg_chk_to_var_inter[w, e])
            end
        end
    end
    return nothing
end

@inline function _compute_post_inter!(
    var_offsets::Vector{Int32},
    Y_N_inter::Matrix{T},
    msg_chk_to_var_inter::Matrix{T},
    post_inter::Matrix{T},
    W::Int
) where {T<:Union{Float32, Int16}}
    n_var_nodes = length(var_offsets) - 1
    @inbounds for v in 1:n_var_nodes
        e_start = Int(var_offsets[v])
        e_stop  = Int(var_offsets[v + 1]) - 1
        @simd ivdep for w in 1:W
            post_inter[w, v] = Y_N_inter[w, v]
        end
        for e in e_start:e_stop
            @simd ivdep for w in 1:W
                post_inter[w, v] = _add_sat(post_inter[w, v], msg_chk_to_var_inter[w, e], typemax(T))
            end
        end
    end
    return nothing
end

# Flooding + OMS (SimdInterMode, `Float32` or `Int16`)
@inline function _decode_single_ite_flooding_inter!(
    rule::RuleOMS{T},
    sat_val::T,
    chk_offsets::Vector{Int32},
    transpose::Vector{Int32},
    work::LDPC_Workspace{SimdInterMode, T},
    W::Int
) where {T<:Union{Float32, Int16}}
    n_chk_nodes      = length(chk_offsets) - 1
    zero_T           = zero(T)
    max_T            = _max_val(T)
    off              = rule.offset
    msg_var_to_chk   = work.msg_var_to_chk_inter
    msg_chk_to_var   = work.msg_chk_to_var_inter
    sign_w           = work.sign_w
    min1_w           = work.min1_w
    min2_w           = work.min2_w
    cst1_w           = work.cst1_w
    cst2_w           = work.cst2_w

    @inbounds for c in 1:n_chk_nodes
        k_start = Int(chk_offsets[c])
        k_stop  = Int(chk_offsets[c + 1]) - 1

        @simd ivdep for w in 1:W
            sign_w[w] = 0x00
            min1_w[w] = max_T
            min2_w[w] = max_T
        end

        for k in k_start:k_stop
            t_k = transpose[k]
            @simd ivdep for w in 1:W
                var_val   = msg_var_to_chk[w, t_k]
                var_abs   = abs(var_val)
                sng       = ifelse(var_val < zero_T, 0x01, 0x00)
                sign_w[w] ⊻= sng
                m1        = min1_w[w]
                m2        = min2_w[w]
                min2_w[w] = min(m2, max(var_abs, m1))
                min1_w[w] = min(m1, var_abs)
            end
        end

        @simd ivdep for w in 1:W
            cst1_w[w] = max(zero_T, T(min2_w[w] - off))
            cst2_w[w] = max(zero_T, T(min1_w[w] - off))
        end

        for k in k_start:k_stop
            t_k = transpose[k]
            @simd ivdep for w in 1:W
                var_val = msg_var_to_chk[w, t_k]
                var_abs = abs(var_val)
                res_abs = ifelse(var_abs == min1_w[w], cst1_w[w], cst2_w[w])
                sng     = sign_w[w] ⊻ ifelse(var_val < zero_T, 0x01, 0x00)
                res     = ifelse(sng != 0x00, -res_abs, res_abs)
                msg_chk_to_var[w, t_k] = _saturate(res, sat_val)
            end
        end
    end
    return nothing
end

# Flooding + SPA (SimdInterMode, `Float32`)
@inline function _decode_single_ite_flooding_inter!(
    ::RuleSPA,
    ::Float32,
    chk_offsets::Vector{Int32},
    transpose::Vector{Int32},
    work::LDPC_Workspace{SimdInterMode, Float32},
    W::Int
)
    n_chk_nodes    = length(chk_offsets) - 1
    max_tanh       = 1.0f0 - eps(Float32)
    msg_var_to_chk = work.msg_var_to_chk_inter
    msg_chk_to_var = work.msg_chk_to_var_inter
    values_inter   = work.values_inter
    sign_w         = work.sign_w
    prod_w         = work.prod_w
    nzeros_w       = work.nzeros_w

    @inbounds for c in 1:n_chk_nodes
        k_start = Int(chk_offsets[c])
        k_stop  = Int(chk_offsets[c + 1]) - 1

        @simd ivdep for w in 1:W
            sign_w[w]   = 0x00
            prod_w[w]   = 1.0f0
            nzeros_w[w] = Int32(0)
        end

        idx = 1
        for k in k_start:k_stop
            t_k = transpose[k]
            @simd ivdep for w in 1:W
                var_val = msg_var_to_chk[w, t_k]
                var_abs = abs(var_val)
                res     = tanh(var_abs * 0.5f0)
                is_z    = (res == 0.0f0)
                sng     = ifelse(signbit(var_val), 0x01, 0x00)
                sign_w[w]          = ifelse(is_z, sign_w[w], sign_w[w] ⊻ sng)
                prod_w[w]          = ifelse(is_z, prod_w[w], prod_w[w] * res)
                nzeros_w[w]        = ifelse(is_z, nzeros_w[w] + Int32(1), nzeros_w[w])
                values_inter[w, idx] = res
            end
            idx += 1
        end

        idx = 1
        for k in k_start:k_stop
            t_k = transpose[k]
            for w in 1:W
                nz    = nzeros_w[w]
                val_v = values_inter[w, idx]
                is_z  = (val_v == 0.0f0)
                if nz == Int32(0)
                    var_val = msg_var_to_chk[w, t_k]
                    res_tmp = prod_w[w] / val_v
                    res_tmp = ifelse(res_tmp < 1.0f0, res_tmp, max_tanh)
                    res_abs = 2.0f0 * atanh(res_tmp)
                    sng     = sign_w[w] ⊻ ifelse(signbit(var_val), 0x01, 0x00)
                    msg_chk_to_var[w, t_k] = ifelse(sng != 0x00, -res_abs, res_abs)
                elseif nz == Int32(1) && is_z
                    res_tmp = ifelse(prod_w[w] < 1.0f0, prod_w[w], max_tanh)
                    res_abs = 2.0f0 * atanh(res_tmp)
                    msg_chk_to_var[w, t_k] = ifelse(sign_w[w] != 0x00, -res_abs, res_abs)
                else
                    msg_chk_to_var[w, t_k] = 0.0f0
                end
            end
            idx += 1
        end
    end
    return nothing
end

# Horizontal Layered + OMS (SimdInterMode, `Float32` or `Int16`)
@inline function _decode_single_ite_layered_inter!(
    rule::RuleOMS{T},
    sat_val::T,
    chk_offsets::Vector{Int32},
    chk_to_var::Vector{Int32},
    work::LDPC_Workspace{SimdInterMode, T},
    W::Int
) where {T<:Union{Float32, Int16}}
    n_chk_nodes   = length(chk_offsets) - 1
    zero_T        = zero(T)
    max_T         = _max_val(T)
    off           = rule.offset
    var_nodes     = work.post_inter
    messages      = work.msg_chk_to_var_inter
    contrib_inter = work.contrib_inter
    sign_w        = work.sign_w
    min1_w        = work.min1_w
    min2_w        = work.min2_w
    cst1_w        = work.cst1_w
    cst2_w        = work.cst2_w

    @inbounds for c in 1:n_chk_nodes
        k_start = Int(chk_offsets[c])
        k_stop  = Int(chk_offsets[c + 1]) - 1

        @simd ivdep for w in 1:W
            sign_w[w] = 0x00
            min1_w[w] = max_T
            min2_w[w] = max_T
        end

        idx = 1
        for k in k_start:k_stop
            u = chk_to_var[k]
            @simd ivdep for w in 1:W
                var_val              = _sub_sat(var_nodes[w, u], messages[w, k])
                contrib_inter[w, idx] = var_val
                var_abs              = abs(var_val)
                sng                  = ifelse(var_val < zero_T, 0x01, 0x00)
                sign_w[w]           ⊻= sng
                m1                   = min1_w[w]
                m2                   = min2_w[w]
                min2_w[w]            = min(m2, max(var_abs, m1))
                min1_w[w]            = min(m1, var_abs)
            end
            idx += 1
        end

        @simd ivdep for w in 1:W
            cst1_w[w] = max(zero_T, T(min2_w[w] - off))
            cst2_w[w] = max(zero_T, T(min1_w[w] - off))
        end

        idx = 1
        for k in k_start:k_stop
            u = chk_to_var[k]
            @simd ivdep for w in 1:W
                var_val          = contrib_inter[w, idx]
                var_abs          = abs(var_val)
                res_abs          = ifelse(var_abs == min1_w[w], cst1_w[w], cst2_w[w])
                sng              = sign_w[w] ⊻ ifelse(var_val < zero_T, 0x01, 0x00)
                new_msg          = _saturate(ifelse(sng != 0x00, -res_abs, res_abs), sat_val)
                messages[w, k]   = new_msg
                var_nodes[w, u]  = _add_sat(var_val, new_msg, sat_val)
            end
            idx += 1
        end
    end
    return nothing
end

# Horizontal Layered + SPA (SimdInterMode, `Float32`)
@inline function _decode_single_ite_layered_inter!(
    ::RuleSPA,
    ::Float32,
    chk_offsets::Vector{Int32},
    chk_to_var::Vector{Int32},
    work::LDPC_Workspace{SimdInterMode, Float32},
    W::Int
)
    n_chk_nodes   = length(chk_offsets) - 1
    max_tanh      = 1.0f0 - eps(Float32)
    var_nodes     = work.post_inter
    messages      = work.msg_chk_to_var_inter
    values_inter  = work.values_inter
    contrib_inter = work.contrib_inter
    sign_w        = work.sign_w
    prod_w        = work.prod_w
    nzeros_w      = work.nzeros_w

    @inbounds for c in 1:n_chk_nodes
        k_start = Int(chk_offsets[c])
        k_stop  = Int(chk_offsets[c + 1]) - 1

        @simd ivdep for w in 1:W
            sign_w[w]   = 0x00
            prod_w[w]   = 1.0f0
            nzeros_w[w] = Int32(0)
        end

        idx = 1
        for k in k_start:k_stop
            u = chk_to_var[k]
            @simd ivdep for w in 1:W
                var_val              = var_nodes[w, u] - messages[w, k]
                contrib_inter[w, idx] = var_val
                var_abs              = abs(var_val)
                res                  = tanh(var_abs * 0.5f0)
                is_z                 = (res == 0.0f0)
                sng                  = ifelse(signbit(var_val), 0x01, 0x00)
                sign_w[w]            = ifelse(is_z, sign_w[w], sign_w[w] ⊻ sng)
                prod_w[w]            = ifelse(is_z, prod_w[w], prod_w[w] * res)
                nzeros_w[w]          = ifelse(is_z, nzeros_w[w] + Int32(1), nzeros_w[w])
                values_inter[w, idx] = res
            end
            idx += 1
        end

        idx = 1
        for k in k_start:k_stop
            u = chk_to_var[k]
            for w in 1:W
                nz      = nzeros_w[w]
                val_v   = values_inter[w, idx]
                var_val = contrib_inter[w, idx]
                is_z    = (val_v == 0.0f0)
                if nz == Int32(0)
                    res_tmp = prod_w[w] / val_v
                    res_tmp = ifelse(res_tmp < 1.0f0, res_tmp, max_tanh)
                    res_abs = 2.0f0 * atanh(res_tmp)
                    sng     = sign_w[w] ⊻ ifelse(signbit(var_val), 0x01, 0x00)
                    new_msg = ifelse(sng != 0x00, -res_abs, res_abs)
                elseif nz == Int32(1) && is_z
                    res_tmp = ifelse(prod_w[w] < 1.0f0, prod_w[w], max_tanh)
                    res_abs = 2.0f0 * atanh(res_tmp)
                    new_msg = ifelse(sign_w[w] != 0x00, -res_abs, res_abs)
                else
                    new_msg = 0.0f0
                end
                messages[w, k]  = new_msg
                var_nodes[w, u] = var_val + new_msg
            end
            idx += 1
        end
    end
    return nothing
end

@inline function _check_syndrome_soft_inter(
    chk_offsets::Vector{Int32},
    chk_to_var::Vector{Int32},
    post_inter::Matrix{T},
    synd_w::Vector{UInt8},
    W::Int
)::Bool where {T<:Union{Float32, Int16}}
    n_chk_nodes = length(chk_offsets) - 1
    zero_T      = zero(T)
    @inbounds for c in 1:n_chk_nodes
        k_start = Int(chk_offsets[c])
        k_stop  = Int(chk_offsets[c + 1]) - 1
        @simd ivdep for w in 1:W
            synd_w[w] = 0x00
        end
        for k in k_start:k_stop
            u = chk_to_var[k]
            @simd ivdep for w in 1:W
                llr = post_inter[w, u]
                synd_w[w] ⊻= ifelse(llr < zero_T, 0x01, 0x00)
            end
        end
        any_fail = 0x00
        @simd for w in 1:W
            any_fail |= synd_w[w]
        end
        if any_fail != 0x00
            return false
        end
    end
    return true
end

# ====================================================================================================
# 3. TOP-LEVEL DISPATCHED ENTRY POINTS (`decode_siho_julia`)
# ====================================================================================================

# 3.a Scalar + Flooding (`ScheduleFlooding`, `ScalarMode`)
function decode_siho_julia(
    graph::LDPC_Graph{ScheduleFlooding, U, T},
    work::LDPC_Workspace{ScalarMode, T},
    sck_Y_N::AbstractVector{T},
    sck_V_K::AbstractVector{B},
    frame_id::UInt32 = UInt32(0),
    n_frames_per_wave::UInt32 = UInt32(1)
)::Int32 where {U<:AbstractUpdateRule, T<:Union{Float32, Int16}, B<:Union{Int32, Int16}}
    K       = length(graph.info_bits_pos)
    n_ite   = Int(graph.n_ite)
    s_depth = Int(graph.syndrome_depth)
    zero_T  = zero(T)
    zero_B  = zero(B)
    one_B   = one(B)

    fill!(work.msg_chk_to_var, zero_T)
    cur_syndrome_depth = 0
    valid_synd = true
    ite_done   = 0

    for ite in 1:n_ite
        _initialize_var_to_chk!(graph.var_offsets, sck_Y_N, work.msg_chk_to_var, work.msg_var_to_chk)
        _decode_single_ite_flooding!(
            graph.rule,
            graph.sat_val,
            graph.chk_offsets,
            graph.transpose,
            work.values,
            work.msg_var_to_chk,
            work.msg_chk_to_var
        )

        if graph.enable_syndrome && ite != n_ite
            _compute_post!(graph.var_offsets, sck_Y_N, work.msg_chk_to_var, work.post)
            syndrome = _check_syndrome_soft(graph.chk_offsets, graph.chk_to_var, work.post)
            cur_syndrome_depth = syndrome ? ((cur_syndrome_depth + 1) % s_depth) : 0
            valid_synd = syndrome && (cur_syndrome_depth == 0)
            if valid_synd
                ite_done = ite
                break
            end
        end

        if ite == n_ite
            ite_done = n_ite
        end
    end

    if ite_done == n_ite
        _compute_post!(graph.var_offsets, sck_Y_N, work.msg_chk_to_var, work.post)
    end

    @inbounds for v in 1:K
        k = graph.info_bits_pos[v]
        sck_V_K[v] = ifelse(work.post[k] < zero_T, one_B, zero_B)
    end

    return ifelse(valid_synd, Int32(0), Int32(1))
end

# 3.b Scalar + Horizontal Layered (`ScheduleLayered`, `ScalarMode`)
function decode_siho_julia(
    graph::LDPC_Graph{ScheduleLayered, U, T},
    work::LDPC_Workspace{ScalarMode, T},
    sck_Y_N::AbstractVector{T},
    sck_V_K::AbstractVector{B},
    frame_id::UInt32 = UInt32(0),
    n_frames_per_wave::UInt32 = UInt32(1)
)::Int32 where {U<:AbstractUpdateRule, T<:Union{Float32, Int16}, B<:Union{Int32, Int16}}
    K           = length(graph.info_bits_pos)
    n_var_nodes = length(graph.var_offsets) - 1
    n_ite       = Int(graph.n_ite)
    s_depth     = Int(graph.syndrome_depth)
    zero_T      = zero(T)
    zero_B      = zero(B)
    one_B       = one(B)

    @inbounds for v in 1:n_var_nodes
        work.post[v] = sck_Y_N[v]
    end
    fill!(work.msg_chk_to_var, zero_T)
    cur_syndrome_depth = 0
    valid_synd = true

    for _ in 1:n_ite
        _decode_single_ite_layered!(
            graph.rule,
            graph.sat_val,
            graph.chk_offsets,
            graph.chk_to_var,
            work.values,
            work.contrib,
            work.post,
            work.msg_chk_to_var
        )

        if graph.enable_syndrome
            syndrome = _check_syndrome_soft(graph.chk_offsets, graph.chk_to_var, work.post)
            cur_syndrome_depth = syndrome ? ((cur_syndrome_depth + 1) % s_depth) : 0
            valid_synd = syndrome && (cur_syndrome_depth == 0)
            if valid_synd
                break
            end
        end
    end

    @inbounds for v in 1:K
        k = graph.info_bits_pos[v]
        sck_V_K[v] = ifelse(work.post[k] < zero_T, one_B, zero_B)
    end

    return ifelse(valid_synd, Int32(0), Int32(1))
end

# 3.c Inter-Frame SIMD + Flooding (`ScheduleFlooding`, `SimdInterMode`)
function decode_siho_julia(
    graph::LDPC_Graph{ScheduleFlooding, U, T},
    work::LDPC_Workspace{SimdInterMode, T},
    sck_Y_N::AbstractVector{T},
    sck_V_K::AbstractVector{B},
    frame_id::UInt32 = UInt32(0),
    n_frames_per_wave::UInt32 = UInt32(1)
)::Int32 where {U<:AbstractUpdateRule, T<:Union{Float32, Int16}, B<:Union{Int32, Int16}}
    K           = length(graph.info_bits_pos)
    n_var_nodes = length(graph.var_offsets) - 1
    n_branches  = length(graph.transpose)
    W           = Int(n_frames_per_wave)
    n_ite       = Int(graph.n_ite)
    s_depth     = Int(graph.syndrome_depth)
    zero_T      = zero(T)

    _ensure_inter_capacity!(work, W, n_var_nodes, n_branches, size(work.values_inter, 2))
    _transpose_in!(work.Y_N_inter, sck_Y_N, n_var_nodes, W)
    fill!(work.msg_chk_to_var_inter, zero_T)

    cur_syndrome_depth = 0
    valid_synd = true
    ite_done   = 0

    for ite in 1:n_ite
        _initialize_var_to_chk_inter!(
            graph.var_offsets,
            work.Y_N_inter,
            work.msg_chk_to_var_inter,
            work.msg_var_to_chk_inter,
            work.post_inter,
            W
        )
        _decode_single_ite_flooding_inter!(
            graph.rule,
            graph.sat_val,
            graph.chk_offsets,
            graph.transpose,
            work,
            W
        )

        if graph.enable_syndrome && ite != n_ite
            _compute_post_inter!(
                graph.var_offsets,
                work.Y_N_inter,
                work.msg_chk_to_var_inter,
                work.post_inter,
                W
            )
            syndrome = _check_syndrome_soft_inter(graph.chk_offsets, graph.chk_to_var, work.post_inter, work.synd_w, W)
            cur_syndrome_depth = syndrome ? ((cur_syndrome_depth + 1) % s_depth) : 0
            valid_synd = syndrome && (cur_syndrome_depth == 0)
            if valid_synd
                ite_done = ite
                break
            end
        end

        if ite == n_ite
            ite_done = n_ite
        end
    end

    if ite_done == n_ite
        _compute_post_inter!(
            graph.var_offsets,
            work.Y_N_inter,
            work.msg_chk_to_var_inter,
            work.post_inter,
            W
        )
    end

    _extract_hard_decision_inter!(sck_V_K, work.post_inter, graph.info_bits_pos, K, W)
    return ifelse(valid_synd, Int32(0), Int32(1))
end

# 3.d Inter-Frame SIMD + Horizontal Layered (`ScheduleLayered`, `SimdInterMode`)
function decode_siho_julia(
    graph::LDPC_Graph{ScheduleLayered, U, T},
    work::LDPC_Workspace{SimdInterMode, T},
    sck_Y_N::AbstractVector{T},
    sck_V_K::AbstractVector{B},
    frame_id::UInt32 = UInt32(0),
    n_frames_per_wave::UInt32 = UInt32(1)
)::Int32 where {U<:AbstractUpdateRule, T<:Union{Float32, Int16}, B<:Union{Int32, Int16}}
    K           = length(graph.info_bits_pos)
    n_var_nodes = length(graph.var_offsets) - 1
    n_branches  = length(graph.transpose)
    W           = Int(n_frames_per_wave)
    n_ite       = Int(graph.n_ite)
    s_depth     = Int(graph.syndrome_depth)
    zero_T      = zero(T)

    _ensure_inter_capacity!(work, W, n_var_nodes, n_branches, size(work.values_inter, 2))
    _transpose_in!(work.post_inter, sck_Y_N, n_var_nodes, W)
    fill!(work.msg_chk_to_var_inter, zero_T)

    cur_syndrome_depth = 0
    valid_synd = true

    for _ in 1:n_ite
        _decode_single_ite_layered_inter!(
            graph.rule,
            graph.sat_val,
            graph.chk_offsets,
            graph.chk_to_var,
            work,
            W
        )

        if graph.enable_syndrome
            syndrome = _check_syndrome_soft_inter(graph.chk_offsets, graph.chk_to_var, work.post_inter, work.synd_w, W)
            cur_syndrome_depth = syndrome ? ((cur_syndrome_depth + 1) % s_depth) : 0
            valid_synd = syndrome && (cur_syndrome_depth == 0)
            if valid_synd
                break
            end
        end
    end

    _extract_hard_decision_inter!(sck_V_K, work.post_inter, graph.info_bits_pos, K, W)
    return ifelse(valid_synd, Int32(0), Int32(1))
end

# Precompile all 12 concrete specializations (2 schedules x 3 rule/type pairs x 2 execution modes)
for S in (ScheduleFlooding, ScheduleLayered)
    for M in (ScalarMode, SimdInterMode)
        precompile(decode_siho_julia, (LDPC_Graph{S, RuleSPA, Float32}, LDPC_Workspace{M, Float32}, Vector{Float32}, Vector{Int32}, UInt32, UInt32))
        precompile(decode_siho_julia, (LDPC_Graph{S, RuleOMS{Float32}, Float32}, LDPC_Workspace{M, Float32}, Vector{Float32}, Vector{Int32}, UInt32, UInt32))
        precompile(decode_siho_julia, (LDPC_Graph{S, RuleOMS{Int16}, Int16}, LDPC_Workspace{M, Int16}, Vector{Int16}, Vector{Int16}, UInt32, UInt32))
        precompile(decode_siho_julia, (LDPC_Graph{S, RuleOMS{Int16}, Int16}, LDPC_Workspace{M, Int16}, Vector{Int16}, Vector{Int32}, UInt32, UInt32))
    end
end
