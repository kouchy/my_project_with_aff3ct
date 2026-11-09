# Generic Modular LDPC Belief Propagation Decoder in Julia for StreamPU (`Stateful_Julia`)
# Tier 3 (`ldpc_bp_opt.jl`): Hardware-Optimized LLVM SIMD / Register-Resident Implementation.
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
struct ScalarMode       <: AbstractExecMode end
struct SimdInterMode{W} <: AbstractExecMode end
SimdInterMode(W::Integer = 1) = SimdInterMode{Int(W)}()

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
        return LDPC_Workspace{SimdInterMode{W}, T}(
            SimdInterMode(W),
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
            zeros(T, W, mc)
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
            Matrix{T}(undef, 0, 0)
        )
    end
end

# Backward-compatible constructor defaulting to `(ScalarMode, Float32)`.
function LDPC_Workspace(n_var_nodes::Int32, n_branches::Int32, max_chk_degree::Int32)
    return LDPC_Workspace(:SCALAR, Float32, n_var_nodes, n_branches, max_chk_degree, Int32(1))
end

# Ensures `SimdInterMode` buffers match the runtime `W = n_frames_per_wave` if `set_n_frames` was called after constructor.
@inline function _ensure_inter_capacity!(
    work::LDPC_Workspace{<:SimdInterMode, T},
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
    ::Val{W}
) where {T<:Union{Float32, Int16}, W}
    # Optimal hardware SIMD lane count for T:
    # - ARM64 NEON (16B): 4 for Float32, 8 for Int16
    # - x86 AVX2   (32B): 8 for Float32, 16 for Int16
    # - x86 AVX512 (64B): 16 for Float32, 32 for Int16
    hw_lanes = _SIMD_REG_BYTES ÷ sizeof(T)
    if hw_lanes >= 32 && W % 32 == 0
        if N_cw % 32 == 0
            _transpose_in_tiled!(Y_N_inter, sck_Y_N, N_cw, W, Val(32))
            return nothing
        end
    elseif hw_lanes >= 16 && W % 16 == 0
        if N_cw % 16 == 0
            _transpose_in_tiled!(Y_N_inter, sck_Y_N, N_cw, W, Val(16))
            return nothing
        end
    elseif hw_lanes >= 8 && W % 8 == 0
        if N_cw % 8 == 0
            _transpose_in_tiled!(Y_N_inter, sck_Y_N, N_cw, W, Val(8))
            return nothing
        end
    elseif W % 4 == 0
        if N_cw % 4 == 0
            _transpose_in_tiled!(Y_N_inter, sck_Y_N, N_cw, W, Val(4))
            return nothing
        end
    end
    @inbounds for w in 1:W
        offset = (w - 1) * N_cw
        for v in 1:N_cw
            Y_N_inter[w, v] = sck_Y_N[offset + v]
        end
    end
    return nothing
end

@inline function _extract_hard_decision_inter!(
    sck_V_K::AbstractVector{B},
    post_inter::Matrix{T},
    info_bits_pos::Vector{Int32},
    K::Int,
    ::Val{W}
) where {B<:Union{Int32, Int16}, T<:Union{Float32, Int16}, W}
    hw_lanes = _SIMD_REG_BYTES ÷ max(sizeof(B), sizeof(T))
    if hw_lanes >= 32 && W % 32 == 0
        if K % 32 == 0
            _extract_hard_decision_tiled!(sck_V_K, post_inter, info_bits_pos, K, W, Val(32))
            return nothing
        end
    elseif hw_lanes >= 16 && W % 16 == 0
        if K % 16 == 0
            _extract_hard_decision_tiled!(sck_V_K, post_inter, info_bits_pos, K, W, Val(16))
            return nothing
        end
    elseif hw_lanes >= 8 && W % 8 == 0
        if K % 8 == 0
            _extract_hard_decision_tiled!(sck_V_K, post_inter, info_bits_pos, K, W, Val(8))
            return nothing
        end
    elseif W % 4 == 0
        if K % 4 == 0
            _extract_hard_decision_tiled!(sck_V_K, post_inter, info_bits_pos, K, W, Val(4))
            return nothing
        end
    end
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
    return nothing
end

# ====================================================================================================
# 2.b REGISTER-RESIDENT LLVM SIMD VECTOR PRIMITIVES & KERNELS (`SimdInterMode`)
# ====================================================================================================

@inline _vbroadcast(::Val{W}, x::T) where {W, T} = ntuple(_ -> Base.VecElement(x), Val(W))

@inline _vload(::Val{W}, ptr::Ptr{T}, col::Int) where {W, T} =
    unsafe_load(Ptr{NTuple{W, Base.VecElement{T}}}(ptr + (col - 1) * W * sizeof(T)))

@inline _vstore!(::Val{W}, ptr::Ptr{T}, col::Int, v::NTuple{W, Base.VecElement{T}}) where {W, T} =
    unsafe_store!(Ptr{NTuple{W, Base.VecElement{T}}}(ptr + (col - 1) * W * sizeof(T)), v)

# --- Float32 SIMD vector operations ---
@generated function _vfadd(a::NTuple{W, Base.VecElement{Float32}}, b::NTuple{W, Base.VecElement{Float32}}) where {W}
    ir = "%r = fadd fast <$W x float> %0, %1\nret <$W x float> %r"
    return :($(Expr(:meta, :inline)); Base.llvmcall($ir, NTuple{$W, Base.VecElement{Float32}}, Tuple{NTuple{$W, Base.VecElement{Float32}}, NTuple{$W, Base.VecElement{Float32}}}, a, b))
end

@generated function _vfsub(a::NTuple{W, Base.VecElement{Float32}}, b::NTuple{W, Base.VecElement{Float32}}) where {W}
    ir = "%r = fsub fast <$W x float> %0, %1\nret <$W x float> %r"
    return :($(Expr(:meta, :inline)); Base.llvmcall($ir, NTuple{$W, Base.VecElement{Float32}}, Tuple{NTuple{$W, Base.VecElement{Float32}}, NTuple{$W, Base.VecElement{Float32}}}, a, b))
end

@generated function _vfmul(a::NTuple{W, Base.VecElement{Float32}}, b::NTuple{W, Base.VecElement{Float32}}) where {W}
    ir = "%r = fmul fast <$W x float> %0, %1\nret <$W x float> %r"
    return :($(Expr(:meta, :inline)); Base.llvmcall($ir, NTuple{$W, Base.VecElement{Float32}}, Tuple{NTuple{$W, Base.VecElement{Float32}}, NTuple{$W, Base.VecElement{Float32}}}, a, b))
end

@generated function _vfdiv(a::NTuple{W, Base.VecElement{Float32}}, b::NTuple{W, Base.VecElement{Float32}}) where {W}
    ir = "%r = fdiv fast <$W x float> %0, %1\nret <$W x float> %r"
    return :($(Expr(:meta, :inline)); Base.llvmcall($ir, NTuple{$W, Base.VecElement{Float32}}, Tuple{NTuple{$W, Base.VecElement{Float32}}, NTuple{$W, Base.VecElement{Float32}}}, a, b))
end

@generated function _vfmin(a::NTuple{W, Base.VecElement{Float32}}, b::NTuple{W, Base.VecElement{Float32}}) where {W}
    ir = "%c = fcmp fast olt <$W x float> %0, %1\n%r = select fast <$W x i1> %c, <$W x float> %0, <$W x float> %1\nret <$W x float> %r"
    return :($(Expr(:meta, :inline)); Base.llvmcall($ir, NTuple{$W, Base.VecElement{Float32}}, Tuple{NTuple{$W, Base.VecElement{Float32}}, NTuple{$W, Base.VecElement{Float32}}}, a, b))
end

@generated function _vfmax(a::NTuple{W, Base.VecElement{Float32}}, b::NTuple{W, Base.VecElement{Float32}}) where {W}
    ir = "%c = fcmp fast ogt <$W x float> %0, %1\n%r = select fast <$W x i1> %c, <$W x float> %0, <$W x float> %1\nret <$W x float> %r"
    return :($(Expr(:meta, :inline)); Base.llvmcall($ir, NTuple{$W, Base.VecElement{Float32}}, Tuple{NTuple{$W, Base.VecElement{Float32}}, NTuple{$W, Base.VecElement{Float32}}}, a, b))
end

@generated function _vfabs(a::NTuple{W, Base.VecElement{Float32}}) where {W}
    c_abs = join(["i32 2147483647" for _ in 1:W], ", ")
    ir = """
    %i = bitcast <$W x float> %0 to <$W x i32>
    %a = and <$W x i32> %i, <$c_abs>
    %r = bitcast <$W x i32> %a to <$W x float>
    ret <$W x float> %r
    """
    return :($(Expr(:meta, :inline)); Base.llvmcall($ir, NTuple{$W, Base.VecElement{Float32}}, Tuple{NTuple{$W, Base.VecElement{Float32}}}, a))
end

@generated function _vfsignmask(a::NTuple{W, Base.VecElement{Float32}}) where {W}
    c_sgn = join(["i32 -2147483648" for _ in 1:W], ", ")
    ir = """
    %c = fcmp olt <$W x float> %0, zeroinitializer
    %s = select <$W x i1> %c, <$W x i32> <$c_sgn>, <$W x i32> zeroinitializer
    ret <$W x i32> %s
    """
    return :($(Expr(:meta, :inline)); Base.llvmcall($ir, NTuple{$W, Base.VecElement{Int32}}, Tuple{NTuple{$W, Base.VecElement{Float32}}}, a))
end

@generated function _vxor_i32(a::NTuple{W, Base.VecElement{Int32}}, b::NTuple{W, Base.VecElement{Int32}}) where {W}
    ir = "%r = xor <$W x i32> %0, %1\nret <$W x i32> %r"
    return :($(Expr(:meta, :inline)); Base.llvmcall($ir, NTuple{$W, Base.VecElement{Int32}}, Tuple{NTuple{$W, Base.VecElement{Int32}}, NTuple{$W, Base.VecElement{Int32}}}, a, b))
end

@generated function _vf32_oms_out(
    var_abs::NTuple{W, Base.VecElement{Float32}},
    min1::NTuple{W, Base.VecElement{Float32}},
    cst1::NTuple{W, Base.VecElement{Float32}},
    cst2::NTuple{W, Base.VecElement{Float32}},
    sng::NTuple{W, Base.VecElement{Int32}}
) where {W}
    ir = """
    %eq  = fcmp oeq <$W x float> %0, %1
    %mag = select <$W x i1> %eq, <$W x float> %2, <$W x float> %3
    %mi  = bitcast <$W x float> %mag to <$W x i32>
    %ri  = xor <$W x i32> %mi, %4
    %res = bitcast <$W x i32> %ri to <$W x float>
    ret <$W x float> %res
    """
    return :($(Expr(:meta, :inline)); Base.llvmcall($ir, NTuple{$W, Base.VecElement{Float32}},
        Tuple{NTuple{$W, Base.VecElement{Float32}}, NTuple{$W, Base.VecElement{Float32}}, NTuple{$W, Base.VecElement{Float32}}, NTuple{$W, Base.VecElement{Float32}}, NTuple{$W, Base.VecElement{Int32}}},
        var_abs, min1, cst1, cst2, sng))
end

# --- Branchless SIMD Cephes `exp`, `log`, `tanh`, `atanh` (matching AFF3CT MIPP `neon_mathfun` / `avx_mathfun`) ---
@generated function _vexp_floor(fx1::NTuple{W, Base.VecElement{Float32}}, one_v::NTuple{W, Base.VecElement{Float32}}) where {W}
    ir = """
    %fxi  = fptosi <$W x float> %0 to <$W x i32>
    %tmp  = sitofp <$W x i32> %fxi to <$W x float>
    %gt   = fcmp ogt <$W x float> %tmp, %0
    %sub1 = select <$W x i1> %gt, <$W x float> %1, <$W x float> zeroinitializer
    %fx   = fsub fast <$W x float> %tmp, %sub1
    ret <$W x float> %fx
    """
    return :($(Expr(:meta, :inline)); Base.llvmcall($ir, NTuple{$W, Base.VecElement{Float32}}, Tuple{NTuple{$W, Base.VecElement{Float32}}, NTuple{$W, Base.VecElement{Float32}}}, fx1, one_v))
end

@generated function _vexp_pow2n(fx::NTuple{W, Base.VecElement{Float32}}, c7f::NTuple{W, Base.VecElement{Int32}}, c23::NTuple{W, Base.VecElement{Int32}}) where {W}
    ir = """
    %mm0 = fptosi <$W x float> %0 to <$W x i32>
    %mm1 = add <$W x i32> %mm0, %1
    %mm2 = shl <$W x i32> %mm1, %2
    %p2n = bitcast <$W x i32> %mm2 to <$W x float>
    ret <$W x float> %p2n
    """
    return :($(Expr(:meta, :inline)); Base.llvmcall($ir, NTuple{$W, Base.VecElement{Float32}}, Tuple{NTuple{$W, Base.VecElement{Float32}}, NTuple{$W, Base.VecElement{Int32}}, NTuple{$W, Base.VecElement{Int32}}}, fx, c7f, c23))
end

@generated function _vlog_decompose(
    x::NTuple{W, Base.VecElement{Float32}},
    sqhf::NTuple{W, Base.VecElement{Float32}},
    one_v::NTuple{W, Base.VecElement{Float32}},
    c23::NTuple{W, Base.VecElement{Int32}},
    c7f::NTuple{W, Base.VecElement{Int32}},
    cmask::NTuple{W, Base.VecElement{Int32}},
    c05i::NTuple{W, Base.VecElement{Int32}}
) where {W}
    ir = """
    %ux0  = bitcast <$W x float> %0 to <$W x i32>
    %emm0 = ashr <$W x i32> %ux0, %3
    %ux1  = and <$W x i32> %ux0, %5
    %ux2  = or <$W x i32> %ux1, %6
    %xm0  = bitcast <$W x i32> %ux2 to <$W x float>
    %emm1 = sub <$W x i32> %emm0, %4
    %e0   = sitofp <$W x i32> %emm1 to <$W x float>
    %e1   = fadd fast <$W x float> %e0, %2
    %lt   = fcmp olt <$W x float> %xm0, %1
    %tmp0 = select <$W x i1> %lt, <$W x float> %xm0, <$W x float> zeroinitializer
    %edec = select <$W x i1> %lt, <$W x float> %2, <$W x float> zeroinitializer
    %xm1  = fsub fast <$W x float> %xm0, %2
    %e    = fsub fast <$W x float> %e1, %edec
    %xr   = fadd fast <$W x float> %xm1, %tmp0
    %res0 = insertvalue [2 x <$W x float>] undef, <$W x float> %xr, 0
    %res1 = insertvalue [2 x <$W x float>] %res0, <$W x float> %e, 1
    ret [2 x <$W x float>] %res1
    """
    return :($(Expr(:meta, :inline)); Base.llvmcall($ir, NTuple{2, NTuple{$W, Base.VecElement{Float32}}},
        Tuple{NTuple{$W, Base.VecElement{Float32}}, NTuple{$W, Base.VecElement{Float32}}, NTuple{$W, Base.VecElement{Float32}}, NTuple{$W, Base.VecElement{Int32}}, NTuple{$W, Base.VecElement{Int32}}, NTuple{$W, Base.VecElement{Int32}}, NTuple{$W, Base.VecElement{Int32}}},
        x, sqhf, one_v, c23, c7f, cmask, c05i))
end

@inline function _vexp_cephes(x::NTuple{W, Base.VecElement{Float32}}) where {W}
    vw    = Val(W)
    one_v = _vbroadcast(vw, 1.0f0)
    x     = _vfmin(x, _vbroadcast(vw, 88.3762626647949f0))
    x     = _vfmax(x, _vbroadcast(vw, -88.3762626647949f0))
    fx1   = _vfadd(_vfmul(x, _vbroadcast(vw, 1.44269504088896341f0)), _vbroadcast(vw, 0.5f0))
    fx    = _vexp_floor(fx1, one_v)
    x     = _vfsub(_vfsub(x, _vfmul(fx, _vbroadcast(vw, 0.693359375f0))), _vfmul(fx, _vbroadcast(vw, -2.12194440f-4)))
    z     = _vfmul(x, x)
    y     = _vfadd(_vfmul(_vbroadcast(vw, 1.9875691500f-4), x), _vbroadcast(vw, 1.3981999507f-3))
    y     = _vfadd(_vfmul(y, x), _vbroadcast(vw, 8.3334519073f-3))
    y     = _vfadd(_vfmul(y, x), _vbroadcast(vw, 4.1665795894f-2))
    y     = _vfadd(_vfmul(y, x), _vbroadcast(vw, 1.6666665459f-1))
    y     = _vfadd(_vfmul(y, x), _vbroadcast(vw, 5.0000001201f-1))
    y     = _vfadd(_vfadd(_vfmul(y, z), x), one_v)
    p2n   = _vexp_pow2n(fx, _vbroadcast(vw, Int32(127)), _vbroadcast(vw, Int32(23)))
    return _vfmul(y, p2n)
end

@inline function _vlog_cephes(x::NTuple{W, Base.VecElement{Float32}}) where {W}
    vw = Val(W)
    xr, e = _vlog_decompose(
        x,
        _vbroadcast(vw, 0.707106781186547524f0),
        _vbroadcast(vw, 1.0f0),
        _vbroadcast(vw, Int32(23)),
        _vbroadcast(vw, Int32(127)),
        _vbroadcast(vw, Int32(-2139095041)),
        _vbroadcast(vw, Int32(1056964608))
    )
    z = _vfmul(xr, xr)
    y = _vfadd(_vfmul(_vbroadcast(vw, 7.0376836292f-2), xr), _vbroadcast(vw, -1.1514610310f-1))
    y = _vfadd(_vfmul(y, xr), _vbroadcast(vw, 1.1676998740f-1))
    y = _vfadd(_vfmul(y, xr), _vbroadcast(vw, -1.2420140846f-1))
    y = _vfadd(_vfmul(y, xr), _vbroadcast(vw, 1.4249322787f-1))
    y = _vfadd(_vfmul(y, xr), _vbroadcast(vw, -1.6668057665f-1))
    y = _vfadd(_vfmul(y, xr), _vbroadcast(vw, 2.0000714765f-1))
    y = _vfadd(_vfmul(y, xr), _vbroadcast(vw, -2.4999993993f-1))
    y = _vfadd(_vfmul(y, xr), _vbroadcast(vw, 3.3333331174f-1))
    y = _vfmul(_vfmul(y, xr), z)
    y = _vfadd(y, _vfmul(e, _vbroadcast(vw, -2.12194440f-4)))
    y = _vfsub(y, _vfmul(z, _vbroadcast(vw, 0.5f0)))
    return _vfadd(_vfadd(xr, y), _vfmul(e, _vbroadcast(vw, 0.693359375f0)))
end

@inline function _vtanh_cephes(x::NTuple{W, Base.VecElement{Float32}}) where {W}
    epx = _vexp_cephes(x)
    emx = _vexp_cephes(_vfsub(_vbroadcast(Val(W), 0.0f0), x))
    return _vfdiv(_vfsub(epx, emx), _vfadd(epx, emx))
end

@inline function _vatanh_cephes(x::NTuple{W, Base.VecElement{Float32}}) where {W}
    one_v  = _vbroadcast(Val(W), 1.0f0)
    half_v = _vbroadcast(Val(W), 0.5f0)
    return _vfmul(half_v, _vlog_cephes(_vfdiv(_vfadd(one_v, x), _vfsub(one_v, x))))
end

@generated function _vspa_in(
    res::NTuple{W, Base.VecElement{Float32}},
    var_val::NTuple{W, Base.VecElement{Float32}},
    sign_v::NTuple{W, Base.VecElement{Int32}},
    prod_v::NTuple{W, Base.VecElement{Float32}},
    nzeros_v::NTuple{W, Base.VecElement{Int32}}
) where {W}
    c_sgn = join(["i32 -2147483648" for _ in 1:W], ", ")
    c_one = join(["i32 1" for _ in 1:W], ", ")
    ir = """
    %isz   = fcmp oeq <$W x float> %0, zeroinitializer
    %vi    = bitcast <$W x float> %1 to <$W x i32>
    %vsng  = and <$W x i32> %vi, <$c_sgn>
    %sxor  = xor <$W x i32> %2, %vsng
    %nsign = select <$W x i1> %isz, <$W x i32> %2, <$W x i32> %sxor
    %pmul  = fmul fast <$W x float> %3, %0
    %nprod = select <$W x i1> %isz, <$W x float> %3, <$W x float> %pmul
    %zinc  = add <$W x i32> %4, <$c_one>
    %nnz   = select <$W x i1> %isz, <$W x i32> %zinc, <$W x i32> %4
    %r0 = insertvalue { <$W x i32>, <$W x float>, <$W x i32> } undef, <$W x i32> %nsign, 0
    %r1 = insertvalue { <$W x i32>, <$W x float>, <$W x i32> } %r0, <$W x float> %nprod, 1
    %r2 = insertvalue { <$W x i32>, <$W x float>, <$W x i32> } %r1, <$W x i32> %nnz, 2
    ret { <$W x i32>, <$W x float>, <$W x i32> } %r2
    """
    return :($(Expr(:meta, :inline)); Base.llvmcall($ir,
        Tuple{NTuple{$W, Base.VecElement{Int32}}, NTuple{$W, Base.VecElement{Float32}}, NTuple{$W, Base.VecElement{Int32}}},
        Tuple{NTuple{$W, Base.VecElement{Float32}}, NTuple{$W, Base.VecElement{Float32}}, NTuple{$W, Base.VecElement{Int32}}, NTuple{$W, Base.VecElement{Float32}}, NTuple{$W, Base.VecElement{Int32}}},
        res, var_val, sign_v, prod_v, nzeros_v))
end

@generated function _vspa_out_prep(
    val_v::NTuple{W, Base.VecElement{Float32}},
    prod_v::NTuple{W, Base.VecElement{Float32}},
    max_tanh_v::NTuple{W, Base.VecElement{Float32}}
) where {W}
    c_one = join(["float 1.0" for _ in 1:W], ", ")
    ir = """
    %isz = fcmp oeq <$W x float> %0, zeroinitializer
    %q   = fdiv fast <$W x float> %1, %0
    %rt0 = select <$W x i1> %isz, <$W x float> %1, <$W x float> %q
    %lt1 = fcmp olt <$W x float> %rt0, <$c_one>
    %rt1 = select <$W x i1> %lt1, <$W x float> %rt0, <$W x float> %2
    ret <$W x float> %rt1
    """
    return :($(Expr(:meta, :inline)); Base.llvmcall($ir, NTuple{$W, Base.VecElement{Float32}},
        Tuple{NTuple{$W, Base.VecElement{Float32}}, NTuple{$W, Base.VecElement{Float32}}, NTuple{$W, Base.VecElement{Float32}}},
        val_v, prod_v, max_tanh_v))
end

@generated function _vspa_out_finish(
    atanh_v::NTuple{W, Base.VecElement{Float32}},
    val_v::NTuple{W, Base.VecElement{Float32}},
    var_val::NTuple{W, Base.VecElement{Float32}},
    sign_v::NTuple{W, Base.VecElement{Int32}},
    nzeros_v::NTuple{W, Base.VecElement{Int32}}
) where {W}
    c_two = join(["float 2.0" for _ in 1:W], ", ")
    c_sgn = join(["i32 -2147483648" for _ in 1:W], ", ")
    c_i1  = join(["i32 1" for _ in 1:W], ", ")
    ir = """
    %rabs  = fmul fast <$W x float> %0, <$c_two>
    %isz   = fcmp oeq <$W x float> %1, zeroinitializer
    %nz0   = icmp eq <$W x i32> %4, zeroinitializer
    %nz1   = icmp eq <$W x i32> %4, <$c_i1>
    %ok1   = and <$W x i1> %nz1, %isz
    %keep  = or <$W x i1> %nz0, %ok1
    %vi    = bitcast <$W x float> %2 to <$W x i32>
    %vsng0 = and <$W x i32> %vi, <$c_sgn>
    %vsng  = select <$W x i1> %isz, <$W x i32> zeroinitializer, <$W x i32> %vsng0
    %sng   = xor <$W x i32> %3, %vsng
    %ri0   = bitcast <$W x float> %rabs to <$W x i32>
    %ri1   = xor <$W x i32> %ri0, %sng
    %rf    = bitcast <$W x i32> %ri1 to <$W x float>
    %res   = select <$W x i1> %keep, <$W x float> %rf, <$W x float> zeroinitializer
    ret <$W x float> %res
    """
    return :($(Expr(:meta, :inline)); Base.llvmcall($ir, NTuple{$W, Base.VecElement{Float32}},
        Tuple{NTuple{$W, Base.VecElement{Float32}}, NTuple{$W, Base.VecElement{Float32}}, NTuple{$W, Base.VecElement{Float32}}, NTuple{$W, Base.VecElement{Int32}}, NTuple{$W, Base.VecElement{Int32}}},
        atanh_v, val_v, var_val, sign_v, nzeros_v))
end

# --- Int16 SIMD vector operations (with hardware saturating arithmetic `@llvm.sadd.sat` / `@llvm.ssub.sat`) ---
@generated function _vi16_add_sat(a::NTuple{W, Base.VecElement{Int16}}, b::NTuple{W, Base.VecElement{Int16}}) where {W}
    mod = """
    declare <$W x i16> @llvm.sadd.sat.v$(W)i16(<$W x i16>, <$W x i16>)
    define <$W x i16> @entry(<$W x i16> %0, <$W x i16> %1) #0 {
        %r = call <$W x i16> @llvm.sadd.sat.v$(W)i16(<$W x i16> %0, <$W x i16> %1)
        ret <$W x i16> %r
    }
    attributes #0 = { alwaysinline }
    """
    return :($(Expr(:meta, :inline)); Base.llvmcall(($mod, "entry"), NTuple{$W, Base.VecElement{Int16}}, Tuple{NTuple{$W, Base.VecElement{Int16}}, NTuple{$W, Base.VecElement{Int16}}}, a, b))
end

@generated function _vi16_sub_sat(a::NTuple{W, Base.VecElement{Int16}}, b::NTuple{W, Base.VecElement{Int16}}) where {W}
    mod = """
    declare <$W x i16> @llvm.ssub.sat.v$(W)i16(<$W x i16>, <$W x i16>)
    define <$W x i16> @entry(<$W x i16> %0, <$W x i16> %1) #0 {
        %r = call <$W x i16> @llvm.ssub.sat.v$(W)i16(<$W x i16> %0, <$W x i16> %1)
        ret <$W x i16> %r
    }
    attributes #0 = { alwaysinline }
    """
    return :($(Expr(:meta, :inline)); Base.llvmcall(($mod, "entry"), NTuple{$W, Base.VecElement{Int16}}, Tuple{NTuple{$W, Base.VecElement{Int16}}, NTuple{$W, Base.VecElement{Int16}}}, a, b))
end

@generated function _vi16_min(a::NTuple{W, Base.VecElement{Int16}}, b::NTuple{W, Base.VecElement{Int16}}) where {W}
    ir = "%c = icmp slt <$W x i16> %0, %1\n%r = select <$W x i1> %c, <$W x i16> %0, <$W x i16> %1\nret <$W x i16> %r"
    return :($(Expr(:meta, :inline)); Base.llvmcall($ir, NTuple{$W, Base.VecElement{Int16}}, Tuple{NTuple{$W, Base.VecElement{Int16}}, NTuple{$W, Base.VecElement{Int16}}}, a, b))
end

@generated function _vi16_max(a::NTuple{W, Base.VecElement{Int16}}, b::NTuple{W, Base.VecElement{Int16}}) where {W}
    ir = "%c = icmp sgt <$W x i16> %0, %1\n%r = select <$W x i1> %c, <$W x i16> %0, <$W x i16> %1\nret <$W x i16> %r"
    return :($(Expr(:meta, :inline)); Base.llvmcall($ir, NTuple{$W, Base.VecElement{Int16}}, Tuple{NTuple{$W, Base.VecElement{Int16}}, NTuple{$W, Base.VecElement{Int16}}}, a, b))
end

@generated function _vi16_abs(a::NTuple{W, Base.VecElement{Int16}}) where {W}
    ir = """
    %neg = sub <$W x i16> zeroinitializer, %0
    %c   = icmp sgt <$W x i16> %0, %neg
    %r   = select <$W x i1> %c, <$W x i16> %0, <$W x i16> %neg
    ret <$W x i16> %r
    """
    return :($(Expr(:meta, :inline)); Base.llvmcall($ir, NTuple{$W, Base.VecElement{Int16}}, Tuple{NTuple{$W, Base.VecElement{Int16}}}, a))
end

@generated function _vi16_signmask(a::NTuple{W, Base.VecElement{Int16}}) where {W}
    c15 = join(["i16 15" for _ in 1:W], ", ")
    ir = "%r = ashr <$W x i16> %0, <$c15>\nret <$W x i16> %r"
    return :($(Expr(:meta, :inline)); Base.llvmcall($ir, NTuple{$W, Base.VecElement{Int16}}, Tuple{NTuple{$W, Base.VecElement{Int16}}}, a))
end

@generated function _vi16_xor(a::NTuple{W, Base.VecElement{Int16}}, b::NTuple{W, Base.VecElement{Int16}}) where {W}
    ir = "%r = xor <$W x i16> %0, %1\nret <$W x i16> %r"
    return :($(Expr(:meta, :inline)); Base.llvmcall($ir, NTuple{$W, Base.VecElement{Int16}}, Tuple{NTuple{$W, Base.VecElement{Int16}}, NTuple{$W, Base.VecElement{Int16}}}, a, b))
end

@generated function _vi16_oms_out(
    var_abs::NTuple{W, Base.VecElement{Int16}},
    min1::NTuple{W, Base.VecElement{Int16}},
    cst1::NTuple{W, Base.VecElement{Int16}},
    cst2::NTuple{W, Base.VecElement{Int16}},
    sng::NTuple{W, Base.VecElement{Int16}},
    sat_v::NTuple{W, Base.VecElement{Int16}},
    nsat_v::NTuple{W, Base.VecElement{Int16}}
) where {W}
    ir = """
    %eq  = icmp eq <$W x i16> %0, %1
    %mag = select <$W x i1> %eq, <$W x i16> %2, <$W x i16> %3
    %x   = xor <$W x i16> %mag, %4
    %s   = sub <$W x i16> %x, %4
    %c1  = icmp sgt <$W x i16> %s, %6
    %m1  = select <$W x i1> %c1, <$W x i16> %s, <$W x i16> %6
    %c2  = icmp slt <$W x i16> %m1, %5
    %res = select <$W x i1> %c2, <$W x i16> %m1, <$W x i16> %5
    ret <$W x i16> %res
    """
    return :($(Expr(:meta, :inline)); Base.llvmcall($ir, NTuple{$W, Base.VecElement{Int16}},
        Tuple{NTuple{$W, Base.VecElement{Int16}}, NTuple{$W, Base.VecElement{Int16}}, NTuple{$W, Base.VecElement{Int16}}, NTuple{$W, Base.VecElement{Int16}}, NTuple{$W, Base.VecElement{Int16}}, NTuple{$W, Base.VecElement{Int16}}, NTuple{$W, Base.VecElement{Int16}}},
        var_abs, min1, cst1, cst2, sng, sat_v, nsat_v))
end

@generated function _vany_nonzero(v::NTuple{W, Base.VecElement{T}}) where {W, T<:Union{Int32, Int16}}
    ebits = sizeof(T) * 8
    tbits = W * ebits
    ir = """
    %b = bitcast <$W x i$ebits> %0 to i$tbits
    %c = icmp ne i$tbits %b, 0
    %r = zext i1 %c to i8
    ret i8 %r
    """
    return :($(Expr(:meta, :inline)); Base.llvmcall($ir, Bool, Tuple{NTuple{$W, Base.VecElement{$T}}}, v))
end

# --- Variable-Node & Post-LLR SIMD Kernels (`Float32` and `Int16`) ---
@inline function _initialize_var_to_chk_inter!(
    var_offsets::Vector{Int32},
    Y_N_inter::Matrix{Float32},
    msg_chk_to_var_inter::Matrix{Float32},
    msg_var_to_chk_inter::Matrix{Float32},
    ::Matrix{Float32},
    vw::Val{W}
) where {W}
    n_var_nodes = length(var_offsets) - 1
    p_Y   = pointer(Y_N_inter)
    p_c2v = pointer(msg_chk_to_var_inter)
    p_v2c = pointer(msg_var_to_chk_inter)

    @inbounds for v in 1:n_var_nodes
        e_start = Int(var_offsets[v])
        e_stop  = Int(var_offsets[v + 1]) - 1
        sum_v   = _vload(vw, p_Y, v)
        for e in e_start:e_stop
            sum_v = _vfadd(sum_v, _vload(vw, p_c2v, e))
        end
        for e in e_start:e_stop
            _vstore!(vw, p_v2c, e, _vfsub(sum_v, _vload(vw, p_c2v, e)))
        end
    end
    return nothing
end

@inline function _initialize_var_to_chk_inter!(
    var_offsets::Vector{Int32},
    Y_N_inter::Matrix{Int16},
    msg_chk_to_var_inter::Matrix{Int16},
    msg_var_to_chk_inter::Matrix{Int16},
    ::Matrix{Int16},
    vw::Val{W}
) where {W}
    n_var_nodes = length(var_offsets) - 1
    p_Y   = pointer(Y_N_inter)
    p_c2v = pointer(msg_chk_to_var_inter)
    p_v2c = pointer(msg_var_to_chk_inter)

    @inbounds for v in 1:n_var_nodes
        e_start = Int(var_offsets[v])
        e_stop  = Int(var_offsets[v + 1]) - 1
        sum_v   = _vload(vw, p_Y, v)
        for e in e_start:e_stop
            sum_v = _vi16_add_sat(sum_v, _vload(vw, p_c2v, e))
        end
        for e in e_start:e_stop
            _vstore!(vw, p_v2c, e, _vi16_sub_sat(sum_v, _vload(vw, p_c2v, e)))
        end
    end
    return nothing
end

@inline function _compute_post_inter!(
    var_offsets::Vector{Int32},
    Y_N_inter::Matrix{Float32},
    msg_chk_to_var_inter::Matrix{Float32},
    post_inter::Matrix{Float32},
    vw::Val{W}
) where {W}
    n_var_nodes = length(var_offsets) - 1
    p_Y    = pointer(Y_N_inter)
    p_c2v  = pointer(msg_chk_to_var_inter)
    p_post = pointer(post_inter)

    @inbounds for v in 1:n_var_nodes
        e_start = Int(var_offsets[v])
        e_stop  = Int(var_offsets[v + 1]) - 1
        sum_v   = _vload(vw, p_Y, v)
        for e in e_start:e_stop
            sum_v = _vfadd(sum_v, _vload(vw, p_c2v, e))
        end
        _vstore!(vw, p_post, v, sum_v)
    end
    return nothing
end

@inline function _compute_post_inter!(
    var_offsets::Vector{Int32},
    Y_N_inter::Matrix{Int16},
    msg_chk_to_var_inter::Matrix{Int16},
    post_inter::Matrix{Int16},
    vw::Val{W}
) where {W}
    n_var_nodes = length(var_offsets) - 1
    p_Y    = pointer(Y_N_inter)
    p_c2v  = pointer(msg_chk_to_var_inter)
    p_post = pointer(post_inter)

    @inbounds for v in 1:n_var_nodes
        e_start = Int(var_offsets[v])
        e_stop  = Int(var_offsets[v + 1]) - 1
        sum_v   = _vload(vw, p_Y, v)
        for e in e_start:e_stop
            sum_v = _vi16_add_sat(sum_v, _vload(vw, p_c2v, e))
        end
        _vstore!(vw, p_post, v, sum_v)
    end
    return nothing
end

# --- Flooding + OMS (`SimdInterMode`, `Float32`) ---
@inline function _decode_single_ite_flooding_inter!(
    rule::RuleOMS{Float32},
    ::Float32,
    chk_offsets::Vector{Int32},
    transpose::Vector{Int32},
    work::LDPC_Workspace{<:SimdInterMode, Float32},
    vw::Val{W}
) where {W}
    n_chk_nodes = length(chk_offsets) - 1
    zero_f32    = _vbroadcast(vw, 0.0f0)
    max_f32     = _vbroadcast(vw, floatmax(Float32))
    off_v       = _vbroadcast(vw, rule.offset)
    zero_i32    = _vbroadcast(vw, Int32(0))
    p_v2c       = pointer(work.msg_var_to_chk_inter)
    p_c2v       = pointer(work.msg_chk_to_var_inter)

    @inbounds for c in 1:n_chk_nodes
        k_start = Int(chk_offsets[c])
        k_stop  = Int(chk_offsets[c + 1]) - 1

        sign_v = zero_i32
        min1_v = max_f32
        min2_v = max_f32

        for k in k_start:k_stop
            t_k     = Int(transpose[k])
            var_val = _vload(vw, p_v2c, t_k)
            var_abs = _vfabs(var_val)
            sign_v  = _vxor_i32(sign_v, _vfsignmask(var_val))
            min2_v  = _vfmin(min2_v, _vfmax(var_abs, min1_v))
            min1_v  = _vfmin(min1_v, var_abs)
        end

        cst1_v = _vfmax(zero_f32, _vfsub(min2_v, off_v))
        cst2_v = _vfmax(zero_f32, _vfsub(min1_v, off_v))

        for k in k_start:k_stop
            t_k     = Int(transpose[k])
            var_val = _vload(vw, p_v2c, t_k)
            var_abs = _vfabs(var_val)
            sng     = _vxor_i32(sign_v, _vfsignmask(var_val))
            res     = _vf32_oms_out(var_abs, min1_v, cst1_v, cst2_v, sng)
            _vstore!(vw, p_c2v, t_k, res)
        end
    end
    return nothing
end

# --- Flooding + OMS (`SimdInterMode`, `Int16`) ---
@inline function _decode_single_ite_flooding_inter!(
    rule::RuleOMS{Int16},
    sat_val::Int16,
    chk_offsets::Vector{Int32},
    transpose::Vector{Int32},
    work::LDPC_Workspace{<:SimdInterMode, Int16},
    vw::Val{W}
) where {W}
    n_chk_nodes = length(chk_offsets) - 1
    zero_v      = _vbroadcast(vw, Int16(0))
    max_v       = _vbroadcast(vw, typemax(Int16))
    off_v       = _vbroadcast(vw, rule.offset)
    sat_v       = _vbroadcast(vw, sat_val)
    nsat_v      = _vbroadcast(vw, Int16(-sat_val))
    p_v2c       = pointer(work.msg_var_to_chk_inter)
    p_c2v       = pointer(work.msg_chk_to_var_inter)

    @inbounds for c in 1:n_chk_nodes
        k_start = Int(chk_offsets[c])
        k_stop  = Int(chk_offsets[c + 1]) - 1

        sign_v = zero_v
        min1_v = max_v
        min2_v = max_v

        for k in k_start:k_stop
            t_k     = Int(transpose[k])
            var_val = _vload(vw, p_v2c, t_k)
            var_abs = _vi16_abs(var_val)
            sign_v  = _vi16_xor(sign_v, _vi16_signmask(var_val))
            min2_v  = _vi16_min(min2_v, _vi16_max(var_abs, min1_v))
            min1_v  = _vi16_min(min1_v, var_abs)
        end

        cst1_v = _vi16_max(zero_v, _vi16_sub_sat(min2_v, off_v))
        cst2_v = _vi16_max(zero_v, _vi16_sub_sat(min1_v, off_v))

        for k in k_start:k_stop
            t_k     = Int(transpose[k])
            var_val = _vload(vw, p_v2c, t_k)
            var_abs = _vi16_abs(var_val)
            sng     = _vi16_xor(sign_v, _vi16_signmask(var_val))
            res     = _vi16_oms_out(var_abs, min1_v, cst1_v, cst2_v, sng, sat_v, nsat_v)
            _vstore!(vw, p_c2v, t_k, res)
        end
    end
    return nothing
end

# --- Flooding + SPA (`SimdInterMode`, `Float32`) ---
@inline function _decode_single_ite_flooding_inter!(
    ::RuleSPA,
    ::Float32,
    chk_offsets::Vector{Int32},
    transpose::Vector{Int32},
    work::LDPC_Workspace{<:SimdInterMode, Float32},
    vw::Val{W}
) where {W}
    n_chk_nodes = length(chk_offsets) - 1
    half_v      = _vbroadcast(vw, 0.5f0)
    max_tanh_v  = _vbroadcast(vw, 1.0f0 - eps(Float32))
    one_v       = _vbroadcast(vw, 1.0f0)
    zero_i32    = _vbroadcast(vw, Int32(0))
    p_v2c       = pointer(work.msg_var_to_chk_inter)
    p_c2v       = pointer(work.msg_chk_to_var_inter)
    p_val       = pointer(work.values_inter)

    @inbounds for c in 1:n_chk_nodes
        k_start = Int(chk_offsets[c])
        k_stop  = Int(chk_offsets[c + 1]) - 1

        sign_v   = zero_i32
        prod_v   = one_v
        nzeros_v = zero_i32

        idx = 1
        for k in k_start:k_stop
            t_k     = Int(transpose[k])
            var_val = _vload(vw, p_v2c, t_k)
            var_abs = _vfabs(var_val)
            res     = _vtanh_cephes(_vfmul(var_abs, half_v))
            sign_v, prod_v, nzeros_v = _vspa_in(res, var_val, sign_v, prod_v, nzeros_v)
            _vstore!(vw, p_val, idx, res)
            idx += 1
        end

        idx = 1
        for k in k_start:k_stop
            t_k     = Int(transpose[k])
            val_v   = _vload(vw, p_val, idx)
            var_val = _vload(vw, p_v2c, t_k)
            rt      = _vspa_out_prep(val_v, prod_v, max_tanh_v)
            ath     = _vatanh_cephes(rt)
            res     = _vspa_out_finish(ath, val_v, var_val, sign_v, nzeros_v)
            _vstore!(vw, p_c2v, t_k, res)
            idx += 1
        end
    end
    return nothing
end

# --- Horizontal Layered + OMS (`SimdInterMode`, `Float32`) ---
@inline function _decode_single_ite_layered_inter!(
    rule::RuleOMS{Float32},
    ::Float32,
    chk_offsets::Vector{Int32},
    chk_to_var::Vector{Int32},
    work::LDPC_Workspace{<:SimdInterMode, Float32},
    vw::Val{W}
) where {W}
    n_chk_nodes = length(chk_offsets) - 1
    zero_f32    = _vbroadcast(vw, 0.0f0)
    max_f32     = _vbroadcast(vw, floatmax(Float32))
    off_v       = _vbroadcast(vw, rule.offset)
    zero_i32    = _vbroadcast(vw, Int32(0))
    p_vn        = pointer(work.post_inter)
    p_msg       = pointer(work.msg_chk_to_var_inter)
    p_con       = pointer(work.contrib_inter)

    @inbounds for c in 1:n_chk_nodes
        k_start = Int(chk_offsets[c])
        k_stop  = Int(chk_offsets[c + 1]) - 1

        sign_v = zero_i32
        min1_v = max_f32
        min2_v = max_f32

        idx = 1
        for k in k_start:k_stop
            u       = Int(chk_to_var[k])
            var_val = _vfsub(_vload(vw, p_vn, u), _vload(vw, p_msg, k))
            _vstore!(vw, p_con, idx, var_val)
            var_abs = _vfabs(var_val)
            sign_v  = _vxor_i32(sign_v, _vfsignmask(var_val))
            min2_v  = _vfmin(min2_v, _vfmax(var_abs, min1_v))
            min1_v  = _vfmin(min1_v, var_abs)
            idx    += 1
        end

        cst1_v = _vfmax(zero_f32, _vfsub(min2_v, off_v))
        cst2_v = _vfmax(zero_f32, _vfsub(min1_v, off_v))

        idx = 1
        for k in k_start:k_stop
            u       = Int(chk_to_var[k])
            var_val = _vload(vw, p_con, idx)
            var_abs = _vfabs(var_val)
            sng     = _vxor_i32(sign_v, _vfsignmask(var_val))
            new_msg = _vf32_oms_out(var_abs, min1_v, cst1_v, cst2_v, sng)
            _vstore!(vw, p_msg, k, new_msg)
            _vstore!(vw, p_vn, u, _vfadd(var_val, new_msg))
            idx += 1
        end
    end
    return nothing
end

# --- Horizontal Layered + OMS (`SimdInterMode`, `Int16`) ---
@inline function _decode_single_ite_layered_inter!(
    rule::RuleOMS{Int16},
    sat_val::Int16,
    chk_offsets::Vector{Int32},
    chk_to_var::Vector{Int32},
    work::LDPC_Workspace{<:SimdInterMode, Int16},
    vw::Val{W}
) where {W}
    n_chk_nodes = length(chk_offsets) - 1
    zero_v      = _vbroadcast(vw, Int16(0))
    max_v       = _vbroadcast(vw, typemax(Int16))
    off_v       = _vbroadcast(vw, rule.offset)
    sat_v       = _vbroadcast(vw, sat_val)
    nsat_v      = _vbroadcast(vw, Int16(-sat_val))
    p_vn        = pointer(work.post_inter)
    p_msg       = pointer(work.msg_chk_to_var_inter)
    p_con       = pointer(work.contrib_inter)

    @inbounds for c in 1:n_chk_nodes
        k_start = Int(chk_offsets[c])
        k_stop  = Int(chk_offsets[c + 1]) - 1

        sign_v = zero_v
        min1_v = max_v
        min2_v = max_v

        idx = 1
        for k in k_start:k_stop
            u       = Int(chk_to_var[k])
            var_val = _vi16_sub_sat(_vload(vw, p_vn, u), _vload(vw, p_msg, k))
            _vstore!(vw, p_con, idx, var_val)
            var_abs = _vi16_abs(var_val)
            sign_v  = _vi16_xor(sign_v, _vi16_signmask(var_val))
            min2_v  = _vi16_min(min2_v, _vi16_max(var_abs, min1_v))
            min1_v  = _vi16_min(min1_v, var_abs)
            idx    += 1
        end

        cst1_v = _vi16_max(zero_v, _vi16_sub_sat(min2_v, off_v))
        cst2_v = _vi16_max(zero_v, _vi16_sub_sat(min1_v, off_v))

        idx = 1
        for k in k_start:k_stop
            u       = Int(chk_to_var[k])
            var_val = _vload(vw, p_con, idx)
            var_abs = _vi16_abs(var_val)
            sng     = _vi16_xor(sign_v, _vi16_signmask(var_val))
            new_msg = _vi16_oms_out(var_abs, min1_v, cst1_v, cst2_v, sng, sat_v, nsat_v)
            _vstore!(vw, p_msg, k, new_msg)
            _vstore!(vw, p_vn, u, _vi16_add_sat(var_val, new_msg))
            idx += 1
        end
    end
    return nothing
end

# --- Horizontal Layered + SPA (`SimdInterMode`, `Float32`) ---
@inline function _decode_single_ite_layered_inter!(
    ::RuleSPA,
    ::Float32,
    chk_offsets::Vector{Int32},
    chk_to_var::Vector{Int32},
    work::LDPC_Workspace{<:SimdInterMode, Float32},
    vw::Val{W}
) where {W}
    n_chk_nodes = length(chk_offsets) - 1
    half_v      = _vbroadcast(vw, 0.5f0)
    max_tanh_v  = _vbroadcast(vw, 1.0f0 - eps(Float32))
    one_v       = _vbroadcast(vw, 1.0f0)
    zero_i32    = _vbroadcast(vw, Int32(0))
    p_vn        = pointer(work.post_inter)
    p_msg       = pointer(work.msg_chk_to_var_inter)
    p_val       = pointer(work.values_inter)
    p_con       = pointer(work.contrib_inter)

    @inbounds for c in 1:n_chk_nodes
        k_start = Int(chk_offsets[c])
        k_stop  = Int(chk_offsets[c + 1]) - 1

        sign_v   = zero_i32
        prod_v   = one_v
        nzeros_v = zero_i32

        idx = 1
        for k in k_start:k_stop
            u       = Int(chk_to_var[k])
            var_val = _vfsub(_vload(vw, p_vn, u), _vload(vw, p_msg, k))
            _vstore!(vw, p_con, idx, var_val)
            var_abs = _vfabs(var_val)
            res     = _vtanh_cephes(_vfmul(var_abs, half_v))
            sign_v, prod_v, nzeros_v = _vspa_in(res, var_val, sign_v, prod_v, nzeros_v)
            _vstore!(vw, p_val, idx, res)
            idx += 1
        end

        idx = 1
        for k in k_start:k_stop
            u       = Int(chk_to_var[k])
            val_v   = _vload(vw, p_val, idx)
            var_val = _vload(vw, p_con, idx)
            rt      = _vspa_out_prep(val_v, prod_v, max_tanh_v)
            ath     = _vatanh_cephes(rt)
            new_msg = _vspa_out_finish(ath, val_v, var_val, sign_v, nzeros_v)
            _vstore!(vw, p_msg, k, new_msg)
            _vstore!(vw, p_vn, u, _vfadd(var_val, new_msg))
            idx += 1
        end
    end
    return nothing
end

# --- Soft Syndrome Check (`SimdInterMode`, `Float32` and `Int16`) ---
@inline function _check_syndrome_soft_inter(
    chk_offsets::Vector{Int32},
    chk_to_var::Vector{Int32},
    post_inter::Matrix{Float32},
    vw::Val{W}
)::Bool where {W}
    n_chk_nodes = length(chk_offsets) - 1
    zero_i32    = _vbroadcast(vw, Int32(0))
    p_post      = pointer(post_inter)
    @inbounds for c in 1:n_chk_nodes
        k_start = Int(chk_offsets[c])
        k_stop  = Int(chk_offsets[c + 1]) - 1
        synd_v  = zero_i32
        for k in k_start:k_stop
            u      = Int(chk_to_var[k])
            synd_v = _vxor_i32(synd_v, _vfsignmask(_vload(vw, p_post, u)))
        end
        if _vany_nonzero(synd_v)
            return false
        end
    end
    return true
end

@inline function _check_syndrome_soft_inter(
    chk_offsets::Vector{Int32},
    chk_to_var::Vector{Int32},
    post_inter::Matrix{Int16},
    vw::Val{W}
)::Bool where {W}
    n_chk_nodes = length(chk_offsets) - 1
    zero_i16    = _vbroadcast(vw, Int16(0))
    p_post      = pointer(post_inter)
    @inbounds for c in 1:n_chk_nodes
        k_start = Int(chk_offsets[c])
        k_stop  = Int(chk_offsets[c + 1]) - 1
        synd_v  = zero_i16
        for k in k_start:k_stop
            u      = Int(chk_to_var[k])
            synd_v = _vi16_xor(synd_v, _vi16_signmask(_vload(vw, p_post, u)))
        end
        if _vany_nonzero(synd_v)
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
@inline function _run_flooding_inter!(
    graph::LDPC_Graph{ScheduleFlooding, U, T},
    work::LDPC_Workspace{<:SimdInterMode, T},
    sck_Y_N::AbstractVector{T},
    sck_V_K::AbstractVector{B},
    val_W::Val{W}
)::Int32 where {U<:AbstractUpdateRule, T<:Union{Float32, Int16}, B<:Union{Int32, Int16}, W}
    K           = length(graph.info_bits_pos)
    n_var_nodes = length(graph.var_offsets) - 1
    n_branches  = length(graph.transpose)
    n_ite       = Int(graph.n_ite)
    s_depth     = Int(graph.syndrome_depth)
    zero_T      = zero(T)

    _ensure_inter_capacity!(work, W, n_var_nodes, n_branches, size(work.values_inter, 2))
    _transpose_in!(work.Y_N_inter, sck_Y_N, n_var_nodes, val_W)
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
            val_W
        )
        _decode_single_ite_flooding_inter!(
            graph.rule,
            graph.sat_val,
            graph.chk_offsets,
            graph.transpose,
            work,
            val_W
        )

        if graph.enable_syndrome && ite != n_ite
            _compute_post_inter!(
                graph.var_offsets,
                work.Y_N_inter,
                work.msg_chk_to_var_inter,
                work.post_inter,
                val_W
            )
            syndrome = _check_syndrome_soft_inter(graph.chk_offsets, graph.chk_to_var, work.post_inter, val_W)
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
            val_W
        )
    end

    _extract_hard_decision_inter!(sck_V_K, work.post_inter, graph.info_bits_pos, K, val_W)
    return ifelse(valid_synd, Int32(0), Int32(1))
end

function decode_siho_julia(
    graph::LDPC_Graph{ScheduleFlooding, U, T},
    work::LDPC_Workspace{SimdInterMode{W}, T},
    sck_Y_N::AbstractVector{T},
    sck_V_K::AbstractVector{B},
    frame_id::UInt32 = UInt32(0),
    n_frames_per_wave::UInt32 = UInt32(W)
)::Int32 where {U<:AbstractUpdateRule, W, T<:Union{Float32, Int16}, B<:Union{Int32, Int16}}
    Int(n_frames_per_wave) == W && return _run_flooding_inter!(graph, work, sck_Y_N, sck_V_K, Val(W))
    return _run_flooding_inter!(graph, work, sck_Y_N, sck_V_K, Val(Int(n_frames_per_wave)))
end

# 3.d Inter-Frame SIMD + Horizontal Layered (`ScheduleLayered`, `SimdInterMode`)
@inline function _run_layered_inter!(
    graph::LDPC_Graph{ScheduleLayered, U, T},
    work::LDPC_Workspace{<:SimdInterMode, T},
    sck_Y_N::AbstractVector{T},
    sck_V_K::AbstractVector{B},
    val_W::Val{W}
)::Int32 where {U<:AbstractUpdateRule, T<:Union{Float32, Int16}, B<:Union{Int32, Int16}, W}
    K           = length(graph.info_bits_pos)
    n_var_nodes = length(graph.var_offsets) - 1
    n_branches  = length(graph.transpose)
    n_ite       = Int(graph.n_ite)
    s_depth     = Int(graph.syndrome_depth)
    zero_T      = zero(T)

    _ensure_inter_capacity!(work, W, n_var_nodes, n_branches, size(work.values_inter, 2))
    _transpose_in!(work.post_inter, sck_Y_N, n_var_nodes, val_W)
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
            val_W
        )

        if graph.enable_syndrome
            syndrome = _check_syndrome_soft_inter(graph.chk_offsets, graph.chk_to_var, work.post_inter, val_W)
            cur_syndrome_depth = syndrome ? ((cur_syndrome_depth + 1) % s_depth) : 0
            valid_synd = syndrome && (cur_syndrome_depth == 0)
            if valid_synd
                break
            end
        end
    end

    _extract_hard_decision_inter!(sck_V_K, work.post_inter, graph.info_bits_pos, K, val_W)
    return ifelse(valid_synd, Int32(0), Int32(1))
end

function decode_siho_julia(
    graph::LDPC_Graph{ScheduleLayered, U, T},
    work::LDPC_Workspace{SimdInterMode{W}, T},
    sck_Y_N::AbstractVector{T},
    sck_V_K::AbstractVector{B},
    frame_id::UInt32 = UInt32(0),
    n_frames_per_wave::UInt32 = UInt32(W)
)::Int32 where {U<:AbstractUpdateRule, W, T<:Union{Float32, Int16}, B<:Union{Int32, Int16}}
    Int(n_frames_per_wave) == W && return _run_layered_inter!(graph, work, sck_Y_N, sck_V_K, Val(W))
    return _run_layered_inter!(graph, work, sck_Y_N, sck_V_K, Val(Int(n_frames_per_wave)))
end

# Ahead-of-execution compilation for ScalarMode and SimdInterMode kernels
const _SIMD_W_F32 = _SIMD_REG_BYTES ÷ sizeof(Float32)
const _SIMD_W_I16 = _SIMD_REG_BYTES ÷ sizeof(Int16)
for S in (ScheduleFlooding, ScheduleLayered)
    for U in (RuleSPA, RuleOMS{Float32})
        precompile(decode_siho_julia, (LDPC_Graph{S, U, Float32}, LDPC_Workspace{ScalarMode, Float32}, Vector{Float32}, Vector{Int32}, UInt32, UInt32))
        for W in (_SIMD_W_F32, 2 * _SIMD_W_F32)
            precompile(decode_siho_julia, (LDPC_Graph{S, U, Float32}, LDPC_Workspace{SimdInterMode{W}, Float32}, Vector{Float32}, Vector{Int32}, UInt32, UInt32))
        end
    end
    precompile(decode_siho_julia, (LDPC_Graph{S, RuleOMS{Int16}, Int16}, LDPC_Workspace{ScalarMode, Int16}, Vector{Int16}, Vector{Int16}, UInt32, UInt32))
    for W in (_SIMD_W_I16, 2 * _SIMD_W_I16)
        precompile(decode_siho_julia, (LDPC_Graph{S, RuleOMS{Int16}, Int16}, LDPC_Workspace{SimdInterMode{W}, Int16}, Vector{Int16}, Vector{Int16}, UInt32, UInt32))
    end
end

