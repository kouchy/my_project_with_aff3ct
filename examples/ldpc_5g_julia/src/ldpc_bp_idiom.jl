# LDPC Belief Propagation decoder in Julia for StreamPU (`Stateful_Julia`)
# Tier "IDIOM" (`ldpc_bp_idiom.jl`): idiomatic high-level Julia relying on LLVM auto-vectorization.
#
# Horizontal layered schedule + Offset Min-Sum, fixed number of iterations (no syndrome), Float32 or
# Int16 (fixed-point, saturating arithmetic). Bit-exact with `ldpc_bp_opt.jl`.
#
# Inter-frame mode: the W frames of a wave are processed together, one frame per "lane". All the
# per-variable / per-branch buffers are `Matrix{T}` of size (W, n): lanes are innermost (column `v`
# holds the W frames), so that every inner `for w in 1:W` loop is a contiguous, independent loop that
# LLVM can vectorize. No `llvmcall`, no `SIMD.jl`, no `VecElement`, no `@generated` code.
# `@simd ivdep` states that the lanes are independent (each iteration `w` only touches element `w` of
# every buffer), which removes LLVM's runtime alias checks. W is a type parameter of the workspace
# (compile-time constant trip count). Scalar mode is the same code with W = 1.

# LOC80-BEGIN JL_IDIOM_INTER.module
struct LDPC_Graph{T<:Union{Float32, Int16}}
    offset::T
    sat_val::T
    n_ite::Int
    chk_offsets::Vector{Int32}   # CSR over check nodes (1-based)
    chk_to_var::Vector{Int32}    # variable node of each branch (1-based)
    info_bits_pos::Vector{Int32} # position of the K information bits in the codeword (1-based)
end

function LDPC_Graph(sched::Symbol, rule::Symbol, ::Type{T}, offset::Real, sat_val::Real,
                    n_ite::Int32, enable_syndrome::Bool, syndrome_depth::Int32,
                    n_var_nodes::Int32, n_chk_nodes::Int32,
                    h_cols_addr::UInt, h_cols_len::Int32,
                    info_bits_addr::UInt, K::Int32) where {T<:Union{Float32, Int16}}
    if !(sched in (:BP_HORIZONTAL_LAYERED, :BP_HL)) || rule !== :OMS || enable_syndrome
        error("ldpc_bp_idiom.jl only supports BP_HORIZONTAL_LAYERED + OMS without syndrome (--dec-no-synd)")
    end
    # `h_cols` = for each check node: its degree followed by its (0-based) variable nodes
    h_cols = unsafe_wrap(Array, Ptr{UInt32}(h_cols_addr), Int(h_cols_len))
    chk_offsets, chk_to_var = Int32[1], Int32[]
    i = 1
    for _ in 1:n_chk_nodes
        deg = Int(h_cols[i])
        append!(chk_to_var, h_cols[i+1:i+deg] .+ 1)
        push!(chk_offsets, length(chk_to_var) + 1)
        i += deg + 1
    end
    info_bits = unsafe_wrap(Array, Ptr{UInt32}(info_bits_addr), Int(K))
    return LDPC_Graph{T}(T(offset), T(sat_val), Int(n_ite), chk_offsets, chk_to_var, Int32.(info_bits) .+ Int32(1))
end

# Per-thread buffers, W = number of frames processed together (1 in scalar mode)
struct LDPC_Workspace{T<:Union{Float32, Int16}, W}
    post::Matrix{T}     # (W, n_var_nodes)    a posteriori LLRs
    msg::Matrix{T}      # (W, n_branches)     check-to-variable messages
    contrib::Matrix{T}  # (W, max_chk_degree) variable-to-check messages of the current check
    sign::Vector{Bool}  # (W) per-check temporaries
    min1::Vector{T}
    min2::Vector{T}
    cst1::Vector{T}
    cst2::Vector{T}
end

function LDPC_Workspace(mode::Symbol, ::Type{T}, n_var_nodes::Int32, n_branches::Int32,
                        max_chk_degree::Int32, n_frames_per_wave::Int32 = Int32(1)) where {T<:Union{Float32, Int16}}
    W = mode === :INTER ? Int(n_frames_per_wave) : 1
    return LDPC_Workspace{T, W}(zeros(T, W, n_var_nodes), zeros(T, W, n_branches), zeros(T, W, max_chk_degree),
                                zeros(Bool, W), zeros(T, W), zeros(T, W), zeros(T, W), zeros(T, W))
end
# LOC80-END JL_IDIOM_INTER.module

# LOC80-BEGIN JL_IDIOM_INTER
# Int16: saturating add/sub (Julia integer arithmetic wraps), messages clamped to ±sat_val.
# Float32: plain arithmetic, no clamping.
sat_add(a::Int16, b::Int16) = clamp(Int32(a) + Int32(b), typemin(Int16), typemax(Int16)) % Int16
sat_sub(a::Int16, b::Int16) = clamp(Int32(a) - Int32(b), typemin(Int16), typemax(Int16)) % Int16
sat_add(a::Float32, b::Float32) = a + b
sat_sub(a::Float32, b::Float32) = a - b
clamp_msg(x::Int16, sat_val::Int16) = clamp(x, -sat_val, sat_val)
clamp_msg(x::Float32, ::Float32) = x
largest(::Type{T}) where {T} = T <: Integer ? typemax(T) : floatmax(T)

# One iteration of horizontal layered Offset Min-Sum on the W frames
function decode_layered_ite!(work::LDPC_Workspace{T, W}, graph::LDPC_Graph{T}) where {T, W}
    (; post, msg, contrib, sign, min1, min2, cst1, cst2) = work
    for c in 1:length(graph.chk_offsets)-1
        branches = graph.chk_offsets[c]:graph.chk_offsets[c+1]-1
        fill!(sign, false)
        fill!(min1, largest(T))
        fill!(min2, largest(T))
        # 1. variable-to-check messages, sign product and the two smallest magnitudes
        for (i, k) in enumerate(branches)
            u = graph.chk_to_var[k]
            @inbounds @simd ivdep for w in 1:W
                v = sat_sub(post[w, u], msg[w, k])
                contrib[w, i] = v
                sign[w] ⊻= v < zero(T)
                min2[w] = min(min2[w], max(abs(v), min1[w]))
                min1[w] = min(min1[w], abs(v))
            end
        end
        @inbounds @simd ivdep for w in 1:W
            cst1[w] = max(zero(T), sat_sub(min2[w], graph.offset))
            cst2[w] = max(zero(T), sat_sub(min1[w], graph.offset))
        end
        # 2. new check-to-variable messages and posterior update
        for (i, k) in enumerate(branches)
            u = graph.chk_to_var[k]
            @inbounds @simd ivdep for w in 1:W
                v = contrib[w, i]
                mag = ifelse(abs(v) == min1[w], cst1[w], cst2[w])
                m = clamp_msg(ifelse(sign[w] ⊻ (v < zero(T)), -mag, mag), graph.sat_val)
                msg[w, k] = m
                post[w, u] = sat_add(v, m)
            end
        end
    end
end

function decode_siho_julia(graph::LDPC_Graph{T}, work::LDPC_Workspace{T, W},
                           sck_Y_N::AbstractVector{T}, sck_V_K::AbstractVector,
                           frame_id::UInt32, n_frames_per_wave::UInt32)::Int32 where {T, W}
    n_var_nodes = size(work.post, 2)
    K = length(graph.info_bits_pos)
    # frames -> lanes layout
    @inbounds for w in 1:W, v in 1:n_var_nodes
        work.post[w, v] = sck_Y_N[(w-1)*n_var_nodes + v]
    end
    fill!(work.msg, zero(T))
    for _ in 1:graph.n_ite
        decode_layered_ite!(work, graph)
    end
    # hard decision, lanes -> frames layout
    @inbounds for w in 1:W, i in 1:K
        sck_V_K[(w-1)*K + i] = work.post[w, graph.info_bits_pos[i]] < zero(T)
    end
    return Int32(0)
end
# LOC80-END JL_IDIOM_INTER

# Ahead-of-execution compilation (scalar W = 1 and the usual native SIMD lane counts)
for W in (1, 4, 8, 16, 32)
    precompile(decode_siho_julia, (LDPC_Graph{Float32}, LDPC_Workspace{Float32, W}, Vector{Float32}, Vector{Int32}, UInt32, UInt32))
    precompile(decode_siho_julia, (LDPC_Graph{Int16}, LDPC_Workspace{Int16, W}, Vector{Int16}, Vector{Int16}, UInt32, UInt32))
end
