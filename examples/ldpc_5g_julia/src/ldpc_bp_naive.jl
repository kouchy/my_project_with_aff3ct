# Generic Modular LDPC Belief Propagation Decoder in Julia for StreamPU (`Stateful_Julia`)
# Tier 1 (`ldpc_bp_naive.jl`): Compact, Textbook Algorithmic Implementation (~160 lines).
#
# Prioritizes maximum readability and closeness to the mathematical equations:
#   - No `@inbounds`, `@simd`, `@generated`, pointer arithmetic, or `Base.llvmcall`.
#   - Directly expresses Flooding (`BP_FLOODING`) and Horizontal Layered (`BP_HORIZONTAL_LAYERED`)
#     schedules with Sum-Product (`SPA`) and Offset Min-Sum (`OMS`) check-node updates.
#   - In inter-frame mode (`INTER`, W frames per wave), simply loops sequentially over each frame `w in 1:W`.

struct LDPC_Graph{T<:Union{Float32, Int16}}
    schedule::Symbol
    rule::Symbol
    offset::T
    sat_val::T
    n_ite::Int
    enable_syndrome::Bool
    syndrome_depth::Int
    var_edges::Vector{UnitRange{Int}}
    chk_edges::Vector{UnitRange{Int}}
    transpose::Vector{Int}
    chk_to_var::Vector{Int}
    info_bits_pos::Vector{Int}
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
    var_degrees = zeros(Int, nv)
    chk_edges   = Vector{UnitRange{Int}}(undef, nc)
    idx, edge   = 1, 1
    for c in 1:nc
        deg = Int(h_cols[idx]); idx += 1
        chk_edges[c] = edge:(edge + deg - 1)
        for _ in 1:deg
            v = Int(h_cols[idx]) + 1; idx += 1
            var_degrees[v] += 1
        end
        edge += deg
    end

    var_edges = Vector{UnitRange{Int}}(undef, nv)
    edge = 1
    for v in 1:nv
        var_edges[v] = edge:(edge + var_degrees[v] - 1)
        edge += var_degrees[v]
    end

    n_branches  = edge - 1
    transpose   = Vector{Int}(undef, n_branches)
    chk_to_var  = Vector{Int}(undef, n_branches)
    connections = zeros(Int, nv)
    idx, k      = 1, 1
    for c in 1:nc
        deg = Int(h_cols[idx]); idx += 1
        for _ in 1:deg
            v = Int(h_cols[idx]) + 1; idx += 1
            transpose[k]  = first(var_edges[v]) + connections[v]
            chk_to_var[k] = v
            connections[v] += 1
            k += 1
        end
    end

    info_bits_pos = [Int(info_bits_raw[i]) + 1 for i in 1:Int(K)]
    sched = (sched_sym === :BP_HORIZONTAL_LAYERED || sched_sym === :BP_HL) ? :LAYERED : :FLOODING
    return LDPC_Graph{T}(
        sched, rule_sym, T(offset_val), T(sat_val_in),
        Int(n_ite), enable_syndrome, Int(syndrome_depth),
        var_edges, chk_edges, transpose, chk_to_var, info_bits_pos
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

mutable struct LDPC_Workspace{T<:Union{Float32, Int16}}
    msg_chk_to_var::Vector{T}
    msg_var_to_chk::Vector{T}
    post::Vector{T}
end

function LDPC_Workspace(
    ::Symbol, ::Type{T}, n_var_nodes::Int32, n_branches::Int32, ::Int32, ::Int32 = Int32(1)
) where {T<:Union{Float32, Int16}}
    return LDPC_Workspace{T}(zeros(T, Int(n_branches)), zeros(T, Int(n_branches)), zeros(T, Int(n_var_nodes)))
end

LDPC_Workspace(nv::Int32, nb::Int32, mc::Int32) = LDPC_Workspace(:SCALAR, Float32, nv, nb, mc)

# Saturating helpers (no-op for Float32, symmetric clamping for Int16)
_widen(x::Float32) = x
_widen(x::Int16)   = Int32(x)
_sat(x::Float32, ::Float32)  = x
_sat(x::Integer, sat::Int16) = Int16(clamp(Int32(x), Int32(-sat), Int32(sat)))
_add(a::Float32, b::Float32, ::Float32 = 0.0f0) = a + b
_add(a::Int16,   b::Int16,   sat::Int16 = typemax(Int16)) = _sat(Int32(a) + Int32(b), sat)
_sub(a::Float32, b::Float32) = a - b
_sub(a::Int16,   b::Int16)   = _sat(Int32(a) - Int32(b), typemax(Int16))

# Check-node extrinsic message calculation for a vector of incoming variable-to-check messages `v2c`
function _check_node_update(v2c::Vector{T}, rule::Symbol, offset::T, sat_val::T) where {T}
    deg = length(v2c)
    c2v = Vector{T}(undef, deg)
    if rule === :SPA
        max_tanh = 1.0f0 - eps(Float32)
        t_vals   = [tanh(abs(Float32(x)) * 0.5f0) for x in v2c]
        for i in 1:deg
            prod_ext = 1.0f0
            sgn_ext  = false
            for j in 1:deg
                if j != i
                    prod_ext *= t_vals[j]
                    sgn_ext  ⊻= signbit(v2c[j])
                end
            end
            mag = 2.0f0 * atanh(min(prod_ext, max_tanh))
            c2v[i] = T(sgn_ext ? -mag : mag)
        end
    else # Offset Min-Sum (OMS)
        mags    = abs.(v2c)
        tot_sgn = reduce(⊻, (x < zero(T) for x in v2c); init = false)
        for i in 1:deg
            min_ext = minimum(mags[j] for j in 1:deg if j != i)
            mag     = max(zero(T), T(min_ext - offset))
            sgn     = tot_sgn ⊻ (v2c[i] < zero(T))
            c2v[i]  = _sat(sgn ? -mag : mag, sat_val)
        end
    end
    return c2v
end

function _check_syndrome(graph::LDPC_Graph{T}, post::Vector{T}) where {T}
    for edges in graph.chk_edges
        parity = reduce(⊻, (post[graph.chk_to_var[k]] < zero(T) for k in edges); init = false)
        parity && return false
    end
    return true
end

function _decode_single_frame!(
    graph::LDPC_Graph{T}, work::LDPC_Workspace{T},
    Y_N::AbstractVector{T}, V_K::AbstractVector{B}
)::Bool where {T<:Union{Float32, Int16}, B<:Union{Int32, Int16}}
    c2v, v2c, post = work.msg_chk_to_var, work.msg_var_to_chk, work.post
    fill!(c2v, zero(T))
    post .= Y_N
    cur_depth, valid = 0, true

    for ite in 1:graph.n_ite
        if graph.schedule === :FLOODING
            # 1. Variable-node update
            for v in eachindex(graph.var_edges)
                edges = graph.var_edges[v]
                tot   = _widen(Y_N[v]) + sum(_widen(c2v[e]) for e in edges)
                post[v] = _sat(tot, typemax(T))
                for e in edges
                    v2c[e] = _sat(tot - _widen(c2v[e]), typemax(T))
                end
            end
            # 2. Check-node update
            for edges in graph.chk_edges
                incoming = [v2c[graph.transpose[k]] for k in edges]
                outgoing = _check_node_update(incoming, graph.rule, graph.offset, graph.sat_val)
                for (idx, k) in enumerate(edges)
                    c2v[graph.transpose[k]] = outgoing[idx]
                end
            end
            # Recompute a-posteriori LLRs for syndrome or final iteration
            if (graph.enable_syndrome && ite != graph.n_ite) || ite == graph.n_ite
                for v in eachindex(graph.var_edges)
                    tot = _widen(Y_N[v]) + sum(_widen(c2v[e]) for e in graph.var_edges[v])
                    post[v] = _sat(tot, typemax(T))
                end
            end
        else # Horizontal Layered schedule
            for edges in graph.chk_edges
                incoming = [_sub(post[graph.chk_to_var[k]], c2v[k]) for k in edges]
                outgoing = _check_node_update(incoming, graph.rule, graph.offset, graph.sat_val)
                for (idx, k) in enumerate(edges)
                    u       = graph.chk_to_var[k]
                    c2v[k]  = outgoing[idx]
                    post[u] = _add(incoming[idx], outgoing[idx], graph.sat_val)
                end
            end
        end

        if graph.enable_syndrome && (graph.schedule === :LAYERED || ite != graph.n_ite)
            synd      = _check_syndrome(graph, post)
            cur_depth = synd ? ((cur_depth + 1) % graph.syndrome_depth) : 0
            valid     = synd && (cur_depth == 0)
            valid && break
        end
    end

    for (i, pos) in enumerate(graph.info_bits_pos)
        V_K[i] = post[pos] < zero(T) ? one(B) : zero(B)
    end
    return valid
end

function decode_siho_julia(
    graph::LDPC_Graph{T}, work::LDPC_Workspace{T},
    sck_Y_N::AbstractVector{T}, sck_V_K::AbstractVector{B},
    ::UInt32 = UInt32(0), n_frames_per_wave::UInt32 = UInt32(1)
)::Int32 where {T<:Union{Float32, Int16}, B<:Union{Int32, Int16}}
    N = length(graph.var_edges)
    K = length(graph.info_bits_pos)
    W = Int(n_frames_per_wave)
    all_valid = true
    for w in 1:W
        y_frame = @view sck_Y_N[((w - 1) * N + 1):(w * N)]
        v_frame = @view sck_V_K[((w - 1) * K + 1):(w * K)]
        all_valid &= _decode_single_frame!(graph, work, y_frame, v_frame)
    end
    return all_valid ? Int32(0) : Int32(1)
end

# Ahead-of-execution compilation for Float32 and Int16 entry points
precompile(decode_siho_julia, (LDPC_Graph{Float32}, LDPC_Workspace{Float32}, Vector{Float32}, Vector{Int32}, UInt32, UInt32))
precompile(decode_siho_julia, (LDPC_Graph{Int16},   LDPC_Workspace{Int16},   Vector{Int16},   Vector{Int16}, UInt32, UInt32))

