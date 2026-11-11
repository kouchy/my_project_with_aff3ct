// AFF3CT / StreamPU module wrapping the specialized LDPC kernels of `ldpc_spec_kernels.hpp`.
// Exposes the standard `decode_siho` task (sockets `Y_N` -> `V_K`) so that it is a drop-in replacement for
// the AFF3CT, Julia and Python decoders in `pipeline_ldpc_5g.py`.
//   - scalar mode : 1 frame per wave;
//   - inter mode  : W frames per wave (W = SIMD lanes for R), one frame per lane.

#ifndef DECODER_LDPC_SPEC_HPP_
#define DECODER_LDPC_SPEC_HPP_

#include <memory>
#include <string>
#include <vector>

#include <aff3ct.hpp>

#include "ldpc_spec_kernels.hpp"

// LOC80-BEGIN CPP_SPEC.module
template<typename B, typename R>
class Decoder_LDPC_spec : public aff3ct::module::Decoder_SIHO<B, R>
{
  private:
    std::shared_ptr<const ldpc_spec::Graph> graph;
    const int n_ite;
    const R offset;
    const R saturation;
    const int W;
    ldpc_spec::avector<R> vn, msg, con;
    std::vector<int8_t> cwd_dummy;

  public:
    Decoder_LDPC_spec(const int K,
                      const int N,
                      const int n_ite,
                      const aff3ct::tools::Sparse_matrix& H,
                      const std::vector<uint32_t>& info_bits_pos,
                      const R offset,
                      const R saturation,
                      const bool inter)
      : aff3ct::module::Decoder_SIHO<B, R>(K, N)
      , graph(std::make_shared<ldpc_spec::Graph>((int)H.get_n_rows(), H.get_col_to_rows(), info_bits_pos))
      , n_ite(n_ite)
      , offset(offset)
      , saturation(saturation)
      , W(inter ? ldpc_spec::simd_width<R>() : 1)
    {
        if (inter && !ldpc_spec::simd_available)
            throw spu::tools::runtime_error(__FILE__, __LINE__, __func__, "No SIMD ISA available for 'INTER'.");
        this->set_name(inter ? "Decoder_LDPC_spec_inter" : "Decoder_LDPC_spec");
        if (this->W > 1) this->set_n_frames_per_wave(this->W);
        for (auto& t : this->tasks)
            t->set_replicability(true);
        const auto n_edges = (size_t)this->graph->chk_off.back();
        this->vn.resize((size_t)N * this->W);
        this->msg.resize(n_edges * this->W);
        this->con.resize((size_t)this->graph->max_chk_deg * this->W);
        this->cwd_dummy.resize(this->W);
    }

    virtual ~Decoder_LDPC_spec() = default;

    Decoder_LDPC_spec* clone() const override
    {
        auto m = new Decoder_LDPC_spec(*this);
        m->deep_copy(*this);
        return m;
    }

  protected:
    int _decode_siho(const R* Y_N, int8_t* CWD, B* V_K, const size_t /*frame_id*/) override
    {
        const auto& g = *this->graph;
        // LOC80-END CPP_SPEC.module
        // LOC80-BEGIN CPP_SPEC_SCALAR.module
        if (this->W == 1)
        {
            ldpc_spec::decode_scalar<R>(
              g, Y_N, this->vn.data(), this->msg.data(), this->con.data(), this->n_ite, this->offset, this->saturation);
            ldpc_spec::hard_decision_scalar<R, B>(g, this->vn.data(), V_K);
        }
        // LOC80-END CPP_SPEC_SCALAR.module
        // LOC80-BEGIN CPP_SPEC_INTER.module
        else
        {
#if !defined(LDPC_SPEC_NO_SIMD)
            ldpc_spec::transpose_in<R>(Y_N, this->vn.data(), g.n_var, this->W);
            ldpc_spec::decode_inter<R>(
              g, this->vn.data(), this->msg.data(), this->con.data(), this->n_ite, this->offset, this->saturation);
            ldpc_spec::hard_decision_inter<R, B>(g, this->vn.data(), V_K, this->W);
#endif
        }
        // LOC80-END CPP_SPEC_INTER.module
        // LOC80-BEGIN CPP_SPEC.module
        for (int f = 0; f < this->W; f++)
            CWD[f] = 0; // no syndrome check (fixed number of iterations)
        return 0;
    }

    int _decode_siho(const R* Y_N, B* V_K, const size_t frame_id) override
    {
        return this->_decode_siho(Y_N, this->cwd_dummy.data(), V_K, frame_id);
    }
};
// LOC80-END CPP_SPEC.module

inline std::string
ldpc_spec_isa_name()
{
#if defined(LDPC_SPEC_AVX512)
    return "AVX-512";
#elif defined(LDPC_SPEC_AVX2)
    return "AVX2";
#elif defined(LDPC_SPEC_SSE41)
    return "SSE4.1";
#elif defined(LDPC_SPEC_NEON)
    return "NEON";
#else
    return "none";
#endif
}

#endif // DECODER_LDPC_SPEC_HPP_

