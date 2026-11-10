#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <functional>
#include <iostream>
#include <memory>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <aff3ct.hpp>
#include <streampu.hpp>

namespace py = pybind11;
using namespace py::literals;

namespace
{

// Constant parameter buffer for the channel/modem 'CP' (sigma) input sockets.
// Binds a raw C++ buffer directly to an input socket (just like `(*channel)["add_noise::CP"] = sigma` in C++),
// avoiding any extra upstream task in a multi-stage Pipeline.
class Sigma_buffer
{
  private:
    std::vector<float> data_;

  public:
    Sigma_buffer(const size_t n_frames, const float sigma = 1.0f)
      : data_(n_frames, sigma)
    {
    }

    void set_sigma(const float sigma)
    {
        std::fill(this->data_.begin(), this->data_.end(), sigma);
    }

    float get_sigma() const
    {
        return this->data_.empty() ? 0.0f : this->data_[0];
    }

    void bind(spu::runtime::Socket& sck_in)
    {
        sck_in.bind(this->data_);
    }
};

// Helper encapsulating an LDPC Sparse_matrix (from a .qc file or 5G NR standard)
// together with its information bit positions, puncturing pattern, CSR/CSC Tanner graph
// arrays (for Python/Numba and Julia decoders), and 32-bit / 16-bit AFF3CT module builders.
class LDPC_Code
{
  private:
    std::string enc_type_ = "LDPC_5G";
    int K_ = 0;
    int N_cw_ = 0;
    int N_ = 0;
    std::string qc_path_;
    std::string g_path_;
    int zc_ = 0;
    int k_ldpc_ = 0;
    aff3ct::tools::Sparse_matrix H_;
    std::vector<uint32_t> info_bits_pos_;
    std::vector<bool> pct_pattern_;
    std::vector<uint32_t> h_cols_;
    int16_t sat_val_i16_ = 0;

    // 0-based CSR/CSC Tanner graph arrays for Python/Numba decoders
    std::vector<int32_t> var_offsets_;
    std::vector<int32_t> chk_offsets_;
    std::vector<int32_t> transpose_;
    std::vector<int32_t> chk_to_var_;
    std::vector<int32_t> info_bits_pos_i32_;

    static std::string resolve_conf_file(const std::string& rel_path)
    {
        const std::vector<std::string> prefixes = {
            "",
            "lib/aff3ct/",
            "../lib/aff3ct/",
            "../../lib/aff3ct/",
            "../../../lib/aff3ct/",
            "../../../../lib/aff3ct/"
        };
        for (const auto& p : prefixes)
        {
            std::ifstream f(p + rel_path);
            if (f.good()) return p + rel_path;
        }
        return cli::modify_path<cli::Is_file>(rel_path);
    }

    void build_csr_arrays()
    {
        const int n_var = (int)this->H_.get_n_rows();
        const int n_chk = (int)this->H_.get_n_cols();

        std::vector<int32_t> var_degrees(n_var, 0);
        this->chk_offsets_.resize(n_chk + 1);
        this->chk_offsets_[0] = 0;

        for (int c = 0; c < n_chk; c++)
        {
            const auto& col = this->H_.get_col_to_rows()[c];
            const int32_t deg = (int32_t)col.size();
            this->chk_offsets_[c + 1] = this->chk_offsets_[c] + deg;
            for (int32_t idx = 0; idx < deg; idx++)
            {
                const int v = (int)col[idx];
                var_degrees[v]++;
            }
        }

        this->var_offsets_.resize(n_var + 1);
        this->var_offsets_[0] = 0;
        for (int v = 0; v < n_var; v++)
            this->var_offsets_[v + 1] = this->var_offsets_[v] + var_degrees[v];

        const int n_branches = (int)this->chk_offsets_.back();
        this->transpose_.resize(n_branches);
        this->chk_to_var_.resize(n_branches);
        std::vector<int32_t> connections(n_var, 0);

        int k = 0;
        for (int c = 0; c < n_chk; c++)
        {
            const auto& col = this->H_.get_col_to_rows()[c];
            for (size_t idx = 0; idx < col.size(); idx++)
            {
                const int v = (int)col[idx];
                const int32_t branch_id = this->var_offsets_[v] + connections[v];
                connections[v]++;
                this->transpose_[k] = branch_id;
                this->chk_to_var_[k] = (int32_t)v;
                k++;
            }
        }

        this->info_bits_pos_i32_.resize(this->K_);
        for (int i = 0; i < this->K_; i++)
            this->info_bits_pos_i32_[i] = (int32_t)this->info_bits_pos_[i];
    }

  public:
    LDPC_Code(const int K = 512,
              const int N = 1024,
              const std::string& enc_type = "LDPC_5G",
              const std::string& qc_path = "",
              const std::string& h_reorder = "NONE")
      : enc_type_(enc_type)
      , K_(K)
      , N_(N)
      , qc_path_(qc_path)
    {
        if (this->qc_path_.empty() && this->enc_type_ == "LDPC_5G")
        {
            auto bg = aff3ct::tools::build_5G_base_graph(this->K_, this->N_);
            this->zc_ = bg.Zc;
            this->k_ldpc_ = bg.K_LDPC;
            const std::string h_rel = "conf/dec/LDPC/5G/NR_" + std::to_string(bg.Bg) + "_" +
                                      std::to_string(bg.index_list) + "_" + std::to_string(bg.Zc) + ".qc";
            const std::string g_rel = "conf/enc/LDPC/5G/NR_" + std::to_string(bg.Bg) + "_" +
                                      std::to_string(bg.index_list) + "_" + std::to_string(bg.Zc) + ".txt";
            this->qc_path_ = resolve_conf_file(h_rel);
            this->g_path_ = resolve_conf_file(g_rel);
        }

        this->H_ = aff3ct::tools::LDPC_matrix_handler::read(this->qc_path_, &this->info_bits_pos_, &this->pct_pattern_);
        if (h_reorder != "NONE")
        {
            this->H_.sort_cols_per_density(h_reorder == "ASC"
                                             ? aff3ct::tools::Matrix::Sort::ASCENDING
                                             : aff3ct::tools::Matrix::Sort::DESCENDING);
        }

        this->N_cw_ = (int)this->H_.get_n_rows();
        const int M = (int)this->H_.get_n_cols();
        if (this->K_ <= 0) this->K_ = this->N_cw_ - M;
        if (this->N_ <= 0) this->N_ = this->N_cw_;

        if (this->info_bits_pos_.empty())
        {
            this->info_bits_pos_.resize(this->K_);
            std::iota(this->info_bits_pos_.begin(), this->info_bits_pos_.end(), 0);
        }
        else
        {
            aff3ct::tools::LDPC_matrix_handler::check_info_pos(this->info_bits_pos_, this->K_, this->N_cw_);
        }

        for (const auto& col : this->H_.get_col_to_rows())
        {
            this->h_cols_.push_back((uint32_t)col.size());
            this->h_cols_.insert(this->h_cols_.end(), col.begin(), col.end());
        }

        const int max_var_deg = std::max<int>(1, (int)this->H_.get_rows_max_degree());
        this->sat_val_i16_ = (int16_t)((1 << ((sizeof(int16_t) * 8 - 2) - (int)std::log2(max_var_deg))) - 1);

        this->build_csr_arrays();
    }

    int get_K() const { return this->K_; }
    int get_N_cw() const { return this->N_cw_; }
    int get_N() const { return this->N_; }
    int get_n_rows() const { return (int)this->H_.get_n_rows(); }
    int get_n_cols() const { return (int)this->H_.get_n_cols(); }
    int get_n_connections() const { return (int)this->H_.get_n_connections(); }
    int get_cols_max_degree() const { return (int)this->H_.get_cols_max_degree(); }
    int get_rows_max_degree() const { return (int)this->H_.get_rows_max_degree(); }
    int16_t get_sat_val_i16() const { return this->sat_val_i16_; }
    const std::string& get_qc_path() const { return this->qc_path_; }

    const aff3ct::tools::Sparse_matrix& get_H() const { return this->H_; }
    const std::vector<uint32_t>& get_info_bits_pos() const { return this->info_bits_pos_; }
    const std::vector<bool>& get_pct_pattern() const { return this->pct_pattern_; }

    py::tuple get_tanner_graph_arrays() const
    {
        auto to_np = [](const std::vector<int32_t>& vec)
        {
            py::array_t<int32_t> arr(vec.size());
            std::copy(vec.begin(), vec.end(), arr.mutable_data());
            return arr;
        };
        return py::make_tuple(to_np(this->var_offsets_),
                              to_np(this->chk_offsets_),
                              to_np(this->transpose_),
                              to_np(this->chk_to_var_),
                              to_np(this->info_bits_pos_i32_));
    }

    std::string get_julia_graph_ctor(const int n_ite = 20,
                                     const bool enable_syndrome = true,
                                     const int syndrome_depth = 1,
                                     const std::string& dec_type = "BP_FLOODING",
                                     const std::string& dec_implem = "SPA",
                                     const std::string& q_type = "f32",
                                     const float offset = 0.0f) const
    {
        const bool is_i16 = (q_type == "i16" || q_type == "int16" || q_type == "Int16");
        std::stringstream ss;
        ss << "LDPC_Graph("
           << ":" << dec_type << ", "
           << ":" << dec_implem << ", "
           << (is_i16 ? "Int16" : "Float32") << ", ";
        if (is_i16)
            ss << "Int16(" << (int16_t)offset << "), Int16(" << this->sat_val_i16_ << "), ";
        else
            ss << "Float32(" << offset << "), Float32(0.0), ";
        ss << "Int32(" << n_ite << "), "
           << (enable_syndrome ? "true" : "false") << ", "
           << "Int32(" << syndrome_depth << "), "
           << "Int32(" << this->H_.get_n_rows() << "), "
           << "Int32(" << this->H_.get_n_cols() << "), "
           << "UInt(" << reinterpret_cast<uintptr_t>(this->h_cols_.data()) << "), Int32(" << this->h_cols_.size() << "), "
           << "UInt(" << reinterpret_cast<uintptr_t>(this->info_bits_pos_.data()) << "), Int32(" << this->K_ << ")"
           << ")";
        return ss.str();
    }

    std::string get_julia_workspace_ctor(const std::string& simd_strategy = "",
                                         const std::string& q_type = "f32",
                                         const int n_frames_per_wave = 1) const
    {
        const bool is_i16 = (q_type == "i16" || q_type == "int16" || q_type == "Int16");
        const std::string mode_sym = (simd_strategy == "INTER") ? ":INTER" : ":SCALAR";
        std::stringstream ss;
        ss << "LDPC_Workspace("
           << mode_sym << ", "
           << (is_i16 ? "Int16" : "Float32") << ", "
           << "Int32(" << this->H_.get_n_rows() << "), "
           << "Int32(" << this->H_.get_n_connections() << "), "
           << "Int32(" << this->H_.get_cols_max_degree() << "), "
           << "Int32(" << std::max(1, n_frames_per_wave) << ")"
           << ")";
        return ss.str();
    }

    // --- 32-bit (B = int, Q = float) builders ---
    aff3ct::module::Encoder_LDPC<int>* build_encoder() const
    {
        if (this->enc_type_ == "LDPC_5G")
        {
            return new aff3ct::module::Encoder_LDPC_QC_fast<int>(
              this->K_, this->N_cw_, this->zc_, this->g_path_.c_str(), this->k_ldpc_);
        }
        return new aff3ct::module::Encoder_LDPC_from_QC<int>(this->K_, this->N_cw_, this->H_);
    }

    aff3ct::module::Puncturer<int, float>* build_puncturer() const
    {
        if (this->N_ != this->N_cw_)
        {
            if (this->enc_type_ == "LDPC_5G")
                return new aff3ct::module::Puncturer_5G<int, float>(this->K_, this->N_, this->N_cw_, this->pct_pattern_);
            if (!this->pct_pattern_.empty())
                return new aff3ct::module::Puncturer_LDPC<int, float>(this->K_, this->N_, this->N_cw_, this->pct_pattern_);
        }
        return new aff3ct::module::Puncturer_NO<int, float>(this->K_, this->N_cw_);
    }

    aff3ct::module::Decoder_SIHO<int, float>* build_decoder(const std::string& type = "BP_FLOODING",
                                                            const std::string& implem = "SPA",
                                                            const int n_ite = 20,
                                                            const std::string& simd_strategy = "",
                                                            const float offset = 0.0f,
                                                            const float norm_factor = 1.0f,
                                                            const bool enable_syndrome = true,
                                                            const int syndrome_depth = 1) const
    {
        aff3ct::factory::Decoder_LDPC dec_params;
        dec_params.K = this->K_;
        dec_params.N_cw = this->N_cw_;
        dec_params.type = type;
        dec_params.implem = implem;
        dec_params.n_ite = n_ite;
        dec_params.simd_strategy = simd_strategy;
        dec_params.offset = offset;
        dec_params.norm_factor = norm_factor;
        dec_params.enable_syndrome = enable_syndrome;
        dec_params.syndrome_depth = syndrome_depth;
        return dec_params.build<int, float>(this->H_, this->info_bits_pos_);
    }

    // --- 16-bit (B = int16_t, Q = int16_t) builders ---
    aff3ct::module::Encoder_LDPC<int16_t>* build_encoder_i16() const
    {
        if (this->enc_type_ == "LDPC_5G")
        {
            return new aff3ct::module::Encoder_LDPC_QC_fast<int16_t>(
              this->K_, this->N_cw_, this->zc_, this->g_path_.c_str(), this->k_ldpc_);
        }
        return new aff3ct::module::Encoder_LDPC_from_QC<int16_t>(this->K_, this->N_cw_, this->H_);
    }

    aff3ct::module::Puncturer<int16_t, int16_t>* build_puncturer_i16() const
    {
        if (this->N_ != this->N_cw_)
        {
            if (this->enc_type_ == "LDPC_5G")
                return new aff3ct::module::Puncturer_5G<int16_t, int16_t>(this->K_, this->N_, this->N_cw_, this->pct_pattern_);
            if (!this->pct_pattern_.empty())
                return new aff3ct::module::Puncturer_LDPC<int16_t, int16_t>(this->K_, this->N_, this->N_cw_, this->pct_pattern_);
        }
        return new aff3ct::module::Puncturer_NO<int16_t, int16_t>(this->K_, this->N_cw_);
    }

    aff3ct::module::Quantizer_pow2_fast<float, int16_t>* build_quantizer_i16(const short n_decimals = 2,
                                                                             const short n_bits = 8) const
    {
        return new aff3ct::module::Quantizer_pow2_fast<float, int16_t>(this->N_, n_decimals, n_bits);
    }

    aff3ct::module::Decoder_SIHO<int16_t, int16_t>* build_decoder_i16(const std::string& type = "BP_FLOODING",
                                                                      const std::string& implem = "OMS",
                                                                      const int n_ite = 20,
                                                                      const std::string& simd_strategy = "",
                                                                      const float offset = 1.0f,
                                                                      const float norm_factor = 1.0f,
                                                                      const bool enable_syndrome = true,
                                                                      const int syndrome_depth = 1) const
    {
        if (simd_strategy.empty() && implem == "OMS")
        {
            struct Update_rule_OMS_sat_i16 : public aff3ct::tools::Update_rule_OMS<int16_t>
            {
                const int16_t sat_val;
                Update_rule_OMS_sat_i16(const int16_t offset, const int16_t sat_val)
                  : aff3ct::tools::Update_rule_OMS<int16_t>(offset)
                  , sat_val(sat_val)
                {
                }
                inline int16_t compute_chk_node_out(const int var_id, const int16_t var_val)
                {
                    const int16_t val = aff3ct::tools::Update_rule_OMS<int16_t>::compute_chk_node_out(var_id, var_val);
                    return std::min<int16_t>(this->sat_val, std::max<int16_t>((int16_t)-this->sat_val, val));
                }
            };
            struct Decoder_flooding_sat_i16
              : public aff3ct::module::Decoder_LDPC_BP_flooding<int16_t, int16_t, Update_rule_OMS_sat_i16>
            {
                using Base = aff3ct::module::Decoder_LDPC_BP_flooding<int16_t, int16_t, Update_rule_OMS_sat_i16>;
                using Base::Base;
                static inline int16_t clamp_i16(int32_t x)
                {
                    return (int16_t)std::min<int32_t>(32767, std::max<int32_t>(-32768, x));
                }
                Decoder_flooding_sat_i16* clone() const override
                {
                    auto m = new Decoder_flooding_sat_i16(*this);
                    m->deep_copy(*this);
                    return m;
                }
                int _decode_siho(const int16_t* Y_N, int8_t* CWD, int16_t* V_K, const size_t frame_id) override
                {
                    this->up_rule.begin_decoding(this->n_ite);
                    auto& msg_c2v = this->msg_chk_to_var[frame_id];
                    auto& msg_v2c = this->msg_var_to_chk[frame_id];
                    const int n_var = (int)this->H.get_n_rows();
                    bool valid_synd = true;
                    int ite = 0;
                    for (; ite < this->n_ite; ite++)
                    {
                        this->up_rule.begin_ite(ite);
                        const int16_t* c2v_ptr = msg_c2v.data();
                        int16_t* v2c_ptr = msg_v2c.data();
                        for (int v = 0; v < n_var; v++)
                        {
                            const int deg = (int)this->H.get_row_to_cols()[v].size();
                            int16_t acc = 0;
                            for (int c = 0; c < deg; c++)
                                acc = clamp_i16((int32_t)acc + (int32_t)c2v_ptr[c]);
                            const int16_t tmp = clamp_i16((int32_t)Y_N[v] + (int32_t)acc);
                            for (int c = 0; c < deg; c++)
                                v2c_ptr[c] = clamp_i16((int32_t)tmp - (int32_t)c2v_ptr[c]);
                            c2v_ptr += deg;
                            v2c_ptr += deg;
                        }
                        this->_decode_single_ite(msg_v2c, msg_c2v);
                        this->up_rule.end_ite();
                        if (this->enable_syndrome && ite != this->n_ite - 1)
                        {
                            c2v_ptr = msg_c2v.data();
                            for (int v = 0; v < n_var; v++)
                            {
                                const int deg = (int)this->H.get_row_to_cols()[v].size();
                                int16_t acc = 0;
                                for (int c = 0; c < deg; c++)
                                    acc = clamp_i16((int32_t)acc + (int32_t)c2v_ptr[c]);
                                this->post[v] = clamp_i16((int32_t)Y_N[v] + (int32_t)acc);
                                c2v_ptr += deg;
                            }
                            valid_synd = this->template check_syndrome_soft<int16_t, aff3ct::tools::LDPC_syndrome>(
                              this->post.data());
                            if (valid_synd) break;
                        }
                    }
                    if (ite == this->n_ite)
                    {
                        const int16_t* c2v_ptr = msg_c2v.data();
                        for (int v = 0; v < n_var; v++)
                        {
                            const int deg = (int)this->H.get_row_to_cols()[v].size();
                            int16_t acc = 0;
                            for (int c = 0; c < deg; c++)
                                acc = clamp_i16((int32_t)acc + (int32_t)c2v_ptr[c]);
                            this->post[v] = clamp_i16((int32_t)Y_N[v] + (int32_t)acc);
                            c2v_ptr += deg;
                        }
                    }
                    this->up_rule.end_decoding();
                    for (int v = 0; v < this->K; v++)
                        V_K[v] = !(this->post[this->info_bits_pos[v]] >= 0);
                    CWD[0] = valid_synd;
                    return !valid_synd;
                }
            };
            struct Decoder_layered_sat_i16
              : public aff3ct::module::Decoder_LDPC_BP_horizontal_layered<int16_t, int16_t, Update_rule_OMS_sat_i16>
            {
                using Base =
                  aff3ct::module::Decoder_LDPC_BP_horizontal_layered<int16_t, int16_t, Update_rule_OMS_sat_i16>;
                using Base::Base;
                static inline int16_t clamp_i16(int32_t x)
                {
                    return (int16_t)std::min<int32_t>(32767, std::max<int32_t>(-32768, x));
                }
                Decoder_layered_sat_i16* clone() const override
                {
                    auto m = new Decoder_layered_sat_i16(*this);
                    m->deep_copy(*this);
                    return m;
                }
                int _decode_siho(const int16_t* Y_N, int8_t* CWD, int16_t* V_K, const size_t frame_id) override
                {
                    auto& vn = this->var_nodes[frame_id];
                    auto& msgs = this->messages[frame_id];
                    for (size_t v = 0; v < vn.size(); v++)
                        vn[v] = clamp_i16((int32_t)vn[v] + (int32_t)Y_N[v]);
                    this->up_rule.begin_decoding(this->n_ite);
                    bool valid_synd = true;
                    const int n_chk = (int)this->H.get_n_cols();
                    for (int ite = 0; ite < this->n_ite; ite++)
                    {
                        this->up_rule.begin_ite(ite);
                        int kr = 0, kw = 0;
                        for (int c = 0; c < n_chk; c++)
                        {
                            const int deg = (int)this->H[c].size();
                            this->up_rule.begin_chk_node_in(c, deg);
                            for (int v = 0; v < deg; v++)
                            {
                                this->contributions[v] = clamp_i16((int32_t)vn[this->H[c][v]] - (int32_t)msgs[kr++]);
                                this->up_rule.compute_chk_node_in(v, this->contributions[v]);
                            }
                            this->up_rule.end_chk_node_in();
                            this->up_rule.begin_chk_node_out(c, deg);
                            for (int v = 0; v < deg; v++)
                            {
                                msgs[kw] = this->up_rule.compute_chk_node_out(v, this->contributions[v]);
                                vn[this->H[c][v]] = clamp_i16((int32_t)this->contributions[v] + (int32_t)msgs[kw++]);
                            }
                            this->up_rule.end_chk_node_out();
                        }
                        this->up_rule.end_ite();
                        valid_synd = this->template check_syndrome_soft<int16_t, aff3ct::tools::LDPC_syndrome>(
                          vn.data());
                        if (valid_synd) break;
                    }
                    this->up_rule.end_decoding();
                    for (int i = 0; i < this->K; i++)
                        V_K[i] = !(vn[this->info_bits_pos[i]] >= 0);
                    const int status = !valid_synd && this->enable_syndrome;
                    CWD[0] = !status;
                    return status;
                }
            };
            Update_rule_OMS_sat_i16 rule((int16_t)offset, this->sat_val_i16_);
            if (type == "BP_HORIZONTAL_LAYERED")
                return new Decoder_layered_sat_i16(
                  this->K_, this->N_cw_, n_ite, this->H_, this->info_bits_pos_, rule, enable_syndrome, syndrome_depth);
            else
                return new Decoder_flooding_sat_i16(
                  this->K_, this->N_cw_, n_ite, this->H_, this->info_bits_pos_, rule, enable_syndrome, syndrome_depth);
        }

        aff3ct::factory::Decoder_LDPC dec_params;
        dec_params.K = this->K_;
        dec_params.N_cw = this->N_cw_;
        dec_params.type = type;
        dec_params.implem = implem;
        dec_params.n_ite = n_ite;
        dec_params.simd_strategy = simd_strategy;
        dec_params.offset = offset;
        dec_params.norm_factor = norm_factor;
        dec_params.enable_syndrome = enable_syndrome;
        dec_params.syndrome_depth = syndrome_depth;
        return dec_params.build<int16_t, int16_t>(this->H_, this->info_bits_pos_);
    }
};

using Monitor_BFER_int = aff3ct::module::Monitor_BFER<int>;
using Monitor_BFER_reduction = aff3ct::tools::Monitor_reduction<Monitor_BFER_int>;

using Monitor_BFER_i16 = aff3ct::module::Monitor_BFER<int16_t>;
using Monitor_BFER_reduction_i16 = aff3ct::tools::Monitor_reduction<Monitor_BFER_i16>;

// Subclass of Source_random_fast<B> that safely handles sub-register frame sizes (e.g. K = 8 on AVX-512
// with F > 1, where odd frames sit at a 32-byte offset and are not 64-byte aligned).
template<typename B>
class Source_random_fast_any : public aff3ct::module::Source_random_fast<B>
{
  private:
    aff3ct::tools::PRNG_MT19937 mt19937_fallback_;

  public:
    Source_random_fast_any(const int K, const int seed = 0)
      : aff3ct::module::Source_random_fast<B>(K, seed)
      , mt19937_fallback_(seed)
    {
    }

    virtual Source_random_fast_any<B>* clone() const override
    {
        auto m = new Source_random_fast_any(*this);
        m->deep_copy(*this);
        return m;
    }

    virtual void set_seed(const int seed) override
    {
        aff3ct::module::Source_random_fast<B>::set_seed(seed);
        this->mt19937_fallback_.seed(seed);
    }

  protected:
    virtual void _generate(B* U_K, const size_t frame_id) override
    {
        if (mipp::isAligned(U_K))
        {
            aff3ct::module::Source_random_fast<B>::_generate(U_K, frame_id);
            return;
        }
        const auto size = static_cast<unsigned>(this->max_data_size);
        for (unsigned i = 0; i < size; i += sizeof(B) * 8)
        {
            auto randoms = this->mt19937_fallback_.rand_u32();
            unsigned j = 0;
            while ((j < sizeof(B) * 8) && (i + j < size))
            {
                U_K[i + j] = static_cast<B>(randoms & 0x1);
                randoms >>= 1;
                j++;
            }
        }
    }
};

// Subclass of Decoder_repetition<B, R> that matches Decoder_repetition_fast<B, R> when K >= mipp::nElReg<R>(),
// and also supports ultra-short block lengths K < mipp::nElReg<R>() (such as K = 8 on AVX-512 where nElReg = 16).
template<typename B, typename R>
class Decoder_repetition_fast_any : public aff3ct::module::Decoder_repetition<B, R>
{
  public:
    Decoder_repetition_fast_any(const int& K, const int& N, const bool buffered_encoding = true)
      : aff3ct::module::Decoder_repetition<B, R>(K, N, buffered_encoding)
    {
        const std::string name = "Decoder_repetition_fast";
        this->set_name(name);
        for (auto& t : this->tasks)
            t->set_replicability(true);
    }

    virtual Decoder_repetition_fast_any<B, R>* clone() const override
    {
        auto m = new Decoder_repetition_fast_any(*this);
        m->deep_copy(*this);
        return m;
    }

  protected:
    virtual int _decode_siso_alt(const R* sys, const R* par, R* ext, const size_t /*frame_id*/) override
    {
        const int n_el = mipp::nElReg<R>();
        const int vec_end = (this->K / n_el) * n_el;
        for (auto i = 0; i < vec_end; i += n_el)
        {
            auto r_ext = mipp::Reg<R>(&sys[i]);
            for (auto j = 0; j < this->rep_count; j++)
                r_ext += &par[j * this->K + i];
            r_ext.store(&ext[i]);
        }
        for (auto i = vec_end; i < this->K; i++)
        {
            R s = sys[i];
            for (auto j = 0; j < this->rep_count; j++)
                s += par[j * this->K + i];
            ext[i] = s;
        }
        return 0;
    }
};

} // namespace

PYBIND11_MODULE(py_aff3ct, m, py::mod_gil_not_used())
{
    m.doc() = "Minimal pybind11 wrapper exposing AFF3CT C++ modules as StreamPU Stateful modules for polyglot SDR chains";

    auto pyspu = (py::object)py::module_::import("streampu");
    // Enable automatic socket allocation in this extension's copy of StreamPU's Buffer_allocator
    spu::tools::Buffer_allocator::set_task_autoalloc(true);
    m.attr("spu") = pyspu;

    // --- Noise & SNR helper functions ---
    m.def(
      "ebn0_to_esn0",
      [](const float ebn0, const float R, const int bps) { return aff3ct::tools::ebn0_to_esn0(ebn0, R, bps); },
      "ebn0"_a,
      "R"_a = 1.0f,
      "bps"_a = 1,
      "Convert Eb/N0 (dB) to Es/N0 (dB).");

    m.def(
      "esn0_to_sigma",
      [](const float esn0, const int upf) { return aff3ct::tools::esn0_to_sigma(esn0, upf); },
      "esn0"_a,
      "upf"_a = 1,
      "Convert Es/N0 (dB) to AWGN standard deviation sigma.");

    m.def(
      "mipp_nElReg_f32",
      []() { return static_cast<int>(mipp::nElReg<float>()); },
      "Return the native SIMD register width in elements for float32.");

    m.def(
      "mipp_nElReg_i16",
      []() { return static_cast<int>(mipp::nElReg<int16_t>()); },
      "Return the native SIMD register width in elements for int16_t.");

    m.def(
      "set_single_wave",
      [](spu::module::Module& mdl, const bool enable)
      {
          struct Stateful_access : public spu::module::Stateful
          {
              using spu::module::Stateful::set_single_wave;
          };
          static_cast<Stateful_access&>(static_cast<spu::module::Stateful&>(mdl)).set_single_wave(enable);
      },
      "module"_a,
      "enable"_a = true,
      "Enable or disable single_wave mode (n_frames_per_wave = n_frames) on a StreamPU module.");

    // --- Sigma_buffer ---
    py::class_<Sigma_buffer>(m, "Sigma_buffer")
      .def(py::init<const size_t, const float>(), "n_frames"_a = 1, "sigma"_a = 1.0f)
      .def("set_sigma", &Sigma_buffer::set_sigma, "sigma"_a)
      .def("get_sigma", &Sigma_buffer::get_sigma)
      .def("bind", &Sigma_buffer::bind, "socket"_a);

    // --- Source_random_fast (32-bit and 16-bit) ---
    py::class_<Source_random_fast_any<int>, spu::module::Source<int>>(m, "Source_random_fast")
      .def(py::init<const int, const int>(), "K"_a, "seed"_a = 0)
      .def("set_seed", &Source_random_fast_any<int>::set_seed, "seed"_a);

    py::class_<Source_random_fast_any<int16_t>, spu::module::Source<int16_t>>(m, "Source_random_fast_i16")
      .def(py::init<const int, const int>(), "K"_a, "seed"_a = 0)
      .def("set_seed", &Source_random_fast_any<int16_t>::set_seed, "seed"_a);

    // --- Repetition Codec ---
    py::class_<aff3ct::module::Encoder_repetition_sys<int>, spu::module::Stateful>(m, "Encoder_repetition_sys")
      .def(py::init<const int&, const int&, const bool>(), "K"_a, "N"_a, "buffered_encoding"_a = true);

    py::class_<aff3ct::module::Decoder_repetition_std<int, float>,
               spu::module::Stateful,
               spu::tools::Interface_reset>(m, "Decoder_repetition_std")
      .def(py::init<const int&, const int&, const bool>(), "K"_a, "N"_a, "buffered_encoding"_a = true);

    py::class_<Decoder_repetition_fast_any<int, float>,
               spu::module::Stateful,
               spu::tools::Interface_reset>(m, "Decoder_repetition_fast")
      .def(py::init<const int&, const int&, const bool>(), "K"_a, "N"_a, "buffered_encoding"_a = true);

    // --- Modem BPSK Fast (32-bit and 16-bit) ---
    py::class_<aff3ct::module::Modem_BPSK_fast<int, float, float>, spu::module::Stateful>(m, "Modem_BPSK_fast")
      .def(py::init<const int, const bool>(), "N"_a, "disable_sig2"_a = false);

    py::class_<aff3ct::module::Modem_BPSK_fast<int16_t, float, float>, spu::module::Stateful>(m, "Modem_BPSK_fast_i16")
      .def(py::init<const int, const bool>(), "N"_a, "disable_sig2"_a = false);

    // --- Quantizer Pow2 Fast (float -> int16_t) ---
    py::class_<aff3ct::module::Quantizer_pow2_fast<float, int16_t>, spu::module::Stateful>(m, "Quantizer_pow2_fast_i16")
      .def(py::init<const int, const short&, const short&>(),
           "N"_a,
           "n_decimals"_a = 2,
           "n_bits"_a = 8);

    // --- Channel AWGN LLR ---
    py::class_<aff3ct::module::Channel_AWGN_LLR<float>,
               spu::module::Stateful,
               spu::tools::Interface_set_seed>(m, "Channel_AWGN_LLR")
      .def(py::init(
             [](const int N, const std::string& implem_str, const int seed, const bool add_users)
             {
                 auto implem = (implem_str == "FAST")
                                 ? aff3ct::tools::Gaussian_noise_generator_implem::FAST
                                 : aff3ct::tools::Gaussian_noise_generator_implem::STD;
                 return new aff3ct::module::Channel_AWGN_LLR<float>(N, implem, seed, add_users);
             }),
           "N"_a,
           "implem"_a = "FAST",
           "seed"_a = 0,
           "add_users"_a = false)
      .def("set_seed", &aff3ct::module::Channel_AWGN_LLR<float>::set_seed, "seed"_a);

    // --- Monitor_BFER & Monitor_BFER_reduction (32-bit) ---
    py::class_<Monitor_BFER_int,
               spu::module::Stateful,
               spu::tools::Interface_is_done,
               spu::tools::Interface_reset>(m, "Monitor_BFER")
      .def(py::init<const int, const unsigned, const unsigned, const bool>(),
           "K"_a,
           "max_fe"_a = 100,
           "max_n_frames"_a = 0,
           "count_unknown_values"_a = false)
      .def("get_K", &Monitor_BFER_int::get_K)
      .def("get_max_fe", &Monitor_BFER_int::get_max_fe)
      .def("get_max_n_frames", &Monitor_BFER_int::get_max_n_frames)
      .def("get_n_analyzed_fra", &Monitor_BFER_int::get_n_analyzed_fra)
      .def("get_n_fe", &Monitor_BFER_int::get_n_fe)
      .def("get_n_be", &Monitor_BFER_int::get_n_be)
      .def("get_fer", &Monitor_BFER_int::get_fer)
      .def("get_ber", &Monitor_BFER_int::get_ber)
      .def("fe_limit_achieved", &Monitor_BFER_int::fe_limit_achieved)
      .def("frame_limit_achieved", &Monitor_BFER_int::frame_limit_achieved)
      .def("is_done", &Monitor_BFER_int::is_done)
      .def("disable_is_done", &Monitor_BFER_int::disable_is_done, "no_is_done"_a)
      .def("reset", &Monitor_BFER_int::reset);

    py::class_<Monitor_BFER_reduction>(m, "Monitor_BFER_reduction")
      .def(py::init<const std::vector<Monitor_BFER_int*>&>(), "monitors"_a)
      .def(
        "set_reduce_frequency_ms",
        [](Monitor_BFER_reduction& self, const int ms)
        { self.set_reduce_frequency(std::chrono::milliseconds(ms)); },
        "ms"_a = 1)
      .def("is_done", &Monitor_BFER_reduction::is_done)
      .def("reduce", &Monitor_BFER_reduction::reduce, "fully"_a = true)
      .def("reset", &Monitor_BFER_reduction::reset)
      .def("get_n_analyzed_fra", &Monitor_BFER_reduction::get_n_analyzed_fra)
      .def("get_n_fe", &Monitor_BFER_reduction::get_n_fe)
      .def("get_n_be", &Monitor_BFER_reduction::get_n_be)
      .def("get_fer", &Monitor_BFER_reduction::get_fer)
      .def("get_ber", &Monitor_BFER_reduction::get_ber);

    // --- Monitor_BFER & Monitor_BFER_reduction (16-bit) ---
    py::class_<Monitor_BFER_i16,
               spu::module::Stateful,
               spu::tools::Interface_is_done,
               spu::tools::Interface_reset>(m, "Monitor_BFER_i16")
      .def(py::init<const int, const unsigned, const unsigned, const bool>(),
           "K"_a,
           "max_fe"_a = 100,
           "max_n_frames"_a = 0,
           "count_unknown_values"_a = false)
      .def("get_K", &Monitor_BFER_i16::get_K)
      .def("get_max_fe", &Monitor_BFER_i16::get_max_fe)
      .def("get_max_n_frames", &Monitor_BFER_i16::get_max_n_frames)
      .def("get_n_analyzed_fra", &Monitor_BFER_i16::get_n_analyzed_fra)
      .def("get_n_fe", &Monitor_BFER_i16::get_n_fe)
      .def("get_n_be", &Monitor_BFER_i16::get_n_be)
      .def("get_fer", &Monitor_BFER_i16::get_fer)
      .def("get_ber", &Monitor_BFER_i16::get_ber)
      .def("fe_limit_achieved", &Monitor_BFER_i16::fe_limit_achieved)
      .def("frame_limit_achieved", &Monitor_BFER_i16::frame_limit_achieved)
      .def("is_done", &Monitor_BFER_i16::is_done)
      .def("disable_is_done", &Monitor_BFER_i16::disable_is_done, "no_is_done"_a)
      .def("reset", &Monitor_BFER_i16::reset);

    py::class_<Monitor_BFER_reduction_i16>(m, "Monitor_BFER_reduction_i16")
      .def(py::init<const std::vector<Monitor_BFER_i16*>&>(), "monitors"_a)
      .def(
        "set_reduce_frequency_ms",
        [](Monitor_BFER_reduction_i16& self, const int ms)
        { self.set_reduce_frequency(std::chrono::milliseconds(ms)); },
        "ms"_a = 1)
      .def("is_done", &Monitor_BFER_reduction_i16::is_done)
      .def("reduce", &Monitor_BFER_reduction_i16::reduce, "fully"_a = true)
      .def("reset", &Monitor_BFER_reduction_i16::reset)
      .def("get_n_analyzed_fra", &Monitor_BFER_reduction_i16::get_n_analyzed_fra)
      .def("get_n_fe", &Monitor_BFER_reduction_i16::get_n_fe)
      .def("get_n_be", &Monitor_BFER_reduction_i16::get_n_be)
      .def("get_fer", &Monitor_BFER_reduction_i16::get_fer)
      .def("get_ber", &Monitor_BFER_reduction_i16::get_ber);

    // --- LDPC Base Classes & LDPC_Code Helper ---
    py::class_<aff3ct::module::Encoder_LDPC<int>, spu::module::Stateful>(m, "Encoder_LDPC");
    py::class_<aff3ct::module::Puncturer<int, float>, spu::module::Stateful>(m, "Puncturer");
    py::class_<aff3ct::module::Decoder_SIHO<int, float>,
               spu::module::Stateful,
               spu::tools::Interface_reset>(m, "Decoder_SIHO");

    py::class_<aff3ct::module::Encoder_LDPC<int16_t>, spu::module::Stateful>(m, "Encoder_LDPC_i16");
    py::class_<aff3ct::module::Puncturer<int16_t, int16_t>, spu::module::Stateful>(m, "Puncturer_i16");
    py::class_<aff3ct::module::Decoder_SIHO<int16_t, int16_t>,
               spu::module::Stateful,
               spu::tools::Interface_reset>(m, "Decoder_SIHO_i16");

    py::class_<LDPC_Code>(m, "LDPC_Code")
      .def(py::init<const int, const int, const std::string&, const std::string&, const std::string&>(),
           "K"_a = 512,
           "N"_a = 1024,
           "enc_type"_a = "LDPC_5G",
           "qc_path"_a = "",
           "h_reorder"_a = "NONE")
      .def_property_readonly("K", &LDPC_Code::get_K)
      .def_property_readonly("N_cw", &LDPC_Code::get_N_cw)
      .def_property_readonly("N", &LDPC_Code::get_N)
      .def_property_readonly("n_rows", &LDPC_Code::get_n_rows)
      .def_property_readonly("n_cols", &LDPC_Code::get_n_cols)
      .def_property_readonly("n_connections", &LDPC_Code::get_n_connections)
      .def_property_readonly("cols_max_degree", &LDPC_Code::get_cols_max_degree)
      .def_property_readonly("rows_max_degree", &LDPC_Code::get_rows_max_degree)
      .def_property_readonly("sat_val_i16", &LDPC_Code::get_sat_val_i16)
      .def("get_qc_path", &LDPC_Code::get_qc_path)
      .def("get_tanner_graph_arrays", &LDPC_Code::get_tanner_graph_arrays)
      .def("get_julia_graph_ctor",
           &LDPC_Code::get_julia_graph_ctor,
           "n_ite"_a = 20,
           "enable_syndrome"_a = true,
           "syndrome_depth"_a = 1,
           "dec_type"_a = "BP_FLOODING",
           "dec_implem"_a = "SPA",
           "q_type"_a = "f32",
           "offset"_a = 0.0f)
      .def("get_julia_workspace_ctor",
           &LDPC_Code::get_julia_workspace_ctor,
           "simd_strategy"_a = "",
           "q_type"_a = "f32",
           "n_frames_per_wave"_a = 1)
      .def("build_encoder", &LDPC_Code::build_encoder, py::return_value_policy::take_ownership)
      .def("build_puncturer", &LDPC_Code::build_puncturer, py::return_value_policy::take_ownership)
      .def("build_decoder",
           &LDPC_Code::build_decoder,
           "type"_a = "BP_FLOODING",
           "implem"_a = "SPA",
           "n_ite"_a = 20,
           "simd_strategy"_a = "",
           "offset"_a = 0.0f,
           "norm_factor"_a = 1.0f,
           "enable_syndrome"_a = true,
           "syndrome_depth"_a = 1,
           py::return_value_policy::take_ownership)
      .def("build_encoder_i16", &LDPC_Code::build_encoder_i16, py::return_value_policy::take_ownership)
      .def("build_puncturer_i16", &LDPC_Code::build_puncturer_i16, py::return_value_policy::take_ownership)
      .def("build_quantizer_i16",
           &LDPC_Code::build_quantizer_i16,
           "n_decimals"_a = 2,
           "n_bits"_a = 8,
           py::return_value_policy::take_ownership)
      .def("build_decoder_i16",
           &LDPC_Code::build_decoder_i16,
           "type"_a = "BP_FLOODING",
           "implem"_a = "OMS",
           "n_ite"_a = 20,
           "simd_strategy"_a = "",
           "offset"_a = 1.0f,
           "norm_factor"_a = 1.0f,
           "enable_syndrome"_a = true,
           "syndrome_depth"_a = 1,
           py::return_value_policy::take_ownership);

    // --- Runtime seeding, reset, and execution helpers matching the C++ binaries ---
    m.def(
      "seed_sequence_modules",
      [](spu::runtime::Sequence& seq)
      {
          std::mt19937 prng;
          for (auto* mod : seq.get_modules<spu::module::Source<int>>())
              mod->set_seed(prng());
          for (auto* mod : seq.get_modules<spu::module::Source<int16_t>>())
              mod->set_seed(prng());
          for (auto* mod : seq.get_modules<aff3ct::module::Channel<>>())
              mod->set_seed(prng());
      },
      "sequence"_a,
      "Deterministically seed all Source and Channel clones in a Sequence using std::mt19937.");

    m.def(
      "seed_pipeline_modules",
      [](spu::runtime::Pipeline& pipe)
      {
          std::mt19937 prng;
          for (auto* mod : pipe.get_modules<spu::module::Source<int>>())
              mod->set_seed(prng());
          for (auto* mod : pipe.get_modules<spu::module::Source<int16_t>>())
              mod->set_seed(prng());
          for (auto* mod : pipe.get_modules<aff3ct::module::Channel<>>())
              mod->set_seed(prng());
      },
      "pipeline"_a,
      "Deterministically seed all Source and Channel clones in a Pipeline using std::mt19937.");

    m.def(
      "reset_sequence_modules",
      [](spu::runtime::Sequence& seq)
      {
          for (auto* mod : seq.get_modules<spu::tools::Interface_reset>())
              mod->reset();
      },
      "sequence"_a,
      "Reset all Interface_reset modules in a Sequence.");

    m.def(
      "reset_pipeline_modules",
      [](spu::runtime::Pipeline& pipe)
      {
          for (auto* mod : pipe.get_modules<spu::tools::Interface_reset>())
              mod->reset();
      },
      "pipeline"_a,
      "Reset all Interface_reset modules in a Pipeline.");

    m.def(
      "configure_sequence_tasks",
      [](spu::runtime::Sequence& seq, const bool stats)
      {
          for (auto& mod : seq.get_modules<spu::module::Module>(false))
              for (auto& tsk : mod->tasks)
              {
                  tsk->set_debug(false);
                  tsk->set_stats(stats);
                  if (!tsk->is_debug() && !tsk->is_stats())
                      tsk->set_fast(true);
              }
      },
      "sequence"_a,
      "stats"_a = false,
      "Configure debug, stats, and fast flags on all tasks in a Sequence.");

    m.def(
      "configure_pipeline_tasks",
      [](spu::runtime::Pipeline& pipe, const bool stats)
      {
          for (auto& mod : pipe.get_modules<spu::module::Module>(false))
              for (auto& tsk : mod->tasks)
              {
                  tsk->set_debug(false);
                  tsk->set_stats(stats);
                  if (!tsk->is_debug() && !tsk->is_stats())
                      tsk->set_fast(true);
              }
      },
      "pipeline"_a,
      "stats"_a = false,
      "Configure debug, stats, and fast flags on all tasks in a Pipeline.");

    m.def(
      "exec_sequence",
      [](spu::runtime::Sequence& seq, Monitor_BFER_reduction& mnt_red) -> double
      {
          py::gil_scoped_release release{};
          const auto t_start = std::chrono::steady_clock::now();
          seq.exec([&]() -> bool { return mnt_red.is_done() || seq.is_done(); });
          mnt_red.reduce();
          const auto t_stop = std::chrono::steady_clock::now();
          return std::chrono::duration<double, std::micro>(t_stop - t_start).count();
      },
      "sequence"_a,
      "monitor_red"_a,
      "Execute a replicated Sequence (int32) until Monitor_BFER_reduction is done and return elapsed time in microseconds.");

    m.def(
      "exec_sequence_i16",
      [](spu::runtime::Sequence& seq, Monitor_BFER_reduction_i16& mnt_red) -> double
      {
          py::gil_scoped_release release{};
          const auto t_start = std::chrono::steady_clock::now();
          seq.exec([&]() -> bool { return mnt_red.is_done() || seq.is_done(); });
          mnt_red.reduce();
          const auto t_stop = std::chrono::steady_clock::now();
          return std::chrono::duration<double, std::micro>(t_stop - t_start).count();
      },
      "sequence"_a,
      "monitor_red"_a,
      "Execute a replicated Sequence (int16) until Monitor_BFER_reduction_i16 is done and return elapsed time in microseconds.");

    m.def(
      "exec_pipeline",
      [](spu::runtime::Pipeline& pipe,
         Monitor_BFER_int& mnt,
         spu::module::Source<int>& src,
         const size_t max_frame,
         const size_t n_frames) -> double
      {
          py::gil_scoped_release release{};
          const auto t_start = std::chrono::steady_clock::now();
          if (max_frame > 0)
          {
              size_t generated_fra = 0;
              mnt.disable_is_done(true);
              pipe.exec(std::vector<std::function<bool()>>{
                [&]() -> bool
                {
                    generated_fra += n_frames;
                    return generated_fra >= max_frame || mnt.fe_limit_achieved() || src.is_done();
                },
                [&]() -> bool { return mnt.fe_limit_achieved(); },
                [&]() -> bool { return mnt.fe_limit_achieved(); } });
          }
          else
          {
              pipe.exec();
          }
          const auto t_stop = std::chrono::steady_clock::now();
          return std::chrono::duration<double, std::micro>(t_stop - t_start).count();
      },
      "pipeline"_a,
      "monitor"_a,
      "source"_a,
      "max_frame"_a = 0,
      "n_frames"_a = 1,
      "Execute a 3-stage Pipeline (int32) until monitor/frame limit is reached and return elapsed time in microseconds.");

    m.def(
      "exec_pipeline_i16",
      [](spu::runtime::Pipeline& pipe,
         Monitor_BFER_i16& mnt,
         spu::module::Source<int16_t>& src,
         const size_t max_frame,
         const size_t n_frames) -> double
      {
          py::gil_scoped_release release{};
          const auto t_start = std::chrono::steady_clock::now();
          if (max_frame > 0)
          {
              size_t generated_fra = 0;
              mnt.disable_is_done(true);
              pipe.exec(std::vector<std::function<bool()>>{
                [&]() -> bool
                {
                    generated_fra += n_frames;
                    return generated_fra >= max_frame || mnt.fe_limit_achieved() || src.is_done();
                },
                [&]() -> bool { return mnt.fe_limit_achieved(); },
                [&]() -> bool { return mnt.fe_limit_achieved(); } });
          }
          else
          {
              pipe.exec();
          }
          const auto t_stop = std::chrono::steady_clock::now();
          return std::chrono::duration<double, std::micro>(t_stop - t_start).count();
      },
      "pipeline"_a,
      "monitor"_a,
      "source"_a,
      "max_frame"_a = 0,
      "n_frames"_a = 1,
      "Execute a 3-stage Pipeline (int16) until monitor/frame limit is reached and return elapsed time in microseconds.");

    m.def(
      "thread_pinning_init",
      []()
      {
          if (!spu::tools::Thread_pinning::is_init())
              spu::tools::Thread_pinning::init();
      },
      "Initialize hwloc Thread_pinning if not already initialized.");

    m.def(
      "thread_pinning_destroy",
      []()
      {
          if (spu::tools::Thread_pinning::is_init())
              spu::tools::Thread_pinning::destroy();
      },
      "Destroy hwloc Thread_pinning if initialized.");
}
