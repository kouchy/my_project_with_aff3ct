#include <algorithm>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <functional>
#include <iostream>
#include <memory>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
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
// together with its information bit positions, puncturing pattern, and Julia state constructors.
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
    }

    int get_K() const { return this->K_; }
    int get_N_cw() const { return this->N_cw_; }
    int get_N() const { return this->N_; }
    int get_n_rows() const { return (int)this->H_.get_n_rows(); }
    int get_n_cols() const { return (int)this->H_.get_n_cols(); }
    int get_n_connections() const { return (int)this->H_.get_n_connections(); }
    int get_cols_max_degree() const { return (int)this->H_.get_cols_max_degree(); }
    int get_rows_max_degree() const { return (int)this->H_.get_rows_max_degree(); }
    const std::string& get_qc_path() const { return this->qc_path_; }

    const aff3ct::tools::Sparse_matrix& get_H() const { return this->H_; }
    const std::vector<uint32_t>& get_info_bits_pos() const { return this->info_bits_pos_; }
    const std::vector<bool>& get_pct_pattern() const { return this->pct_pattern_; }

    std::string get_julia_graph_ctor(const int n_ite = 20,
                                     const bool enable_syndrome = true,
                                     const int syndrome_depth = 1) const
    {
        std::stringstream ss;
        ss << "LDPC_Graph("
           << "Int32(" << n_ite << "), "
           << (enable_syndrome ? "true" : "false") << ", "
           << "Int32(" << syndrome_depth << "), "
           << "Int32(" << this->H_.get_n_rows() << "), "
           << "Int32(" << this->H_.get_n_cols() << "), "
           << "UInt(" << reinterpret_cast<uintptr_t>(this->h_cols_.data()) << "), Int32(" << this->h_cols_.size() << "), "
           << "UInt(" << reinterpret_cast<uintptr_t>(this->info_bits_pos_.data()) << "), Int32(" << this->K_ << ")"
           << ")";
        return ss.str();
    }

    std::string get_julia_workspace_ctor() const
    {
        std::stringstream ss;
        ss << "LDPC_Workspace("
           << "Int32(" << this->H_.get_n_rows() << "), "
           << "Int32(" << this->H_.get_n_connections() << "), "
           << "Int32(" << this->H_.get_cols_max_degree() << ")"
           << ")";
        return ss.str();
    }

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
};

using Monitor_BFER_int = aff3ct::module::Monitor_BFER<int>;
using Monitor_BFER_reduction = aff3ct::tools::Monitor_reduction<Monitor_BFER_int>;

} // namespace

PYBIND11_MODULE(py_aff3ct, m)
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
      "set_single_wave",
      [](spu::module::Module& mdl, const bool enable)
      {
          struct Stateful_access : public spu::module::Stateful
          {
              using spu::module::Stateful::set_single_wave;
          };
          if (auto* sl = dynamic_cast<spu::module::Stateless*>(&mdl))
              sl->set_single_wave(enable);
          else if (auto* sf = dynamic_cast<spu::module::Stateful*>(&mdl))
              static_cast<Stateful_access*>(sf)->set_single_wave(enable);
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

    // --- Source_random_fast ---
    py::class_<aff3ct::module::Source_random_fast<int>, spu::module::Source<int>>(m, "Source_random_fast")
      .def(py::init<const int, const int>(), "K"_a, "seed"_a = 0)
      .def("set_seed", &aff3ct::module::Source_random_fast<int>::set_seed, "seed"_a);

    // --- Repetition Codec ---
    py::class_<aff3ct::module::Encoder_repetition_sys<int>, spu::module::Stateful>(m, "Encoder_repetition_sys")
      .def(py::init<const int&, const int&, const bool>(), "K"_a, "N"_a, "buffered_encoding"_a = true);

    py::class_<aff3ct::module::Decoder_repetition_std<int, float>,
               spu::module::Stateful,
               spu::tools::Interface_reset>(m, "Decoder_repetition_std")
      .def(py::init<const int&, const int&, const bool>(), "K"_a, "N"_a, "buffered_encoding"_a = true);

    py::class_<aff3ct::module::Decoder_repetition_fast<int, float>,
               spu::module::Stateful,
               spu::tools::Interface_reset>(m, "Decoder_repetition_fast")
      .def(py::init<const int&, const int&, const bool>(), "K"_a, "N"_a, "buffered_encoding"_a = true);

    // --- Modem BPSK Fast ---
    py::class_<aff3ct::module::Modem_BPSK_fast<int, float, float>, spu::module::Stateful>(m, "Modem_BPSK_fast")
      .def(py::init<const int, const bool>(), "N"_a, "disable_sig2"_a = false);

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

    // --- Monitor_BFER & Monitor_BFER_reduction ---
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

    // --- LDPC Base Classes & LDPC_Code Helper ---
    py::class_<aff3ct::module::Encoder_LDPC<int>, spu::module::Stateful>(m, "Encoder_LDPC");
    py::class_<aff3ct::module::Puncturer<int, float>, spu::module::Stateful>(m, "Puncturer");
    py::class_<aff3ct::module::Decoder_SIHO<int, float>,
               spu::module::Stateful,
               spu::tools::Interface_reset>(m, "Decoder_SIHO");

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
      .def("get_qc_path", &LDPC_Code::get_qc_path)
      .def("get_julia_graph_ctor",
           &LDPC_Code::get_julia_graph_ctor,
           "n_ite"_a = 20,
           "enable_syndrome"_a = true,
           "syndrome_depth"_a = 1)
      .def("get_julia_workspace_ctor", &LDPC_Code::get_julia_workspace_ctor)
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
           py::return_value_policy::take_ownership);

    // --- Runtime seeding, reset, and execution helpers matching the C++ binaries ---
    m.def(
      "seed_sequence_modules",
      [](spu::runtime::Sequence& seq)
      {
          std::mt19937 prng;
          for (auto* mod : seq.get_modules<spu::module::Source<>>())
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
          for (auto* mod : pipe.get_modules<spu::module::Source<>>())
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
      "Execute a replicated Sequence until Monitor_BFER_reduction is done and return elapsed time in microseconds.");

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
      "Execute a 3-stage Pipeline until monitor/frame limit is reached and return elapsed time in microseconds.");
}
