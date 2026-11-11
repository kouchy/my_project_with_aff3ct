// Receiver-only benchmark chain: "antenna" module replaying pre-generated noisy frames.
//
//   stage 0 : Antenna_replay::receive   (one wave of F noisy frames per call, paced like an ADC by the runtime)
//   stage 1 : demodulator [-> quantizer] [-> depuncturer] -> decoder   (replicated on t threads)
//   stage 2 : Monitor_BFER::check_errors (compares the decoded bits with the reference bits replayed in stage 0)
//
// The frames are generated once, before the measurements, by the real transmitter chain (source, encoder,
// [puncturer], modulator, AWGN channel) and stored in a ring buffer larger than the last-level cache (64 MiB of
// channel samples by default), so that the receiver reads "cold" samples as it would from a DMA buffer. The ring
// buffer is shared (not copied) by the clones of the module.
#pragma once

#include <algorithm>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <streampu.hpp>

template<typename B = int, typename R = float>
class Antenna_replay : public spu::module::Stateful
{
  protected:
    const size_t K;
    const size_t N;
    size_t n_buf;                          // number of frames in the ring buffer (multiple of n_frames)
    std::shared_ptr<std::vector<B>> ref_U; // reference information bits (n_buf x K)
    std::shared_ptr<std::vector<R>> chn_Y; // noisy channel samples      (n_buf x N)
    size_t rd = 0;                         // next frame to replay
    size_t wr = 0;                         // next frame to record (fill phase)

  public:
    Antenna_replay(const size_t K, const size_t N, const size_t buffer_bytes = size_t(64) << 20)
      : spu::module::Stateful()
      , K(K)
      , N(N)
      , n_buf(std::max<size_t>(1, buffer_bytes / (N * sizeof(R))))
      , ref_U(std::make_shared<std::vector<B>>())
      , chn_Y(std::make_shared<std::vector<R>>())
    {
        this->set_name("Antenna_replay");
        this->set_short_name("Antenna_replay");
        // one codelet call per wave of F frames (contiguous in the ring buffer): a single memcpy per socket
        this->set_single_wave(true);

        auto& p1 = this->create_task("receive");
        auto ps_U = this->template create_socket_out<B>(p1, "U_K", K);
        auto ps_Y = this->template create_socket_out<R>(p1, "Y_N", N);
        this->create_codelet(p1,
                             [ps_U, ps_Y](spu::module::Module& m, spu::runtime::Task& t, const size_t) -> int
                             {
                                 auto& a = static_cast<Antenna_replay&>(m);
                                 a._receive(t[ps_U].template get_dataptr<B>(), t[ps_Y].template get_dataptr<R>());
                                 return spu::runtime::status_t::SUCCESS;
                             });

        auto& p2 = this->create_task("record");
        auto ps_Ui = this->template create_socket_in<B>(p2, "U_K", K);
        auto ps_Yi = this->template create_socket_in<R>(p2, "Y_N", N);
        this->create_codelet(p2,
                             [ps_Ui, ps_Yi](spu::module::Module& m, spu::runtime::Task& t, const size_t) -> int
                             {
                                 auto& a = static_cast<Antenna_replay&>(m);
                                 a._record(t[ps_Ui].template get_dataptr<const B>(),
                                           t[ps_Yi].template get_dataptr<const R>());
                                 return spu::runtime::status_t::SUCCESS;
                             });
    }

    virtual ~Antenna_replay() = default;

    virtual Antenna_replay* clone() const override
    {
        auto m = new Antenna_replay(*this); // shares the ring buffer (shared_ptr)
        m->deep_copy(*this);
        return m;
    }

    size_t get_n_buf() const { return this->n_buf; }
    size_t get_buffer_bytes() const { return this->n_buf * this->N * sizeof(R); }
    bool is_filled() const { return this->wr >= this->n_buf; }
    void rewind() { this->rd = 0; }

    // Generate the ring buffer by executing the transmitter tasks (already bound together and to `record`) until
    // the buffer is full. The number of frames is rounded up to a multiple of the number of frames per wave.
    void fill(const std::vector<spu::runtime::Task*>& tx_tasks)
    {
        const size_t F = this->get_n_frames();
        this->n_buf = ((this->n_buf + F - 1) / F) * F;
        this->ref_U->assign(this->n_buf * this->K, B(0));
        this->chn_Y->assign(this->n_buf * this->N, R(0));
        this->wr = 0;
        this->rd = 0;
        auto& rec = (*this)("record");
        while (this->wr < this->n_buf)
        {
            for (auto* t : tx_tasks)
                t->exec();
            rec.exec();
        }
    }

  protected:
    // called once per wave (single-wave module): copy F contiguous frames (n_buf is a multiple of F)
    void _receive(B* U_K, R* Y_N)
    {
        const size_t F = this->get_n_frames();
        std::memcpy(U_K, this->ref_U->data() + this->rd * this->K, F * this->K * sizeof(B));
        std::memcpy(Y_N, this->chn_Y->data() + this->rd * this->N, F * this->N * sizeof(R));
        this->rd += F;
        if (this->rd >= this->n_buf) this->rd = 0;
    }

    void _record(const B* U_K, const R* Y_N)
    {
        const size_t F = this->get_n_frames();
        if (this->wr + F > this->n_buf) return;
        std::memcpy(this->ref_U->data() + this->wr * this->K, U_K, F * this->K * sizeof(B));
        std::memcpy(this->chn_Y->data() + this->wr * this->N, Y_N, F * this->N * sizeof(R));
        this->wr += F;
    }
};

