// CPU-compute MoE executor for the cpu and hybrid MoE strategies, and its Python binding.
//
// Decode ships activations to the CPU, computes the routed experts here (reading
// the pinned host expert banks at full RAM bandwidth), and ships the results back.
// One task (one MoE layer) is in flight at a time: submit() wakes the persistent worker
// pool, the pool runs pass 1 (gate_up) and pass 2 (down), and sync() waits for it.
// The GPU rings a doorbell to submit and waits on it to sync (GpuSync, gpu_sync.h), so the
// whole decode step stays in one CUDA graph; cudaLaunchHostFunc nodes are the fallback.
//
// Weight formats live in kernels/, one file each behind the MoeFormat interface (format.h).
// Compute is FP32-accumulate; the intermediate is stored bf16 to match the GPU decode path.

#include <algorithm>
#include <array>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <chrono>
#include <cstring>
#include <map>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <tuple>
#include <utility>
#include <vector>

#include <freetoken/hip_compat.h>
#include <torch/extension.h>

#if !defined(_WIN32)
#include <dlfcn.h>
#endif

#include "format.h"
#include "gpu_sync.h"
#include "gpu_sync_io.h"
#include "threads.h"

namespace freetoken::cpu_moe {

std::unique_ptr<MoeFormat> make_bf16_format(const MoeParams& p, const FormatArgs& a);
std::unique_ptr<MoeFormat> make_nvfp4_format(const MoeParams& p, const FormatArgs& a);
std::unique_ptr<MoeFormat> make_nvfp4_b12x_format(const MoeParams& p, const FormatArgs& a);
std::unique_ptr<MoeFormat> make_mxfp4_format(const MoeParams& p, const FormatArgs& a);
std::unique_ptr<MoeFormat> make_ds_fp4_format(const MoeParams& p, const FormatArgs& a);
std::unique_ptr<MoeFormat> make_fp8_block_format(const MoeParams& p, const FormatArgs& a);
std::unique_ptr<MoeFormat> make_gguf_format(const MoeParams& p, const FormatArgs& a);
std::unique_ptr<MoeFormat> make_gguf_scaled_format(const MoeParams& p, const FormatArgs& a);
void set_ggml_cpu(uintptr_t addr);
void quant_act_b12x(const bf16_t* x, int K, int32_t* a4, float* asb);
#if CPU_MOE_X86
void quant_act_b12x_avx2(const bf16_t* x, int K, int32_t* a4, float* asb);
#endif
void bind_locked_fetch(pybind11::module_& m);

namespace {

constexpr int kMaxTopK = 64;

// One MoE layer on the first num_tokens rows of the pinned IO.
// It is also the doorbell word the GPU writes, a constant baked into each captured graph; num_tokens >= 1 keeps it nonzero, since 0 means no doorbell.
struct MoeTask {
  uint64_t layer : 32;
  uint64_t num_tokens : 32;
};
static_assert(sizeof(MoeTask) == sizeof(uint64_t), "MoeTask must pack into one 64-bit word");

inline uint64_t to_word(MoeTask t) {
  uint64_t w;
  std::memcpy(&w, &t, sizeof(w));
  return w;
}

inline MoeTask from_word(uint64_t w) {
  MoeTask t;
  std::memcpy(&t, &w, sizeof(t));
  return t;
}

// Every weight format this module serves, keyed by the name cpu_executor.py passes; weight_formats() exports the names.
const std::pair<const char*, std::unique_ptr<MoeFormat> (*)(const MoeParams&, const FormatArgs&)> kFormats[] = {
    {"bf16", make_bf16_format},     {"nvfp4", make_nvfp4_format},         {"mxfp4_triton", make_mxfp4_format},
    {"ds_fp4", make_ds_fp4_format}, {"fp8_block", make_fp8_block_format}, {"gguf", make_gguf_format},
    {"gguf_scaled", make_gguf_scaled_format}, {"nvfp4_b12x", make_nvfp4_b12x_format},
};

std::unique_ptr<MoeFormat> make_format(const std::string& name, const MoeParams& p, const FormatArgs& a) {
  for (const auto& [n, make] : kFormats)
    if (name == n) return make(p, a);
  throw std::invalid_argument("unknown CPU MoE weight format " + name);
}

// One format instance as cpu_executor.py passes it: name, bank tables by role, integer options.
using FormatSpec = std::tuple<std::string, std::map<std::string, uint64_t>, std::map<std::string, int64_t>>;

// cuStreamWriteValue32, resolved at run time like the memops of gpu_sync_cuda.cpp.
using cuWrite32_fn = int (*)(void* stream, unsigned long long addr, unsigned int value, unsigned int flags);
cuWrite32_fn resolve_write32() {
#if defined(_WIN32) || defined(__HIP_PLATFORM_AMD__) || defined(USE_ROCM)
  return nullptr;
#else
  static cuWrite32_fn fn = [] {
    void* h = dlopen("libcuda.so.1", RTLD_LAZY | RTLD_LOCAL);
    if (h == nullptr) return static_cast<cuWrite32_fn>(nullptr);
    auto f = reinterpret_cast<cuWrite32_fn>(dlsym(h, "cuStreamWriteValue32_v2"));
    if (f == nullptr) f = reinterpret_cast<cuWrite32_fn>(dlsym(h, "cuStreamWriteValue32"));
    return f;
  }();
  return fn;
#endif
}

struct CpuMoeExecutor {
  // A host-function callback gets one pointer, so it carries the executor with the task.
  struct HostTask {
    CpuMoeExecutor* exec;
    MoeTask task;
  };

  const int num_layers, num_experts;
  const bool apply_on_input;
  MoeParams params;  // the formats keep a reference; declared before fmts
  std::vector<std::unique_ptr<MoeFormat>> fmts;
  std::vector<int> layer_fmt;  // [num_layers], index into fmts
  int rb = kRowBlock;           // row_block() of the current task's format
  int n_iblk = 0, n_hblk = 0;  // rb blocks per intermediate row and per output row

  // Pinned IO shared by every layer and batch size: [kSlots * max_tokens, H] x and y, [kSlots * max_tokens, top_k] ids and w.
  // Slot s holds rows [s * max_tokens, (s + 1) * max_tokens): a task on slot 1 can be published while one on slot 0 runs.
  static constexpr int kSlots = 2;
  const bf16_t* const io_x0;
  const int32_t* const io_ids0;
  const float* const io_w0;
  bf16_t* const io_y0;
  // the rows of the slot whose task the pool runs, set by the coordinator before it submits
  const bf16_t* io_x;
  const int32_t* io_ids;
  const float* io_w;
  bf16_t* io_y;
  AlignedVector<bf16_t> g_scratch;   // [max_tokens * top_k * I] intermediate

  // Worker pool.
  int num_threads;
  std::vector<int> core_ids;          // worker tid -> logical CPU to pin to (may be empty)
  std::vector<std::thread> workers;
  std::mutex task_mtx;
  std::condition_variable task_cv;
  std::mutex sync_mtx;
  std::condition_variable sync_cv;

  std::atomic<bool> stop{false};  // spinning workers read it without the lock
  uint64_t cur_gen = 0;
  MoeTask cur_task{};
  // Separate cache lines: idle workers poll `submitted` while busy workers hit p1_next/p2_next.
  alignas(64) std::atomic<uint64_t> submitted{0};
  alignas(64) std::atomic<uint64_t> completed{0};

  // Spin-then-park waits. Off unless the Python wrapper finds spare CPUs for the spinners.
  static constexpr auto kSpinWindow = std::chrono::milliseconds(50);
  std::atomic<bool> spin_wait{false};
  bool coord_pinned = false;

  // Work split of the current task.
  alignas(64) std::atomic<int64_t> p1_next{0};
  std::atomic<int64_t> p2_next{0};
  std::atomic<int64_t> prt_next{0};  // intermediate-row prep phase (prep_inter)
  int64_t p1_total = 0, p2_total = 0, prt_total = 0;
  std::vector<int> p1_route;  // tok * top_k + k of each routed (token, route), in pass-1 order
  std::vector<int> prev_experts;
  int prev_layer = -1;

  // the input prep on the workers, kInChunk blocks per item (FREETOKEN_CPU_MOE_SPLIT_PREP)
  static constexpr int kInChunk = 16;
  int in_blocks = 0;
  int64_t in_total = 0;
  alignas(64) std::atomic<int64_t> in_next{0};

  // flow: one queue, no barriers, see run_flow()
  static constexpr int kFlowMaxTokens = 4;
  int flow_p1 = 0, flow_p2 = 2;
  bool task_flow = false;
  int flow_p1_per = 0, flow_p2_per = 0;
  int64_t flow_p1_total = 0, flow_total = 0;
  alignas(64) std::atomic<int64_t> flow_next{0};
  std::unique_ptr<std::atomic<int>[]> route_left, route_ready, blk_left;
  AlignedVector<float> partial;
  std::atomic<int> done_count{0};
  std::atomic<int> bar_count{0};
  std::atomic<int> bar_sense{0};

  // ---- Flag-based GPU<->CPU handshake (replaces the per-layer cudaLaunchHostFunc pair) ----
  // The GPU rings the doorbell with a task word at submit; this coordinator thread busy-polls it, runs the task on the worker pool, and completes it, which the GPU waits on at sync.
  // This removes the ~2x30-50us host-func dispatch round trips per MoE layer per decode step that otherwise idle the GPU (~6 ms/step on a 75-layer model).
  std::array<std::unique_ptr<GpuSync>, kSlots> gpu_sync;
  std::array<std::unique_ptr<KernelIo>, kSlots> kernel_io;  // replaces the copies around the doorbell once enable_kernel_io() runs
  int last_slot = kSlots - 1;
  int due_slot = 0;  // the next while the slots alternate, else the same one again
  std::thread coord_thread;
  std::atomic<bool> coord_stop{false};
  std::atomic<int64_t> flag_served{0};  // completed doorbells (tests, watchdog)
  std::map<uint64_t, HostTask> host_tasks;  // host-function userData, keyed by doorbell word (graph-stable)

  // Cache prefetch of the next layer's predicted CPU experts, run on the pool between layers.
  static constexpr int kMaxHint = 32;
  const size_t kPfChunk = [] {
    const char* s = getenv("FREETOKEN_CPU_MOE_PF_CHUNK_KB");
    return static_cast<size_t>((s && s[0]) ? std::max(1, atoi(s)) : 64) << 10;
  }();
  const int32_t* hint = nullptr;  // [num_layers][2 + kMaxHint]: stamp, count, expert ids
  std::vector<int32_t> hint_seen;
  bool cur_prefetch = false;
  bool pf_inflight = false;
  int pf_mode = 1;  // 0 load, 1 prefetcht1, 2 prefetcht2, 3 prefetcht0
  std::vector<const uint8_t*> pf_ptr;
  std::vector<size_t> pf_len;
  std::vector<int64_t> pf_first;
  alignas(64) std::atomic<int64_t> pf_next{0};
  int64_t pf_total = 0;
  std::atomic<bool> pf_abort{false};
  std::atomic<int64_t> pf_tasks{0}, pf_bytes{0}, pf_aborted{0};
  std::atomic<uint64_t> pf_sink{0};

  // per-task phase profile, field layout in profile_layout()
  static constexpr int kProfTask = 10, kProfThread = 10;
  std::vector<int64_t> prof;
  int64_t prof_cap = 0;
  std::atomic<int64_t> prof_n{0};
  int64_t prof_cur = -1;
  std::atomic<bool> prof_on{false};

  // copy-engine demand copy into GPU slots, see moe/demand_copy.py
  std::atomic<const int32_t*> ce_rows{nullptr};  // [rows][width]: seq, layer, n, 0, slots, experts, gens
  int ce_nrows = 0;
  int ce_width = 0, ce_maxb = 0;
  uint64_t ce_ready = 0, ce_ack = 0;  // device uint32 ready[cache_size], ack[rows]
  std::vector<std::vector<uint64_t>> ce_src, ce_dst, ce_feat;  // [layer][bank], [pool][bank], [layout][bank]
  std::vector<int> ce_pool, ce_layout;
  std::vector<int64_t> ce_pool_base;
  std::vector<int32_t> ce_seen;
  cudaStream_t ce_stream = nullptr;
  int ce_device = 0;
  cuWrite32_fn ce_write32 = nullptr;
  std::mutex ce_mtx;
  // the coordinator must never enter the driver: a launch blocked on the stalled GPU holds its lock
  std::thread ce_thread;
  std::atomic<bool> ce_stop{false};
  std::atomic<int64_t> ce_rows_served{0}, ce_experts{0}, ce_bytes{0}, ce_issue_ns{0};

  CpuMoeExecutor(int num_threads_, int num_layers_, int num_experts_, int top_k_,
                 int hidden_size, int inter_size, int max_tokens_, int activation_id,
                 int apply_router_weight_on_input, std::vector<FormatSpec> formats,
                 std::vector<int> layer_formats, double swiglu_alpha_, double swiglu_limit_,
                 std::vector<int> core_ids_, uintptr_t x_ptr, uintptr_t ids_ptr, uintptr_t w_ptr,
                 uintptr_t y_ptr)
      : num_layers(num_layers_),
        num_experts(num_experts_),
        apply_on_input(apply_router_weight_on_input != 0),
        params{top_k_, max_tokens_, hidden_size, inter_size, activation_id, static_cast<float>(swiglu_alpha_),
               static_cast<float>(swiglu_limit_)},
        io_x0(reinterpret_cast<const bf16_t*>(x_ptr)),
        io_ids0(reinterpret_cast<const int32_t*>(ids_ptr)),
        io_w0(reinterpret_cast<const float*>(w_ptr)),
        io_y0(reinterpret_cast<bf16_t*>(y_ptr)),
        io_x(io_x0),
        io_ids(io_ids0),
        io_w(io_w0),
        io_y(io_y0),
        num_threads(num_threads_ > 0 ? num_threads_ : 1),
        core_ids(std::move(core_ids_)) {
    if (params.top_k > kMaxTopK) throw std::invalid_argument("CPU MoE supports top_k <= " + std::to_string(kMaxTopK));
    for (auto& [name, banks, options] : formats)
      fmts.push_back(make_format(name, params, FormatArgs{name, std::move(banks), std::move(options)}));
    if (static_cast<int>(layer_formats.size()) != num_layers)
      throw std::invalid_argument("CPU MoE needs the format of every layer");
    for (const int f : layer_formats)
      if (f < 0 || f >= static_cast<int>(fmts.size())) throw std::out_of_range("CPU MoE layer format index out of range");
    layer_fmt = std::move(layer_formats);
    for (const auto& f : fmts)
      if (f->row_block() < 1 || f->row_block() > kMaxRowBlock)
        throw std::invalid_argument("CPU MoE format row block out of range");
    set_row_block(kRowBlock);
    g_scratch.assign(static_cast<size_t>(params.max_tokens) * params.top_k * params.I, 0);
    p1_route.assign(static_cast<size_t>(params.max_tokens) * params.top_k, 0);
    const char* fs = getenv("FREETOKEN_CPU_MOE_FLOW");
    flow_p1 = (fs && fs[0]) ? std::clamp(atoi(fs), 0, 4) : 1;
    if (flow_p1 > 0) {
      if (const char* s2 = getenv("FREETOKEN_CPU_MOE_FLOW_P2")) flow_p2 = std::clamp(atoi(s2), 1, 8);
      const size_t tokens = std::min(params.max_tokens, kFlowMaxTokens), routes = tokens * params.top_k;
      route_left = std::make_unique<std::atomic<int>[]>(routes);
      route_ready = std::make_unique<std::atomic<int>[]>(routes);
      blk_left = std::make_unique<std::atomic<int>[]>(tokens * params.H);
      partial.assign(routes * params.H, 0.0f);
    }
    for (int t = 0; t < num_threads; ++t)
      workers.emplace_back([this, t] { worker_loop(t); });
  }

  void set_row_block(int r) {
    rb = r;
    n_iblk = (params.I + rb - 1) / rb;
    n_hblk = (params.H + rb - 1) / rb;
  }

  void pin_self(int tid) {
    if (!core_ids.empty()) pin_this_thread(core_ids[tid % static_cast<int>(core_ids.size())]);
  }

  ~CpuMoeExecutor() {
    coord_stop.store(true);
    if (coord_thread.joinable()) coord_thread.join();
    ce_stop.store(true);
    if (ce_thread.joinable()) ce_thread.join();
    if (ce_stream != nullptr) {
      cudaStreamSynchronize(ce_stream);
      cudaStreamDestroy(ce_stream);
    }
    {
      std::lock_guard<std::mutex> lk(task_mtx);
      stop = true;
    }
    task_cv.notify_all();
    for (auto& th : workers)
      if (th.joinable()) th.join();
  }

  MoeTask task(int layer_id, int num_tokens) const {
    if (layer_id < 0 || layer_id >= num_layers || num_tokens < 1 || num_tokens > params.max_tokens)
      throw std::out_of_range("CPU MoE task (layer " + std::to_string(layer_id) + ", " +
                              std::to_string(num_tokens) + " tokens) is outside the executor's layers or IO");
    return MoeTask{static_cast<uint64_t>(layer_id), static_cast<uint64_t>(num_tokens)};
  }

  std::string isa_name() const {
    std::vector<std::string> names;
    for (const auto& f : fmts)
      if (const std::string n = f->isa_name(); std::find(names.begin(), names.end(), n) == names.end()) names.push_back(n);
    std::string out;
    for (const auto& n : names) out += (out.empty() ? "" : ",") + n;
    return out;
  }

  const MoeFormat& format_of(const MoeTask& t) const { return *fmts[layer_fmt[t.layer]]; }

  // Whether any route of the task names an expert; do_pass1 and do_pass2 skip the others.
  bool has_routes(const MoeTask& t) const {
    const int64_t n = static_cast<int64_t>(t.num_tokens) * params.top_k;
    for (int64_t i = 0; i < n; ++i)
      if (io_ids[i] >= 0 && io_ids[i] < num_experts) return true;
    return false;
  }

  void barrier(int& local_sense) {
    local_sense ^= 1;
    if (bar_count.fetch_add(1) + 1 == num_threads) {
      bar_count.store(0);
      bar_sense.store(local_sense);
    } else {
      for (int v; (v = bar_sense.load()) != local_sense;) wait_change(bar_sense, v);
    }
  }

  // Pass 1 item p = one rb-row block of one routed (token, route)'s intermediate row.
  bool do_pass1(const MoeTask& t, int64_t p) {
    const int top_k = params.top_k, H = params.H, I = params.I;
    const int64_t ib = p % n_iblk;
    const int tk = p1_route[static_cast<size_t>(p / n_iblk)];
    const int k = tk % top_k;
    const int tok = tk / top_k;
    const int e = io_ids[static_cast<size_t>(tok) * top_k + k];
    if (e < 0 || e >= num_experts) return false;
    const float w_in = apply_on_input ? io_w[static_cast<size_t>(tok) * top_k + k] : 1.0f;
    const int i0 = static_cast<int>(ib) * rb;
    const int i1 = std::min(I, i0 + rb);
    format_of(t).gate_up(static_cast<int>(t.layer), e, tok, i0, i1, io_x + (size_t)tok * H, w_in,
                         g_scratch.data() + ((size_t)tok * top_k + k) * I);
    return true;
  }

  // Pass 2 item p = one rb-row block of one token's output row, summed over its routes.
  void do_pass2(const MoeTask& t, int64_t p) {
    const int top_k = params.top_k, H = params.H, I = params.I;
    const int64_t hb = p % n_hblk;
    const int tok = static_cast<int>(p / n_hblk);
    const int h0 = static_cast<int>(hb) * rb;
    const int h1 = std::min(H, h0 + rb);
    DownRoute routes[kMaxTopK];
    float w_out[kMaxTopK];
    int n = 0;
    for (int k = 0; k < top_k; ++k) {
      const int e = io_ids[static_cast<size_t>(tok) * top_k + k];
      if (e < 0 || e >= num_experts) continue;
      const size_t r = (size_t)tok * top_k + k;
      routes[n] = {e, static_cast<int64_t>(r), g_scratch.data() + r * I};
      w_out[n] = apply_on_input ? 1.0f : io_w[r];
      ++n;
    }
    float part[kMaxTopK * kMaxRowBlock];
    const MoeFormat& f = format_of(t);
    if (n > 0) f.down(static_cast<int>(t.layer), h0, h1, routes, n, part);
    const bool rounds_routes = f.rounds_routes();
    bf16_t* y_row = io_y + (size_t)tok * H;
    for (int h = h0; h < h1; ++h) {
      float acc = 0.0f;
      for (int j = 0; j < n; ++j) {
        const float v = part[j * rb + h - h0] * w_out[j];
        acc += rounds_routes ? bf16_to_f32(f32_to_bf16(v)) : v;
      }
      y_row[h] = f32_to_bf16(acc);
    }
  }

  void touch(const uint8_t* p, size_t n) {
#if CPU_MOE_X86
    if (pf_mode == 1) {
      for (size_t i = 0; i < n; i += 64) _mm_prefetch(reinterpret_cast<const char*>(p + i), _MM_HINT_T1);
      return;
    }
    if (pf_mode == 2) {
      for (size_t i = 0; i < n; i += 64) _mm_prefetch(reinterpret_cast<const char*>(p + i), _MM_HINT_T2);
      return;
    }
    if (pf_mode == 3) {
      for (size_t i = 0; i < n; i += 64) _mm_prefetch(reinterpret_cast<const char*>(p + i), _MM_HINT_T0);
      return;
    }
#endif
    uint64_t acc = 0;
    for (size_t i = 0; i < n; i += 64) acc += *reinterpret_cast<const volatile uint64_t*>(p + i);
    pf_sink.fetch_add(acc, std::memory_order_relaxed);
  }

  void do_prefetch() {
    const auto ranges = static_cast<int64_t>(pf_ptr.size());
    for (;;) {
      if (pf_abort.load(std::memory_order_relaxed)) return;
      const int64_t c = pf_next.fetch_add(1, std::memory_order_relaxed);
      if (c >= pf_total) return;
      const int64_t r = std::upper_bound(pf_first.begin(), pf_first.begin() + ranges, c) - pf_first.begin() - 1;
      const size_t off = static_cast<size_t>(c - pf_first[r]) * kPfChunk;
      touch(pf_ptr[r] + off, std::min(kPfChunk, pf_len[r] - off));
    }
  }

  void set_prefetch_hint(uintptr_t ptr, int mode) {
    hint = reinterpret_cast<const int32_t*>(ptr);
    hint_seen.assign(num_layers, -1);
    pf_mode = mode;
  }

  // false when the hint of `layer` is stale or names nothing
  bool submit_prefetch(int layer) {
    if (!hint || layer < 0 || layer >= num_layers) return false;
    const int32_t* h = hint + static_cast<size_t>(layer) * (2 + kMaxHint);
    const int32_t stamp = __atomic_load_n(h, __ATOMIC_ACQUIRE);
    if (stamp == hint_seen[layer]) return false;
    hint_seen[layer] = stamp;
    const int n = std::min<int>(__atomic_load_n(h + 1, __ATOMIC_RELAXED), kMaxHint);
    const MoeFormat& fmt = *fmts[layer_fmt[layer]];
    pf_ptr.clear();
    pf_len.clear();
    pf_first.clear();
    int64_t chunks = 0;
    for (int i = 0; i < n; ++i) {
      const int e = __atomic_load_n(h + 2 + i, __ATOMIC_RELAXED);
      if (e < 0 || e >= num_experts) continue;
      const void* ptrs[kMaxRanges];
      size_t lens[kMaxRanges];
      const int k = fmt.expert_ranges(layer, e, ptrs, lens);
      for (int j = 0; j < k; ++j) {
        pf_first.push_back(chunks);
        pf_ptr.push_back(static_cast<const uint8_t*>(ptrs[j]));
        pf_len.push_back(lens[j]);
        chunks += static_cast<int64_t>((lens[j] + kPfChunk - 1) / kPfChunk);
        pf_bytes.fetch_add(static_cast<int64_t>(lens[j]), std::memory_order_relaxed);
      }
    }
    if (chunks == 0) return false;
    pf_total = chunks;
    pf_next.store(0, std::memory_order_relaxed);
    pf_abort.store(false, std::memory_order_relaxed);
    done_count.store(0, std::memory_order_relaxed);
    {
      std::lock_guard<std::mutex> lk(task_mtx);
      cur_prefetch = true;
      ++cur_gen;
      submitted.store(cur_gen, std::memory_order_release);
    }
    task_cv.notify_all();
    pf_tasks.fetch_add(1, std::memory_order_relaxed);
    return true;
  }

  // a real task takes the pool: stop the prefetch
  void retire_prefetch(bool spin) {
    if (!pf_inflight) return;
    if (pf_next.load(std::memory_order_relaxed) < pf_total) {
      pf_abort.store(true, std::memory_order_relaxed);
      pf_aborted.fetch_add(1, std::memory_order_relaxed);
    }
    sync(spin);
    pf_inflight = false;
  }

  void prefetch_now(int layer) {
    if (submit_prefetch(layer)) sync();
  }

  void ce_setup(uintptr_t rows, int width, int maxb, uintptr_t ready, uintptr_t ack, std::vector<std::vector<uint64_t>> src,
                std::vector<std::vector<uint64_t>> dst, std::vector<std::vector<uint64_t>> feat, std::vector<int> pool_of_layer,
                std::vector<int> layer_layout, std::vector<int64_t> pool_base, int device) {
    std::lock_guard<std::mutex> lk(ce_mtx);
    ce_write32 = resolve_write32();
    if (ce_write32 == nullptr) throw std::runtime_error("copy-engine copies need cuStreamWriteValue32");
    if (src.empty() || pool_of_layer.size() != src.size() || layer_layout.size() != src.size())
      throw std::invalid_argument("copy-engine rows need a source, a pool and a layout each");
    ce_device = device;
    if (ce_stream == nullptr) {
      cudaSetDevice(device);
      if (cudaStreamCreateWithFlags(&ce_stream, cudaStreamNonBlocking) != cudaSuccess)
        throw std::runtime_error("copy-engine copy stream");
    }
    ce_width = width;
    ce_maxb = maxb;
    ce_ready = ready;
    ce_ack = ack;
    ce_src = std::move(src);
    ce_dst = std::move(dst);
    ce_feat = std::move(feat);
    ce_pool = std::move(pool_of_layer);
    ce_layout = std::move(layer_layout);
    ce_pool_base = std::move(pool_base);
    ce_nrows = static_cast<int>(ce_src.size());
    ce_seen.assign(ce_nrows, 0);
    ce_rows.store(reinterpret_cast<const int32_t*>(rows), std::memory_order_release);
    if (!ce_thread.joinable()) ce_thread = std::thread([this] { ce_loop(); });
  }

  // spins while rows keep coming, dozes like the coordinator once decode stops
  void ce_loop() {
    using clock = std::chrono::steady_clock;
    constexpr auto kHotWindow = std::chrono::milliseconds(50);
    auto last_active = clock::now();
    unsigned empty_polls = 0;
    int64_t sleep_us = 50;
    bool dozing = false;
    while (!ce_stop.load(std::memory_order_relaxed)) {
      if (ce_poll()) {
        last_active = clock::now();
        empty_polls = 0;
        sleep_us = 50;
        dozing = false;
        continue;
      }
      if (!dozing) {
        if ((++empty_polls & 1023u) != 0 || clock::now() - last_active < kHotWindow) {
          cpu_relax();
          continue;
        }
        dozing = true;
      }
      std::this_thread::sleep_for(std::chrono::microseconds(sleep_us));
      sleep_us = std::min<int64_t>(sleep_us * 2, 500);
    }
  }

  // stop serving rows; the caller drains first
  void ce_detach() {
    std::lock_guard<std::mutex> lk(ce_mtx);
    ce_rows.store(nullptr, std::memory_order_release);
  }

  // issue the copies of every row with a new seq; whether any was found
  bool ce_poll() {
    if (ce_rows.load(std::memory_order_relaxed) == nullptr) return false;
    std::unique_lock<std::mutex> lk(ce_mtx, std::try_to_lock);
    if (!lk.owns_lock()) return false;
    const int32_t* rows = ce_rows.load(std::memory_order_acquire);
    if (rows == nullptr) return false;
    static thread_local int bound_device = -1;
    bool served = false;
    for (int t = 0; t < ce_nrows; ++t) {
      const int32_t* row = rows + static_cast<size_t>(t) * ce_width;
      const int32_t seq = __atomic_load_n(row, __ATOMIC_ACQUIRE);
      if (seq == ce_seen[t]) continue;
      const auto t0 = std::chrono::steady_clock::now();
      if (bound_device != ce_device) {
        cudaSetDevice(ce_device);
        bound_device = ce_device;
      }
      const int n = std::min<int>(__atomic_load_n(row + 2, __ATOMIC_RELAXED), ce_maxb);
      const int pool = ce_pool[t];
      const auto& feat = ce_feat[ce_layout[t]];
      for (int i = 0; i < n; ++i) {
        const int64_t slot = __atomic_load_n(row + 4 + i, __ATOMIC_RELAXED);
        const int64_t e = __atomic_load_n(row + 4 + ce_maxb + i, __ATOMIC_RELAXED);
        const auto gen = static_cast<unsigned int>(__atomic_load_n(row + 4 + 2 * ce_maxb + i, __ATOMIC_RELAXED));
        for (size_t b = 0; b < feat.size(); ++b) {
          if (feat[b] == 0 || ce_src[t][b] == 0) continue;
          cudaMemcpyAsync(reinterpret_cast<void*>(ce_dst[pool][b] + slot * feat[b]),
                          reinterpret_cast<const void*>(ce_src[t][b] + e * feat[b]), feat[b], cudaMemcpyDefault, ce_stream);
          ce_bytes.fetch_add(static_cast<int64_t>(feat[b]), std::memory_order_relaxed);
        }
        ce_write32(ce_stream, ce_ready + 4ull * static_cast<unsigned long long>(ce_pool_base[pool] + slot), gen, 0);
        ce_experts.fetch_add(1, std::memory_order_relaxed);
      }
      ce_seen[t] = seq;
      ce_write32(ce_stream, ce_ack + 4ull * static_cast<unsigned long long>(t), static_cast<unsigned int>(seq), 0);
      served = true;
      ce_rows_served.fetch_add(1, std::memory_order_relaxed);
      ce_issue_ns.fetch_add(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count(),
                            std::memory_order_relaxed);
    }
    return served;
  }

  // callers synchronize the compute stream first
  void ce_drain() {
    for (;;) {
      bool fresh = false;
      {
        std::lock_guard<std::mutex> lk(ce_mtx);
        const int32_t* rows = ce_rows.load(std::memory_order_acquire);
        if (rows == nullptr) break;
        for (int t = 0; t < ce_nrows && !fresh; ++t)
          fresh = __atomic_load_n(rows + static_cast<size_t>(t) * ce_width, __ATOMIC_ACQUIRE) != ce_seen[t];
      }
      if (!fresh) break;
      ce_poll();
    }
    std::lock_guard<std::mutex> lk(ce_mtx);
    if (ce_stream != nullptr) cudaStreamSynchronize(ce_stream);
  }

  static int64_t now_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
  }
  static bool shared_first() {
    static const bool on = [] {
      const char* s = getenv("FREETOKEN_CPU_MOE_SHARED_FIRST");
      return s && s[0] && s[0] != '0';
    }();
    return on;
  }
  static bool env_on(const char* name) {
    const char* s = getenv(name);
    return s && s[0] && s[0] != '0';
  }
  const bool split_prep_on = env_on("FREETOKEN_CPU_MOE_SPLIT_PREP");
  bool split_prep() const { return split_prep_on; }
  static bool prefetch_idle_only() {
    static const bool on = [] {
      const char* s = getenv("FREETOKEN_CPU_MOE_PREFETCH_IDLE");
      return s && s[0] && s[0] != '0';
    }();
    return on;
  }
  bool rung_any() const {
    for (const auto& g : gpu_sync)
      if (__atomic_load_n(static_cast<const int64_t*>(g->poll_address()), __ATOMIC_ACQUIRE) != 0) return true;
    return false;
  }
  // worker w takes row blocks w, w + num_threads, ...: the same weight tiles every task
  const bool static_split_on = env_on("FREETOKEN_CPU_MOE_STATIC");
  bool static_split() const { return static_split_on; }
  int prof_task_fields() const { return kProfTask + params.max_tokens * params.top_k; }
  int prof_stride() const { return prof_task_fields() + kProfThread * num_threads; }

  // -1 when the profile is off or full
  int64_t prof_begin(const MoeTask& t, int slot, int64_t ring, int64_t retired) {
    const int64_t i = prof_n.load(std::memory_order_relaxed);
    if (!prof_on.load(std::memory_order_relaxed) || i >= prof_cap) return -1;
    int64_t* r = prof.data() + i * prof_stride();
    std::fill(r, r + prof_stride(), 0);
    const int n = static_cast<int>(t.num_tokens) * params.top_k;
    int routes = 0, distinct = 0;
    for (int j = 0; j < params.max_tokens * params.top_k; ++j) {
      const int e = j < n ? io_ids[j] : -1;
      r[kProfTask + j] = e;
      if (e < 0 || e >= num_experts) continue;
      ++routes;
      distinct += std::find(io_ids, io_ids + j, e) == io_ids + j;
    }
    r[0] = ring;
    r[1] = retired;
    r[5] = t.layer;
    r[6] = t.num_tokens;
    r[7] = slot;
    r[8] = routes;
    r[9] = distinct;
    return i;
  }

  void prof_mark(int field) {
    if (prof_cur >= 0) prof[prof_cur * prof_stride() + field] = now_ns();
  }

  void prof_end() {
    if (prof_cur < 0) return;
    prof_mark(4);
    prof_n.store(prof_cur + 1, std::memory_order_release);
    prof_cur = -1;
  }

  void run_routed(const MoeTask& t, int slot, int64_t ring, bool spin) {
    prof_cur = prof_begin(t, slot, ring, ring != 0 ? now_ns() : 0);
    submit(t);
    prof_mark(2);
    sync(spin);
    prof_mark(3);
  }

  // gate_up items, then down items once their route is prepped
  void run_flow(const MoeTask& t, int64_t* tp) {
    MoeFormat& f = *fmts[layer_fmt[t.layer]];
    const int top_k = params.top_k, H = params.H, I = params.I, layer = static_cast<int>(t.layer);
    bool down = false;
    for (;;) {
      const int64_t q = flow_next.fetch_add(1, std::memory_order_relaxed);
      if (q >= flow_total) break;
      const int64_t a = tp ? now_ns() : 0;
      if (q < flow_p1_total) {
        const int j = static_cast<int>(q / flow_p1_per);
        const int i0 = static_cast<int>(q % flow_p1_per) * flow_p1 * rb;
        const int n = std::min(flow_p1, (I - i0 + rb - 1) / rb);
        const int r = p1_route[j];
        const float w_in = apply_on_input ? io_w[r] : 1.0f;
        f.gate_up_tiles(layer, io_ids[r], r / top_k, i0, n, w_in, g_scratch.data() + (size_t)r * I);
        if (route_left[j].fetch_sub(1, std::memory_order_acq_rel) == 1) {
          if (f.preps_inter()) f.prep_inter(r, g_scratch.data() + (size_t)r * I);
          route_ready[j].store(1, std::memory_order_release);
        }
        if (tp) {
          tp[6] += now_ns() - a;
          ++tp[8];
        }
        continue;
      }
      if (!down && tp) tp[1] = tp[2] = tp[3] = tp[4] = a;
      down = true;
      const int64_t q2 = q - flow_p1_total;
      const int j = static_cast<int>(q2 / flow_p2_per), b = static_cast<int>(q2 % flow_p2_per);
      for (int v; (v = route_ready[j].load(std::memory_order_acquire)) == 0;) wait_change(route_ready[j], v);
      const int r = p1_route[j], tok = r / top_k;
      const int h0 = b * flow_p2 * rb;
      const int n = std::min(flow_p2, (H - h0 + rb - 1) / rb);
      f.down_tiles(layer, h0, n, DownRoute{io_ids[r], r, g_scratch.data() + (size_t)r * I}, partial.data() + (size_t)r * H + h0);
      if (blk_left[tok * flow_p2_per + b].fetch_sub(1, std::memory_order_acq_rel) == 1) flow_sum(f, tok, h0, std::min(H, h0 + n * rb));
      if (tp) {
        tp[7] += now_ns() - a;
        ++tp[9];
      }
    }
    if (!down && tp) tp[1] = tp[2] = tp[3] = tp[4] = now_ns();
  }

  // do_pass2's sum over a token's routes, from their down outputs in `partial`
  void flow_sum(const MoeFormat& f, int tok, int h0, int h1) {
    const int top_k = params.top_k, H = params.H;
    const bool rounds_routes = f.rounds_routes();
    bf16_t* y_row = io_y + (size_t)tok * H;
    for (int h = h0; h < h1; ++h) {
      float acc = 0.0f;
      for (int k = 0; k < top_k; ++k) {
        const size_t r = (size_t)tok * top_k + k;
        const int e = io_ids[r];
        if (e < 0 || e >= num_experts) continue;
        const float v = partial[r * H + h] * (apply_on_input ? 1.0f : io_w[r]);
        acc += rounds_routes ? bf16_to_f32(f32_to_bf16(v)) : v;
      }
      y_row[h] = f32_to_bf16(acc);
    }
  }

  void run_task_body(const MoeTask& t, int tid) {
    if (cur_prefetch) {
      do_prefetch();
      return;
    }
    int64_t* tp = prof_cur >= 0 ? prof.data() + prof_cur * prof_stride() + prof_task_fields() + kProfThread * tid : nullptr;
    if (tp) tp[0] = now_ns();
    int local_sense = 0;
    if (in_total > 0) {
      MoeFormat& f = *fmts[layer_fmt[t.layer]];
      const int per = (in_blocks + kInChunk - 1) / kInChunk;
      for (;;) {
        const int64_t c = in_next.fetch_add(1, std::memory_order_relaxed);
        if (c >= in_total) break;
        const int tok = static_cast<int>(c / per), b0 = static_cast<int>(c % per) * kInChunk;
        f.prep_input_range(tok, io_x + (size_t)tok * params.H, b0, std::min(in_blocks, b0 + kInChunk));
      }
      barrier(local_sense);
    }
    if (task_flow) {
      run_flow(t, tp);
      if (tp) tp[5] = now_ns();
      return;
    }
    const auto pass1 = [&](int64_t p) {
      if (tp) {
        const int64_t a = now_ns();
        if (do_pass1(t, p)) {
          tp[6] += now_ns() - a;
          ++tp[8];
        }
      } else {
        do_pass1(t, p);
      }
    };
    const auto pass2 = [&](int64_t p) {
      if (tp) {
        const int64_t a = now_ns();
        do_pass2(t, p);
        tp[7] += now_ns() - a;
        ++tp[9];
      } else {
        do_pass2(t, p);
      }
    };
    const bool own = static_split();
    if (own) {
      for (int64_t p = 0; p < p1_total; p += n_iblk)
        for (int64_t ib = tid; ib < n_iblk; ib += num_threads) pass1(p + ib);
    } else {
      for (;;) {
        const int64_t p = p1_next.fetch_add(1, std::memory_order_relaxed);
        if (p >= p1_total) break;
        pass1(p);
      }
    }
    if (tp) tp[1] = now_ns();
    barrier(local_sense);
    if (tp) tp[2] = now_ns();
    // Prepare the intermediate rows (per token,route) for the down GEMV, see prep_inter.
    // Needs all of pass1 done (a full row spans every iblk).
    if (prt_total > 0) {
      MoeFormat& f = *fmts[layer_fmt[t.layer]];
      for (;;) {
        const int64_t i = prt_next.fetch_add(1, std::memory_order_relaxed);
        if (i >= prt_total) break;
        const int r = p1_route[static_cast<size_t>(i)];
        f.prep_inter(r, g_scratch.data() + (size_t)r * params.I);
      }
      if (tp) tp[3] = now_ns();
      barrier(local_sense);
    } else if (tp) {
      tp[3] = tp[2];
    }
    if (tp) tp[4] = now_ns();
    if (own) {
      for (int64_t p = 0; p < p2_total; p += n_hblk)
        for (int64_t hb = tid; hb < n_hblk; hb += num_threads) pass2(p + hb);
    } else {
      for (;;) {
        const int64_t p = p2_next.fetch_add(1, std::memory_order_relaxed);
        if (p >= p2_total) break;
        pass2(p);
      }
    }
    if (tp) tp[5] = now_ns();
  }

  // Tasks arrive once per MoE layer, sub-ms apart. Spinning skips a futex wakeup per worker per layer; parking after kSpinWindow keeps an idle server off the CPU.
  void worker_loop(int tid) {
    pin_self(tid);
    uint64_t my_gen = 0;
    using wclk = std::chrono::steady_clock;
    auto last_active = wclk::now();
    for (;;) {
      uint64_t gen = submitted.load(std::memory_order_acquire);
      if (gen == my_gen && spin_wait.load(std::memory_order_relaxed)) {
        unsigned polls = 0;
        while (gen == my_gen) {
          wait_change(submitted, my_gen);
          if (stop.load(std::memory_order_relaxed)) return;
          if ((++polls & 1023u) == 0 && wclk::now() - last_active > kSpinWindow) break;
          gen = submitted.load(std::memory_order_acquire);
        }
      }
      if (gen == my_gen) {
        std::unique_lock<std::mutex> lk(task_mtx);
        task_cv.wait(lk, [&] { return stop.load() || cur_gen != my_gen; });
        if (stop.load()) return;
        gen = cur_gen;
      }
      // submit() writes cur_task before the release store to `submitted`.
      my_gen = gen;
      run_task_body(cur_task, tid);
      last_active = wclk::now();
      if (done_count.fetch_add(1) + 1 == num_threads) {
        completed.store(my_gen, std::memory_order_release);
        {
          std::lock_guard<std::mutex> lk(sync_mtx);
        }
        sync_cv.notify_all();
      }
    }
  }

  void submit(MoeTask t) {
    const int num_tokens = static_cast<int>(t.num_tokens);
    MoeFormat& fmt = *fmts[layer_fmt[t.layer]];
    set_row_block(fmt.row_block());
    int routed = 0;
    for (int r = 0; r < num_tokens * params.top_k; ++r)
      if (io_ids[r] >= 0 && io_ids[r] < num_experts) p1_route[routed++] = r;
    const auto in_prev = [&](int r) {
      return std::find(prev_experts.begin(), prev_experts.end(), io_ids[r]) != prev_experts.end();
    };
    if (shared_first() && static_cast<int>(t.layer) == prev_layer)
      std::stable_partition(p1_route.begin(), p1_route.begin() + routed, in_prev);
    prev_layer = static_cast<int>(t.layer);
    prev_experts.clear();
    for (int i = 0; i < routed; ++i) prev_experts.push_back(io_ids[p1_route[i]]);
    p1_total = static_cast<int64_t>(routed) * n_iblk;
    p2_total = static_cast<int64_t>(num_tokens) * n_hblk;
    prt_total = fmt.preps_inter() ? routed : 0;
    task_flow = flow_p1 > 0 && fmt.has_tiles() && num_tokens <= kFlowMaxTokens;
    if (task_flow) {
      flow_p1_per = (n_iblk + flow_p1 - 1) / flow_p1;
      flow_p2_per = (n_hblk + flow_p2 - 1) / flow_p2;
      flow_p1_total = static_cast<int64_t>(routed) * flow_p1_per;
      flow_total = flow_p1_total + static_cast<int64_t>(routed) * flow_p2_per;
      flow_next.store(0, std::memory_order_relaxed);
      for (int j = 0; j < routed; ++j) {
        route_left[j].store(flow_p1_per, std::memory_order_relaxed);
        route_ready[j].store(0, std::memory_order_relaxed);
      }
      for (int tok = 0; tok < num_tokens; ++tok) {
        int n = 0;
        for (int k = 0; k < params.top_k; ++k) {
          const int e = io_ids[tok * params.top_k + k];
          n += e >= 0 && e < num_experts;
        }
        for (int b = 0; b < flow_p2_per; ++b) blk_left[tok * flow_p2_per + b].store(n, std::memory_order_relaxed);
        if (n == 0) std::memset(io_y + (size_t)tok * params.H, 0, sizeof(bf16_t) * params.H);
      }
    }
    p1_next.store(0, std::memory_order_relaxed);
    p2_next.store(0, std::memory_order_relaxed);
    prt_next.store(0, std::memory_order_relaxed);
    done_count.store(0, std::memory_order_relaxed);
    bar_count.store(0, std::memory_order_relaxed);
    bar_sense.store(0, std::memory_order_relaxed);
    // Prepare the per-token input once, up front.
    // Single-threaded: tiny for decode, and done before the workers are woken below.
    in_blocks = split_prep() ? fmt.prep_input_blocks() : 0;
    in_total = in_blocks > 0 ? static_cast<int64_t>(num_tokens) * ((in_blocks + kInChunk - 1) / kInChunk) : 0;
    in_next.store(0, std::memory_order_relaxed);
    if (in_total == 0)
      for (int tok = 0; tok < num_tokens; ++tok) fmt.prep_input(tok, io_x + (size_t)tok * params.H);
    {
      std::lock_guard<std::mutex> lk(task_mtx);
      cur_task = t;
      cur_prefetch = false;
      ++cur_gen;
      submitted.store(cur_gen, std::memory_order_release);
    }
    task_cv.notify_all();
  }

  // Only a coordinator on its own core may spin: host-func callbacks and eager calls share cores with the workers.
  void sync(bool spin = false) {
    const uint64_t target = submitted.load(std::memory_order_acquire);
    if (spin) {
      using sclk = std::chrono::steady_clock;
      const auto t0 = sclk::now();
      unsigned polls = 0;
      for (uint64_t c; (c = completed.load(std::memory_order_acquire)) < target;) {
        wait_change(completed, c);
        if ((++polls & 1023u) == 0 && sclk::now() - t0 > kSpinWindow) break;
      }
      if (completed.load(std::memory_order_acquire) >= target) return;
    }
    std::unique_lock<std::mutex> lk(sync_mtx);
    sync_cv.wait(lk, [&] { return completed.load(std::memory_order_acquire) >= target; });
  }

  void submit_with_cuda_stream(uintptr_t stream, int layer_id, int num_tokens) {
    const MoeTask t = task(layer_id, num_tokens);
    HostTask& h = host_tasks.try_emplace(to_word(t), HostTask{this, t}).first->second;
    cudaLaunchHostFunc(reinterpret_cast<cudaStream_t>(stream), &CpuMoeExecutor::submit_cb, &h);
  }

  void sync_with_cuda_stream(uintptr_t stream) {
    cudaLaunchHostFunc(reinterpret_cast<cudaStream_t>(stream), &CpuMoeExecutor::sync_cb, this);
  }

  // Start the busy-poll coordinator over the doorbell. ``pin_core`` >= 0
  // pins the coordinator to that logical CPU (the worker auto-sizing reserves it), so its
  // polling never migrates onto / contends with a GEMV worker's core.
  void use_slot(int slot) {
    const size_t rows = static_cast<size_t>(slot) * params.max_tokens;
    io_x = io_x0 + rows * params.H;
    io_ids = io_ids0 + rows * params.top_k;
    io_w = io_w0 + rows * params.top_k;
    io_y = io_y0 + rows * params.H;
  }

  void start_flag_coordinator(int pin_core) {
    for (auto& g : gpu_sync) g = std::make_unique<GpuSync>();
    coord_stop.store(false);
    coord_pinned = CPU_MOE_HAS_AFFINITY && pin_core >= 0;
    coord_thread = std::thread([this, pin_core] {
      if (pin_core >= 0) pin_this_thread(pin_core);
      coordinator_loop();
    });
  }

  void flag_signal(uintptr_t stream, int layer_id, int num_tokens, int slot) {
    gpu_sync.at(slot)->gpu_submit(stream, to_word(task(layer_id, num_tokens)));
  }
  void flag_wait(uintptr_t stream, int slot) { gpu_sync.at(slot)->gpu_wait(stream); }
  void enable_kernel_io() {
    for (int s = 0; s < kSlots; ++s) {
      const size_t rows = static_cast<size_t>(s) * params.max_tokens;
      kernel_io[s] = std::make_unique<KernelIo>(gpu_sync[s]->ready_word(), gpu_sync[s]->done_word(), io_x0 + rows * params.H,
                                                io_ids0 + rows * params.top_k, io_w0 + rows * params.top_k, io_y0 + rows * params.H,
                                                params.H, params.top_k);
    }
  }
  void flag_publish(uintptr_t stream, uintptr_t x, uintptr_t ids, uintptr_t w, int layer_id, int num_tokens, int slot) {
    kernel_io.at(slot)->publish(stream, reinterpret_cast<const void*>(x), reinterpret_cast<const int32_t*>(ids),
                                reinterpret_cast<const float*>(w), num_tokens, to_word(task(layer_id, num_tokens)));
  }
  void flag_combine(uintptr_t stream, uintptr_t out, int num_tokens, bool accumulate, int slot) {
    kernel_io.at(slot)->combine(stream, reinterpret_cast<void*>(out), num_tokens, accumulate);
  }
  bool flag_pending() const {
    for (const auto& g : gpu_sync)
      if (g && g->pending()) return true;
    return false;
  }
  // Complete a doorbell the coordinator never answered so the stream can make progress.
  void poison_flag() {
    for (auto& g : gpu_sync)
      if (g) g->cpu_done();
  }

  void coordinator_loop() {
    // Idle backoff (edge-device friendliness): spin hot only while decode traffic is
    // flowing, with a TIME-based hot window (an iteration count varies 2-6x with pause
    // cost, so some configs dozed every token). 50 ms since the last
    // flag comfortably covers intra- and inter-token gaps -- including prefill bursts and
    // --moe-cpu-layers gaps -- so steady decode never
    // sleeps and keeps the ~us-level wakeup. Past it, i.e. the engine is actually idle,
    // escalate to timed sleeps capped at 2 ms: a dozing coordinator costs <0.1% of a
    // core instead of 100%, and the only price is a <=2 ms discovery delay on the FIRST
    // MoE layer after an idle period (irrelevant next to prefill). The clock is sampled
    // every 1024 empty polls (~0.1-1 ms) to keep the hot loop cheap.
    using coord_clock = std::chrono::steady_clock;
    constexpr auto kHotWindow = std::chrono::milliseconds(50);
    constexpr int64_t kSleepCapUs = 2000;
    auto last_active = coord_clock::now();
    unsigned empty_polls = 0;  // unsigned: the hot-phase ++ must not overflow into UB
    int64_t sleep_us = 100;
    bool dozing = false;
    while (!coord_stop.load(std::memory_order_relaxed)) {
      // the slots in turn, starting after the one served last: a slot is rung again only after its task was waited on
      uint64_t word = 0;
      int slot = 0;
      for (int i = 1; i <= kSlots && word == 0; ++i) {
        slot = (last_slot + i) % kSlots;
        word = gpu_sync[slot]->cpu_poll();
      }
      if (word != 0) {
        const int64_t ring = prof_on.load(std::memory_order_relaxed) ? now_ns() : 0;
        due_slot = slot != last_slot ? (slot + 1) % kSlots : slot;
        last_slot = slot;
        use_slot(slot);
        const MoeTask t = from_word(word);
        const bool spin = coord_pinned && spin_wait.load(std::memory_order_relaxed);
        retire_prefetch(spin);
        if (has_routes(t)) {
          run_routed(t, slot, ring, spin);
        } else {
          // What the pool would write for a task without routes, minus its fork and join.
          std::memset(io_y, 0, sizeof(bf16_t) * t.num_tokens * params.H);
        }
        // Release: the workers' y stores are visible before the GPU sees done.
        gpu_sync[slot]->cpu_done();
        prof_end();
        flag_served.fetch_add(1, std::memory_order_relaxed);
        // a rung doorbell would retire the prefetch at once
        if (hint && !(prefetch_idle_only() && rung_any())) pf_inflight = submit_prefetch(static_cast<int>(t.layer) + 1);
        last_active = coord_clock::now();
        empty_polls = 0;
        sleep_us = 100;
        dozing = false;
        continue;
      }
      // Hot phase: pause-spin, consulting the clock only every 1024 empty polls (the
      // amortization must gate the CLOCK, not the sleep -- gating the sleep decision
      // let 1023/1024 idle iterations hot-poll).
      // Doze phase: sleep on EVERY iteration until traffic returns; one wake-up scan
      // (~0.5 us) per 2 ms sleep is ~0.03% duty.
      if (!dozing) {
        if ((++empty_polls & 1023u) != 0 ||
            coord_clock::now() - last_active < kHotWindow) {
#if CPU_MOE_X86
          if (umwait_enabled()) {
            // watch the slot due next; another one is seen at the latest when umwait times out
            umonitor_line(gpu_sync[due_slot]->poll_address());
            bool rung = false;
            for (const auto& g : gpu_sync)
              rung = rung || __atomic_load_n(static_cast<const int64_t*>(g->poll_address()), __ATOMIC_ACQUIRE) != 0;
            if (!rung) umwait_c01();
            continue;
          }
#endif
          cpu_relax();
          continue;
        }
        dozing = true;
      }
      std::this_thread::sleep_for(std::chrono::microseconds(sleep_us));
      sleep_us = std::min<int64_t>(sleep_us * 2, kSleepCapUs);
    }
    // Teardown: complete any in-flight wait so a replay caught mid-shutdown does not stall
    // until the watchdog fires. Runs before the destructor's join() returns, while the doorbell is still alive.
    for (auto& g : gpu_sync) g->cpu_done();
  }

  // Eager (non-graph) path: run one task to completion on the pool.
  void run(int layer_id, int num_tokens) {
    use_slot(0);
    const MoeTask t = task(layer_id, num_tokens);
    if (!prof_on.load(std::memory_order_relaxed)) {
      submit(t);
      sync();
      return;
    }
    run_routed(t, 0, now_ns(), false);
    prof_end();
  }

  // records for up to `cap` tasks from now on; call while no task runs
  void profile_start(int64_t cap) {
    prof.assign(static_cast<size_t>(cap) * prof_stride(), 0);
    prof_cap = cap;
    prof_n.store(0);
    prof_on.store(cap > 0);
  }

  // the records so far as [n, stride] int64, then empties the profile; call while no task runs
  torch::Tensor profile_take() {
    const int64_t n = prof_n.load(std::memory_order_acquire);
    torch::Tensor out = torch::empty({n, prof_stride()}, torch::kInt64);
    std::memcpy(out.data_ptr<int64_t>(), prof.data(), sizeof(int64_t) * n * prof_stride());
    prof_n.store(0);
    return out;
  }

  static void CUDART_CB submit_cb(void* ud) {
    HostTask* h = reinterpret_cast<HostTask*>(ud);
    h->exec->use_slot(0);
    h->exec->submit(h->task);
  }
  static void CUDART_CB sync_cb(void* ud) { reinterpret_cast<CpuMoeExecutor*>(ud)->sync(); }
};

}  // namespace
}  // namespace freetoken::cpu_moe

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  namespace py = pybind11;
  using freetoken::cpu_moe::CpuMoeExecutor;
  using freetoken::cpu_moe::GpuSync;
  py::class_<CpuMoeExecutor>(m, "CpuMoeExecutor")
      .def(py::init<int, int, int, int, int, int, int, int, int, std::vector<freetoken::cpu_moe::FormatSpec>,
                    std::vector<int>, double, double, std::vector<int>, uintptr_t, uintptr_t, uintptr_t,
                    uintptr_t>(),
           py::arg("num_threads"), py::arg("num_layers"), py::arg("num_experts"),
           py::arg("top_k"), py::arg("hidden_size"), py::arg("inter_size"),
           py::arg("max_tokens"), py::arg("activation_id"),
           py::arg("apply_router_weight_on_input"), py::arg("formats"), py::arg("layer_formats"),
           py::arg("swiglu_alpha"), py::arg("swiglu_limit"),
           py::arg("core_ids"), py::arg("x_ptr"), py::arg("ids_ptr"), py::arg("w_ptr"),
           py::arg("y_ptr"))
      .def("submit_with_cuda_stream", &CpuMoeExecutor::submit_with_cuda_stream,
           py::arg("stream"), py::arg("layer_id"), py::arg("num_tokens"),
           py::call_guard<py::gil_scoped_release>())
      .def("sync_with_cuda_stream", &CpuMoeExecutor::sync_with_cuda_stream,
           py::arg("stream"), py::call_guard<py::gil_scoped_release>())
      .def("run", &CpuMoeExecutor::run, py::arg("layer_id"), py::arg("num_tokens"),
           py::call_guard<py::gil_scoped_release>())
      .def("flag_served", [](const CpuMoeExecutor& e) { return e.flag_served.load(); })
      .def("start_flag_coordinator", &CpuMoeExecutor::start_flag_coordinator, py::arg("pin_core"))
      .def("flag_signal", &CpuMoeExecutor::flag_signal, py::arg("stream"), py::arg("layer_id"),
           py::arg("num_tokens"), py::arg("slot") = 0)
      .def("flag_wait", &CpuMoeExecutor::flag_wait, py::arg("stream"), py::arg("slot") = 0)
      .def("flag_pending", &CpuMoeExecutor::flag_pending)
      .def("enable_kernel_io", &CpuMoeExecutor::enable_kernel_io)
      .def("flag_publish", &CpuMoeExecutor::flag_publish, py::arg("stream"), py::arg("x"), py::arg("ids"), py::arg("w"),
           py::arg("layer_id"), py::arg("num_tokens"), py::arg("slot") = 0)
      .def("flag_combine", &CpuMoeExecutor::flag_combine, py::arg("stream"), py::arg("out"), py::arg("num_tokens"),
           py::arg("accumulate"), py::arg("slot") = 0)
      .def_property_readonly_static("slots", [](py::object) { return CpuMoeExecutor::kSlots; })
      .def("poison_flag", &CpuMoeExecutor::poison_flag)
      .def("set_prefetch_hint", &CpuMoeExecutor::set_prefetch_hint, py::arg("ptr"), py::arg("mode"))
      .def("prefetch_now", &CpuMoeExecutor::prefetch_now, py::arg("layer_id"), py::call_guard<py::gil_scoped_release>())
      .def("prefetch_stats",
           [](const CpuMoeExecutor& e) {
             return py::make_tuple(e.pf_tasks.load(), e.pf_bytes.load(), e.pf_aborted.load());
           })
      .def_property_readonly_static("prefetch_hint_width", [](py::object) { return 2 + CpuMoeExecutor::kMaxHint; })
      .def("profile_start", &CpuMoeExecutor::profile_start, py::arg("cap"))
      .def("profile_take", &CpuMoeExecutor::profile_take)
      .def("profile_layout",
           [](const CpuMoeExecutor& e) {
             return py::make_tuple(
                 py::make_tuple("ring", "retired", "submitted", "synced", "answered", "layer", "tokens", "slot", "routes", "distinct"),
                 e.params.max_tokens * e.params.top_k,
                 py::make_tuple("wake", "pass1_end", "barrier1", "prep_end", "barrier2", "pass2_end", "pass1_busy", "pass2_busy",
                                "pass1_items", "pass2_items"),
                 e.num_threads);
           })
      .def("ce_setup", &CpuMoeExecutor::ce_setup, py::arg("rows"), py::arg("width"), py::arg("maxb"), py::arg("ready"),
           py::arg("ack"), py::arg("src"), py::arg("dst"), py::arg("feat"), py::arg("pool_of_layer"), py::arg("layer_layout"),
           py::arg("pool_base"), py::arg("device"))
      .def("ce_poll", &CpuMoeExecutor::ce_poll, py::call_guard<py::gil_scoped_release>())
      .def("ce_drain", &CpuMoeExecutor::ce_drain, py::call_guard<py::gil_scoped_release>())
      .def("ce_detach", &CpuMoeExecutor::ce_detach)
      .def("ce_stats",
           [](const CpuMoeExecutor& e) {
             return py::make_tuple(e.ce_rows_served.load(), e.ce_experts.load(), e.ce_bytes.load(), e.ce_issue_ns.load());
           })
      .def("isa_name", &CpuMoeExecutor::isa_name)
      .def("set_spin_wait",
           [](CpuMoeExecutor& e, bool v) { e.spin_wait.store(v, std::memory_order_relaxed); },
           py::arg("enabled"))
      .def("get_spin_wait", [](const CpuMoeExecutor& e) { return e.spin_wait.load(); });
  freetoken::cpu_moe::bind_locked_fetch(m);
  m.def("memops_probe", &GpuSync::probe, py::arg("stream"));
  m.def("kernel_io_probe", &freetoken::cpu_moe::KernelIo::probe, py::arg("stream"));
  m.def("host_func_capture_safe", &GpuSync::host_func_capture_safe);
  // ABI capability marker: the highest ActKind this build implements in the
  // GENERIC epilogue. CpuMoeExecutor.__init__ probes it before requesting an act
  // id the epilogue must handle -- a prebuilt .so from before ACT_SWIGLUOAI
  // accepts id 3 without error and silently computes the wrong activation
  // (act_apply falls through to gelu_tanh); the probe turns a stale extension
  // into a loud rebuild instruction instead of wrong model outputs.
  m.def("max_generic_act_id", []() { return static_cast<int>(freetoken::cpu_moe::ACT_SWIGLU_CLAMP); });
  // The ggml-cpu variant the gguf format calls: the kernel table address that freetoken.kernel.ggml_cpu.select() returns.
  m.def("set_ggml_cpu", &freetoken::cpu_moe::set_ggml_cpu, py::arg("api"));
  // the b12x activation quant of a bf16 row: (packed int8 x4, block scales)
  m.def(
      "b12x_quant_act",
      [](const torch::Tensor& x, bool simd) {
        TORCH_CHECK(x.dtype() == torch::kBFloat16 && x.dim() == 1 && x.is_contiguous() && x.size(0) % 16 == 0);
        const int K = static_cast<int>(x.size(0));
        torch::Tensor a4 = torch::empty({K / 4}, torch::kInt32), asb = torch::empty({K / 16}, torch::kFloat32);
        const auto* src = reinterpret_cast<const freetoken::cpu_moe::bf16_t*>(x.data_ptr());
#if CPU_MOE_X86
        if (simd) {
          TORCH_CHECK(__builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma"), "needs AVX2");
          freetoken::cpu_moe::quant_act_b12x_avx2(src, K, a4.data_ptr<int32_t>(), asb.data_ptr<float>());
          return py::make_tuple(a4, asb);
        }
#endif
        TORCH_CHECK(!simd, "no SIMD quant on this ISA");
        freetoken::cpu_moe::quant_act_b12x(src, K, a4.data_ptr<int32_t>(), asb.data_ptr<float>());
        return py::make_tuple(a4, asb);
      },
      py::arg("x"), py::arg("simd"));
  m.def("weight_formats", []() {
    py::list names;
    for (const auto& f : freetoken::cpu_moe::kFormats) names.append(f.first);
    return names;
  });
}
