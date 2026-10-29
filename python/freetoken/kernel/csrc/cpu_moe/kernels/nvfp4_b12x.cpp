#include <cfloat>
#include <vector>

#include "../format.h"

namespace freetoken::cpu_moe {
namespace {

// flashinfer SM12x W4A16 packing, see moe_w4a16_prepare.py
constexpr int kTileN = 64;
constexpr int kTileBytes = 512;

alignas(16) const int8_t kE2M1x2B[16] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};

struct ScalePos {
  uint8_t at[kTileN];
  ScalePos() {
    const int perm4[4] = {0, 2, 1, 3};
    for (int m = 0; m < kTileN; ++m) {
      const int q = (m % 8) * 8 + m / 8;
      at[m] = static_cast<uint8_t>((q & ~3) | perm4[q & 3]);
    }
  }
};
const ScalePos kScalePos;

float fp16_bits_to_f32(uint16_t h) {
  const int exp = (h >> 10) & 0x1F, mant = h & 0x3FF;
  if (exp == 0) return std::ldexp(static_cast<float>(mant), -24);
  return std::ldexp(1.0f + mant / 1024.0f, exp - 15);
}

void tile_dot_scalar(const uint8_t* w, const uint8_t* s, int N, int K, int nt, const float* x,
                     const float* scale_lut, const float* e2m1, float* out) {
  static const int slot_h[8] = {0, 0, 1, 1, 0, 0, 1, 1};
  static const int slot_k[8] = {0, 8, 0, 8, 1, 9, 1, 9};
  float acc[kTileN] = {0};
  const size_t stride = (size_t)N * 8;
  for (int kt = 0; kt < K / 16; ++kt) {
    uint32_t tile[128];
    std::memcpy(tile, w + kt * stride + (size_t)nt * kTileBytes, sizeof(tile));
    const float* xb = x + (size_t)kt * 16;
    float part[kTileN] = {0};
    for (int p = 0; p < 128; ++p) {
      const int c = (p % 4) * 16 + (p / 4) / 4, r = 2 * ((p / 4) % 4);
      for (int sl = 0; sl < 8; ++sl)
        part[c + 8 * slot_h[sl]] += e2m1[(tile[p] >> (4 * sl)) & 0xF] * xb[r + slot_k[sl]];
    }
    const uint8_t* sc = s + (size_t)kt * N + (size_t)nt * kTileN;
    for (int m = 0; m < kTileN; ++m) acc[m] += part[m] * scale_lut[sc[kScalePos.at[m]]];
  }
  std::memcpy(out, acc, sizeof(acc));
}

}  // namespace

// int32 j of block b holds q[2j], q[2j+1], q[2j+8], q[2j+9]
void quant_act_b12x(const bf16_t* x, int K, int32_t* a4, float* asb) {
  for (int b = 0; b < K / 16; ++b) {
    float v[16], amax = 0.0f;
    for (int i = 0; i < 16; ++i) {
      v[i] = bf16_to_f32(x[(size_t)b * 16 + i]);
      amax = std::max(amax, std::fabs(v[i]));
    }
    const float s = amax > 0.0f ? amax / 127.0f : 1.0f;
    asb[b] = s;
    const float inv = 1.0f / s;
    int8_t q[16];
    for (int i = 0; i < 16; ++i)
      q[i] = static_cast<int8_t>(std::max(-127, std::min(127, (int)std::lround(v[i] * inv))));
    int8_t* out = reinterpret_cast<int8_t*>(a4 + (size_t)b * 4);
    for (int j = 0; j < 4; ++j) {
      out[4 * j + 0] = q[2 * j];
      out[4 * j + 1] = q[2 * j + 1];
      out[4 * j + 2] = q[2 * j + 8];
      out[4 * j + 3] = q[2 * j + 9];
    }
  }
}

#if CPU_MOE_X86
// bitwise quant_act_b12x: lround's ties away from zero, 0 where it is undefined
CPU_MOE_TARGET("avx2,fma")
void quant_act_b12x_avx2(const bf16_t* x, int K, int32_t* a4, float* asb) {
  const __m256 abs_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff));
  const __m256 half = _mm256_set1_ps(0.5f), one = _mm256_set1_ps(1.0f), fmax = _mm256_set1_ps(FLT_MAX);
  const __m256i lo = _mm256_set1_epi32(-127), hi = _mm256_set1_epi32(127);
  for (int b = 0; b < K / 16; ++b) {
    __m256 v[2];
    __m256 amax = _mm256_setzero_ps();
    for (int h = 0; h < 2; ++h) {
      const __m128i raw = _mm_loadu_si128(reinterpret_cast<const __m128i*>(x + (size_t)b * 16 + 8 * h));
      v[h] = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(raw), 16));
      amax = _mm256_max_ps(_mm256_and_ps(v[h], abs_mask), amax);
    }
    alignas(32) float m[8];
    _mm256_store_ps(m, amax);
    float am = 0.0f;
    for (const float e : m) am = std::max(am, e);
    const float s = am > 0.0f ? am / 127.0f : 1.0f;
    asb[b] = s;
    const __m256 inv = _mm256_set1_ps(1.0f / s);
    alignas(32) int32_t q[16];
    for (int h = 0; h < 2; ++h) {
      const __m256 y = _mm256_mul_ps(v[h], inv);
      const __m256 t = _mm256_round_ps(y, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
      const __m256 frac = _mm256_and_ps(_mm256_sub_ps(y, t), abs_mask);
      const __m256 step = _mm256_and_ps(_mm256_cmp_ps(frac, half, _CMP_GE_OQ), _mm256_or_ps(one, _mm256_andnot_ps(abs_mask, y)));
      __m256i r = _mm256_cvttps_epi32(_mm256_add_ps(t, step));
      r = _mm256_and_si256(r, _mm256_castps_si256(_mm256_cmp_ps(_mm256_and_ps(y, abs_mask), fmax, _CMP_LE_OQ)));
      _mm256_store_si256(reinterpret_cast<__m256i*>(q + 8 * h), _mm256_max_epi32(lo, _mm256_min_epi32(hi, r)));
    }
    int8_t* out = reinterpret_cast<int8_t*>(a4 + (size_t)b * 4);
    for (int j = 0; j < 4; ++j) {
      out[4 * j + 0] = static_cast<int8_t>(q[2 * j]);
      out[4 * j + 1] = static_cast<int8_t>(q[2 * j + 1]);
      out[4 * j + 2] = static_cast<int8_t>(q[2 * j + 8]);
      out[4 * j + 3] = static_cast<int8_t>(q[2 * j + 9]);
    }
  }
}
#endif

namespace {

#if CPU_MOE_X86
inline bool cpu_has_avxvnniint8() {
  const char* no = getenv("FREETOKEN_CPU_MOE_NO_VNNI_INT8");
  if (no && no[0] && no[0] != '0') return false;
  return cpu_has_avxvnni() && __builtin_cpu_supports("avxvnniint8");
}

int b12x_pf_tiles() {
  static const int v = [] {
    const char* s = getenv("FREETOKEN_CPU_MOE_B12X_PF");
    return (s && s[0]) ? atoi(s) : 2;
  }();
  return v;
}

// 0 t0, 1 t1, 2 t2, 3 nta
int b12x_pf_hint() {
  static const int v = [] {
    const char* s = getenv("FREETOKEN_CPU_MOE_B12X_PF_HINT");
    if (s == nullptr) return 0;
    if (!std::strcmp(s, "t1")) return 1;
    if (!std::strcmp(s, "t2")) return 2;
    if (!std::strcmp(s, "nta")) return 3;
    return 0;
  }();
  return v;
}

bool b12x_pf_scales() {
  static const bool v = [] {
    const char* s = getenv("FREETOKEN_CPU_MOE_B12X_PF_SCALES");
    return s && s[0] && s[0] != '0';
  }();
  return v;
}

bool b12x_pf_prologue() {
  static const bool v = [] {
    const char* s = getenv("FREETOKEN_CPU_MOE_B12X_PF_PROLOGUE");
    return s && s[0] && s[0] != '0';
  }();
  return v;
}

// k-tiles ahead for a second, t2 prefetch; 0 none
int b12x_pf_far() {
  static const int v = [] {
    const char* s = getenv("FREETOKEN_CPU_MOE_B12X_PF_FAR");
    return (s && s[0]) ? atoi(s) : 0;
  }();
  return v;
}

inline void prefetch_line(const char* p, int hint) {
  switch (hint) {
    case 1: _mm_prefetch(p, _MM_HINT_T1); break;
    case 2: _mm_prefetch(p, _MM_HINT_T2); break;
    case 3: _mm_prefetch(p, _MM_HINT_NTA); break;
    default: _mm_prefetch(p, _MM_HINT_T0);
  }
}

struct TileStream {
  const uint8_t* w;
  const uint8_t* s;
  const int32_t* a4;
  const float* asb;
};

constexpr int kMaxStreams = 8;

// each stream is one column tile; they advance k-tile by k-tile together
template <bool INT8>
CPU_MOE_TARGET("avx2,fma,f16c,avxvnni,avxvnniint8")
void tiles_dot_vnni(const TileStream* st, int n, int N, int K, float* out) {
  const __m256i gather = _mm256_setr_epi8(0, 2, 0, 2, 4, 6, 4, 6, 8, 10, 8, 10, 12, 14, 12, 14,
                                          1, 3, 1, 3, 5, 7, 5, 7, 9, 11, 9, 11, 13, 15, 13, 15);
  const __m256i lut = _mm256_broadcastsi128_si256(_mm_load_si128(reinterpret_cast<const __m128i*>(kE2M1x2B)));
  const __m256i m0f = _mm256_set1_epi8(0x0F);
  const __m128i sc_order = _mm_setr_epi8(0, 1, 4, 5, 2, 3, 6, 7, -1, -1, -1, -1, -1, -1, -1, -1);
  __m256 acc[kMaxStreams][8];
  for (int i = 0; i < n; ++i)
    for (auto& a : acc[i]) a = _mm256_setzero_ps();
  const size_t stride = (size_t)N * 8;
  const int nkt = K / 16, pf = b12x_pf_tiles(), hint = b12x_pf_hint(), far = b12x_pf_far();
  const bool pf_scales = b12x_pf_scales();
  const auto fetch = [&](int i, int kt, int h) {
    const char* t = reinterpret_cast<const char*>(st[i].w + (size_t)kt * stride);
    for (int l = 0; l < kTileBytes; l += 64) prefetch_line(t + l, h);
    if (pf_scales) prefetch_line(reinterpret_cast<const char*>(st[i].s + (size_t)kt * N), h);
  };
  if (b12x_pf_prologue())
    for (int kt = 0; kt < std::min(pf, nkt); ++kt)
      for (int i = 0; i < n; ++i) fetch(i, kt, hint);
  for (int kt = 0; kt < nkt; ++kt) {
    if (far > 0 && kt + far < nkt)
      for (int i = 0; i < n; ++i) fetch(i, kt + far, 2);
    if (pf > 0 && kt + pf < nkt)
      for (int i = 0; i < n; ++i) fetch(i, kt + pf, hint);
    for (int i = 0; i < n; ++i) {
      const uint8_t* tile = st[i].w + kt * stride;
      const int32_t* a4 = st[i].a4 + (size_t)kt * 4;
      const __m256i av[4] = {_mm256_set1_epi32(a4[0]), _mm256_set1_epi32(a4[1]), _mm256_set1_epi32(a4[2]),
                             _mm256_set1_epi32(a4[3])};
      const __m256 as = _mm256_set1_ps(st[i].asb[kt]);
      const uint8_t* srow = st[i].s + (size_t)kt * N;
      for (int tc = 0; tc < 8; ++tc) {
        const uint8_t* line = tile + tc * 64;
        __m256i di = _mm256_setzero_si256();
        for (int j = 0; j < 4; ++j) {
          const __m256i raw = _mm256_broadcastsi128_si256(_mm_loadu_si128(reinterpret_cast<const __m128i*>(line + 16 * j)));
          const __m256i g = _mm256_shuffle_epi8(raw, gather);
          const __m256i codes = _mm256_and_si256(_mm256_blend_epi16(g, _mm256_srli_epi16(g, 4), 0xAA), m0f);
          const __m256i wv = _mm256_shuffle_epi8(lut, codes);
          if constexpr (INT8) {
            di = _mm256_dpbssd_epi32(di, wv, av[j]);
          } else {
            di = _mm256_dpbusd_avx_epi32(di, _mm256_sign_epi8(wv, wv), _mm256_sign_epi8(av[j], wv));
          }
        }
        uint64_t sb;
        std::memcpy(&sb, srow + tc * 8, 8);
        const __m128i s8 = _mm_shuffle_epi8(_mm_cvtsi64_si128((long long)sb), sc_order);
        const __m256 sc = _mm256_cvtph_ps(_mm_slli_epi16(_mm_cvtepu8_epi16(s8), 7));
        acc[i][tc] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(di), _mm256_mul_ps(sc, as), acc[i][tc]);
      }
    }
  }
  for (int i = 0; i < n; ++i)
    for (int tc = 0; tc < 8; ++tc) {
      alignas(32) float v[8];
      _mm256_store_ps(v, acc[i][tc]);
      for (int l = 0; l < 8; ++l) out[i * kTileN + (l % 4) * 16 + tc + 8 * (l / 4)] = v[l];
    }
}
#endif

enum class Path { Scalar, Vnni, VnniInt8 };

class Nvfp4B12xFormat final : public MoeFormat {
 public:
  Nvfp4B12xFormat(const MoeParams& p, const FormatArgs& a) : MoeFormat(p) {
    if (p.H % kTileN != 0 || p.I % kTileN != 0)
      throw std::invalid_argument("b12x CPU MoE needs hidden and intermediate sizes that are multiples of 64");
    gate_up_ = a.bank("gate_up");
    gu_scale_ = a.bank("gate_up_scale");
    gu_alpha_ = a.bank("gate_up_alpha");
    down_ = a.bank("down");
    dn_scale_ = a.bank("down_scale");
    dn_alpha_ = a.bank("down_alpha");
#if CPU_MOE_X86
    if (cpu_has_avxvnniint8()) path_ = Path::VnniInt8;
    else if (cpu_has_avxvnni() && __builtin_cpu_supports("f16c")) path_ = Path::Vnni;
    const char* sq = getenv("FREETOKEN_CPU_MOE_SCALAR_QUANT");
    simd_quant_ = path_ != Path::Scalar && !(sq && sq[0] && sq[0] != '0');
#endif
    const size_t tokens = p.max_tokens, routes = (size_t)p.max_tokens * p.top_k;
    if (path_ == Path::Scalar) {
      xf_.resize(tokens * p.H);
      gf_.resize(routes * p.I);
    } else {
      xa_.resize(tokens * (p.H / 4));
      xs_.resize(tokens * (p.H / 16));
      ga_.resize(routes * (p.I / 4));
      gs_.resize(routes * (p.I / 16));
    }
    for (int i = 0; i < 16; ++i) e2m1_lut_[i] = kE2M1[i];
    for (int i = 0; i < 256; ++i) scale_lut_[i] = fp16_bits_to_f32(static_cast<uint16_t>(i << 7));
  }

  std::string isa_name() const override {
    switch (path_) {
      case Path::VnniInt8: return "avx2+vnni-int8(nvfp4-b12x-w4a8)";
      case Path::Vnni: return "avx2+vnni(nvfp4-b12x-w4a8)";
      default: return "scalar(nvfp4-b12x)";
    }
  }

  int row_block() const override { return kTileN; }

  int expert_ranges(int layer, int e, const void** ptrs, size_t* lens) const override {
    const size_t H = p.H, I = p.I;
    const size_t gu = 2 * I * (H / 2), gus = 2 * I * (H / 16), dn = H * (I / 2), dns = H * (I / 16);
    ptrs[0] = static_cast<const uint8_t*>(tbl_at(gate_up_, layer)) + e * gu;
    lens[0] = gu;
    ptrs[1] = static_cast<const uint8_t*>(tbl_at(gu_scale_, layer)) + e * gus;
    lens[1] = gus;
    ptrs[2] = static_cast<const uint8_t*>(tbl_at(down_, layer)) + e * dn;
    lens[2] = dn;
    ptrs[3] = static_cast<const uint8_t*>(tbl_at(dn_scale_, layer)) + e * dns;
    lens[3] = dns;
    return 4;
  }

  void prep_input(int tok, const bf16_t* x) override {
    if (path_ == Path::Scalar) {
      for (int i = 0; i < p.H; ++i) xf_[(size_t)tok * p.H + i] = bf16_to_f32(x[i]);
    } else {
      quant(x, p.H, xa_.data() + (size_t)tok * (p.H / 4), xs_.data() + (size_t)tok * (p.H / 16));
    }
  }

  int prep_input_blocks() const override { return path_ == Path::Scalar ? 0 : p.H / 16; }

  void prep_input_range(int tok, const bf16_t* x, int b0, int b1) override {
    quant(x + (size_t)b0 * 16, (b1 - b0) * 16, xa_.data() + (size_t)tok * (p.H / 4) + (size_t)b0 * 4,
          xs_.data() + (size_t)tok * (p.H / 16) + b0);
  }

  bool preps_inter() const override { return true; }

  void prep_inter(int64_t r, bf16_t* g) override {
    if (path_ == Path::Scalar) {
      for (int i = 0; i < p.I; ++i) gf_[(size_t)r * p.I + i] = bf16_to_f32(g[i]);
    } else {
      quant(g, p.I, ga_.data() + (size_t)r * (p.I / 4), gs_.data() + (size_t)r * (p.I / 16));
    }
  }

  void quant(const bf16_t* x, int K, int32_t* a4, float* asb) const {
#if CPU_MOE_X86
    if (simd_quant_) return quant_act_b12x_avx2(x, K, a4, asb);
#endif
    quant_act_b12x(x, K, a4, asb);
  }

  void gate_up(int layer, int e, int tok, int i0, int i1, const bf16_t* /*x*/, float w_in,
               bf16_t* g) const override {
    const int H = p.H, I = p.I, N = 2 * I;
    const uint8_t* w = static_cast<const uint8_t*>(tbl_at(gate_up_, layer)) + (size_t)e * N * (H / 2);
    const uint8_t* s = static_cast<const uint8_t*>(tbl_at(gu_scale_, layer)) + (size_t)e * N * (H / 16);
    const float alpha = static_cast<const float*>(tbl_at(gu_alpha_, layer))[e];
    float v[2 * kTileN];
    const Act a = act_x(tok);
    const int gate_nt = i0 / kTileN, up_nt = (I + i0) / kTileN;
    float k;
#if CPU_MOE_X86
    if (path_ != Path::Scalar) {
      const TileStream st[2] = {{w + (size_t)gate_nt * kTileBytes, s + (size_t)gate_nt * kTileN, a.a4, a.asb},
                                {w + (size_t)up_nt * kTileBytes, s + (size_t)up_nt * kTileN, a.a4, a.asb}};
      if (path_ == Path::VnniInt8) tiles_dot_vnni<true>(st, 2, N, H, v);
      else tiles_dot_vnni<false>(st, 2, N, H, v);
      k = 0x1p-127f * alpha;
    } else
#endif
    {
      tile_dot_scalar(w, s, N, H, gate_nt, a.f, scale_lut_, e2m1_lut_, v);
      tile_dot_scalar(w, s, N, H, up_nt, a.f, scale_lut_, e2m1_lut_, v + kTileN);
      k = 0x1p-126f * alpha;
    }
    gated_rows(p, i0, i1, w_in, g, [&](int row) { return (row < I ? v[row - i0] : v[kTileN + row - I - i0]) * k; });
  }

  void down(int layer, int h0, int h1, const DownRoute* routes, int n, float* out) const override {
    const int H = p.H, I = p.I, nt = h0 / kTileN;
    const uint8_t* wl = static_cast<const uint8_t*>(tbl_at(down_, layer));
    const uint8_t* sl = static_cast<const uint8_t*>(tbl_at(dn_scale_, layer));
    const float* al = static_cast<const float*>(tbl_at(dn_alpha_, layer));
    for (int j0 = 0; j0 < n; j0 += kMaxStreams) {
      const int m = std::min(kMaxStreams, n - j0);
      float v[kMaxStreams * kTileN];
      float k;
#if CPU_MOE_X86
      if (path_ != Path::Scalar) {
        TileStream st[kMaxStreams];
        for (int i = 0; i < m; ++i) {
          const DownRoute& r = routes[j0 + i];
          const Act a = act_g(r.r);
          st[i] = {wl + (size_t)r.e * H * (I / 2) + (size_t)nt * kTileBytes, sl + (size_t)r.e * H * (I / 16) + (size_t)nt * kTileN,
                   a.a4, a.asb};
        }
        if (path_ == Path::VnniInt8) tiles_dot_vnni<true>(st, m, H, I, v);
        else tiles_dot_vnni<false>(st, m, H, I, v);
        k = 0x1p-127f;
      } else
#endif
      {
        for (int i = 0; i < m; ++i) {
          const DownRoute& r = routes[j0 + i];
          tile_dot_scalar(wl + (size_t)r.e * H * (I / 2), sl + (size_t)r.e * H * (I / 16), H, I, nt, act_g(r.r).f,
                          scale_lut_, e2m1_lut_, v + i * kTileN);
        }
        k = 0x1p-126f;
      }
      for (int i = 0; i < m; ++i) {
        const float ki = k * al[routes[j0 + i].e];
        for (int h = h0; h < h1; ++h) out[(j0 + i) * kTileN + h - h0] = v[i * kTileN + h - h0] * ki;
      }
    }
  }

  bool has_tiles() const override { return path_ != Path::Scalar; }

  void gate_up_tiles(int layer, int e, int tok, int i0, int n, float w_in, bf16_t* g) const override {
#if CPU_MOE_X86
    const int H = p.H, I = p.I, N = 2 * I;
    const uint8_t* w = static_cast<const uint8_t*>(tbl_at(gate_up_, layer)) + (size_t)e * N * (H / 2);
    const uint8_t* s = static_cast<const uint8_t*>(tbl_at(gu_scale_, layer)) + (size_t)e * N * (H / 16);
    const float k = 0x1p-127f * static_cast<const float*>(tbl_at(gu_alpha_, layer))[e];
    const Act a = act_x(tok);
    TileStream st[kMaxStreams];
    for (int t = 0; t < n; ++t) {
      const int gate_nt = i0 / kTileN + t, up_nt = (I + i0) / kTileN + t;
      st[t] = {w + (size_t)gate_nt * kTileBytes, s + (size_t)gate_nt * kTileN, a.a4, a.asb};
      st[n + t] = {w + (size_t)up_nt * kTileBytes, s + (size_t)up_nt * kTileN, a.a4, a.asb};
    }
    float v[kMaxStreams * kTileN];
    if (path_ == Path::VnniInt8) tiles_dot_vnni<true>(st, 2 * n, N, H, v);
    else tiles_dot_vnni<false>(st, 2 * n, N, H, v);
    const int span = n * kTileN;
    gated_rows(p, i0, i0 + span, w_in, g, [&](int row) { return (row < I ? v[row - i0] : v[span + row - I - i0]) * k; });
#endif
  }

  void down_tiles(int layer, int h0, int n, const DownRoute& route, float* out) const override {
#if CPU_MOE_X86
    const int H = p.H, I = p.I;
    const uint8_t* w = static_cast<const uint8_t*>(tbl_at(down_, layer)) + (size_t)route.e * H * (I / 2);
    const uint8_t* s = static_cast<const uint8_t*>(tbl_at(dn_scale_, layer)) + (size_t)route.e * H * (I / 16);
    const float ki = 0x1p-127f * static_cast<const float*>(tbl_at(dn_alpha_, layer))[route.e];
    const Act a = act_g(route.r);
    TileStream st[kMaxStreams];
    for (int t = 0; t < n; ++t) {
      const int nt = h0 / kTileN + t;
      st[t] = {w + (size_t)nt * kTileBytes, s + (size_t)nt * kTileN, a.a4, a.asb};
    }
    float v[kMaxStreams * kTileN];
    if (path_ == Path::VnniInt8) tiles_dot_vnni<true>(st, n, H, I, v);
    else tiles_dot_vnni<false>(st, n, H, I, v);
    for (int i = 0; i < n * kTileN; ++i) out[i] = v[i] * ki;
#endif
  }

 private:
  struct Act {
    const float* f;
    const int32_t* a4;
    const float* asb;
  };

  Act act_x(int tok) const {
    if (path_ == Path::Scalar) return {xf_.data() + (size_t)tok * p.H, nullptr, nullptr};
    return {nullptr, xa_.data() + (size_t)tok * (p.H / 4), xs_.data() + (size_t)tok * (p.H / 16)};
  }

  Act act_g(int64_t r) const {
    if (path_ == Path::Scalar) return {gf_.data() + (size_t)r * p.I, nullptr, nullptr};
    return {nullptr, ga_.data() + (size_t)r * (p.I / 4), gs_.data() + (size_t)r * (p.I / 16)};
  }

  Path path_ = Path::Scalar;
  bool simd_quant_ = false;
  const uint64_t* gate_up_ = nullptr;
  const uint64_t* gu_scale_ = nullptr;
  const uint64_t* gu_alpha_ = nullptr;
  const uint64_t* down_ = nullptr;
  const uint64_t* dn_scale_ = nullptr;
  const uint64_t* dn_alpha_ = nullptr;
  float e2m1_lut_[16];
  float scale_lut_[256];
  AlignedVector<float> xf_, gf_;
  AlignedVector<int32_t> xa_, ga_;
  AlignedVector<float> xs_, gs_;
};

}  // namespace

std::unique_ptr<MoeFormat> make_nvfp4_b12x_format(const MoeParams& p, const FormatArgs& a) {
  return std::make_unique<Nvfp4B12xFormat>(p, a);
}

}  // namespace freetoken::cpu_moe
