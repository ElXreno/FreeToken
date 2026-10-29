// Thread placement and spin helpers shared by the CPU MoE pool and the locked-layer fetch threads.
#pragma once

#include <atomic>

#include "common.h"

#if defined(__linux__)
#include <pthread.h>
#include <sched.h>
#define CPU_MOE_HAS_AFFINITY 1
#else
#define CPU_MOE_HAS_AFFINITY 0
#endif

namespace freetoken::cpu_moe {

inline void pin_this_thread(int cpu) {
#if CPU_MOE_HAS_AFFINITY
  cpu_set_t set;
  CPU_ZERO(&set);
  CPU_SET(cpu, &set);
  pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
#else
  (void)cpu;
#endif
}

inline void cpu_relax() {
#if CPU_MOE_X86
  _mm_pause();
#elif defined(__aarch64__)
  asm volatile("yield" ::: "memory");
#endif
}

inline bool umwait_enabled() {
#if CPU_MOE_X86
  static const bool on = [] {
    const char* no = getenv("FREETOKEN_CPU_MOE_NO_UMWAIT");
    if (no && no[0] && no[0] != '0') return false;
    return __builtin_cpu_supports("waitpkg") != 0;
  }();
  return on;
#else
  return false;
#endif
}

#if CPU_MOE_X86
CPU_MOE_TARGET("waitpkg") inline void umonitor_line(const void* addr) { _umonitor(const_cast<void*>(addr)); }
// C0.1 until a write to the monitored line or ~3 us of TSC
CPU_MOE_TARGET("waitpkg") inline void umwait_c01() { _umwait(1, __rdtsc() + 10000); }
#endif

// One step of waiting for `a` to stop holding `old`.
template <class T>
inline void wait_change(const std::atomic<T>& a, T old) {
#if CPU_MOE_X86
  if (umwait_enabled()) {
    umonitor_line(&a);
    if (a.load(std::memory_order_acquire) == old) umwait_c01();
    return;
  }
#endif
  (void)a;
  (void)old;
  cpu_relax();
}

}  // namespace freetoken::cpu_moe
