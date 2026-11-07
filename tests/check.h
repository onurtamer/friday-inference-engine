// Minimal test assertions that stay active under NDEBUG (unlike assert()).
// Each failed check is printed and counted; return CHECK_RESULT() from main().
#pragma once
#include <cmath>
#include <cstdio>

namespace friday_test {
inline int& failures() { static int n = 0; return n; }
}  // namespace friday_test

#define CHECK(cond)                                                              \
    do {                                                                         \
        if (!(cond)) {                                                           \
            std::fprintf(stderr, "CHECK failed: %s (%s:%d)\n", #cond, __FILE__,  \
                         __LINE__);                                              \
            ++friday_test::failures();                                           \
        }                                                                        \
    } while (0)

#define CHECK_NEAR(a, b, tol)                                                    \
    do {                                                                         \
        const double _a = (double)(a), _b = (double)(b), _t = (double)(tol);    \
        if (!(std::fabs(_a - _b) <= _t)) {                                       \
            std::fprintf(stderr, "CHECK_NEAR failed: %s=%g %s=%g tol=%g (%s:%d)\n", \
                         #a, _a, #b, _b, _t, __FILE__, __LINE__);                \
            ++friday_test::failures();                                           \
        }                                                                        \
    } while (0)

#define CHECK_RESULT() (friday_test::failures() == 0 ? 0 : 1)
