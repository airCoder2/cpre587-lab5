#pragma once

// Disable all timers
// #define DISABLE_TIMING

namespace ML {
namespace Config {
constexpr bool ENABLE_SIMD = false;
constexpr bool FANCY_LOGGING = true;

// Floating Point Compare Epsilon
//constexpr float EPSILON = 0.001;
constexpr float EPSILON = 0.0000000000000000000001;
} // namespace Config
} // namespace ML::Config
