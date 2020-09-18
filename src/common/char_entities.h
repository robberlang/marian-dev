#pragma once

#include <string>

namespace marian {
namespace charentities {
struct EntityRep {
  constexpr EntityRep(char32_t val1 = 0, char32_t val2 = 0) : charVal1_(val1), charVal2_(val2) {}
  constexpr EntityRep& operator=(char32_t val) {
    charVal1_ = val;
    return *this;
  }
  char32_t charVal1_;
  char32_t charVal2_;
  std::string str();
};

EntityRep getCharRepOfEntity(const char* entityName);
}  // namespace charentities
}  // namespace marian
