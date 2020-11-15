#include "data/alignment.h"
#include "common/utils.h"

#include <algorithm>

namespace marian {
namespace data {

WordAlignment::WordAlignment() {}

WordAlignment::WordAlignment(const std::vector<Point>& align) : data_(align) {}

WordAlignment::WordAlignment(const std::string& line) {
  std::vector<std::string> atok = utils::splitAny(line, " -");
  for(size_t i = 0; i < atok.size(); i += 2)
    data_.emplace_back(Point{ (size_t)std::stoi(atok[i]), (size_t)std::stoi(atok[i + 1]), 1.f });
}

void WordAlignment::sort() {
  std::sort(data_.begin(), data_.end(), [](const Point& a, const Point& b) {
    return (a.srcPos == b.srcPos) ? a.tgtPos < b.tgtPos : a.srcPos < b.srcPos;
  });
}

std::string WordAlignment::toString() const {
  std::stringstream str;
  for(auto p = begin(); p != end(); ++p) {
    if(p != begin())
      str << " ";
    str << p->srcPos << "-" << p->tgtPos;
  }
  return str.str();
}

WordAlignment ConvertSoftAlignToHardAlign(SoftAlignment alignSoft,
                                          float threshold /*= 1.f*/,
                                          bool matchLastWithLast /*= false*/) {
  WordAlignment align;
  if(alignSoft.empty()) {
    return align;
  }
  // Alignments by maximum value
  if(threshold == 1.f) {
    size_t endAlignSoft = matchLastWithLast ? alignSoft.size() - 1 : alignSoft.size();
    //std::vector<bool> sourceMatched(alignSoft[0].size(), false);
    for(size_t t = 0; t < endAlignSoft; ++t) {
      // Retrieved alignments are in reversed order
      size_t maxArg = 0;
      for(size_t s = 1; s < alignSoft[0].size(); ++s) {
        if(alignSoft[t][maxArg] < alignSoft[t][s]) {
          maxArg = s;
        }
      }
      //sourceMatched[maxArg] = true;
      align.push_back(maxArg, t, 1.f);
    }
    if(matchLastWithLast) {
      //sourceMatched[alignSoft[0].size() - 1] = true;
      align.push_back(alignSoft[0].size() - 1, alignSoft.size() - 1, 1.f);
    }
    //for(size_t i = 0; i < sourceMatched.size(); ++i) {
    //  if(!sourceMatched[i]) {
    //    size_t maxArg = 0;
    //    for(size_t s = 1; s < alignSoft.size(); ++s) {
    //      if(alignSoft[maxArg][i] < alignSoft[s][i]) {
    //        maxArg = s;
    //      }
    //    }
    //    align.push_back(i, maxArg, 1.f);
    //  }
    //}
  } else {
    // Alignments by greather-than-threshold
    for(size_t t = 0; t < alignSoft.size(); ++t) {
      // Retrieved alignments are in reversed order
      for(size_t s = 0; s < alignSoft[0].size(); ++s) {
        if(alignSoft[t][s] > threshold) {
          align.push_back(s, t, alignSoft[t][s]);
        }
      }
    }
  }

  // Sort alignment pairs in ascending order
  align.sort();

  return align;
}

std::string SoftAlignToString(SoftAlignment align) {
  std::stringstream str;
  bool first = true;
  for(size_t t = 0; t < align.size(); ++t) {
    if(!first)
      str << " ";
    for(size_t s = 0; s < align[t].size(); ++s) {
      if(s != 0)
        str << ",";
      str << align[t][s];
    }
    first = false;
  }
  return str.str();
}

}  // namespace data
}  // namespace marian
