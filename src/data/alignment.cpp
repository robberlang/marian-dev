#include "data/alignment.h"
#include "common/utils.h"

#include <algorithm>
#include <tuple>

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
    str << p->srcPos << "-" << p->tgtPos << "," << p->prob;
  }
  return str.str();
}

WordAlignment ConvertSoftAlignToHardAlign(SoftAlignment alignSoft,
                                          float threshold /*= 0.1f*/,
                                          bool useStrategy /*=true*/,
                                          bool matchLastWithLast /*= true*/) {
  WordAlignment align;
  if(alignSoft.empty()) {
    return align;
  }
  size_t endSrc = alignSoft[0].size();
  size_t endTrg = alignSoft.size();
  if(matchLastWithLast) {
    --endSrc;
    --endTrg;
  }
  // Alignments by maximum value
  if(useStrategy) {
    std::vector<std::tuple<size_t, size_t, float>> alignProbs;
    for(size_t t = 0; t < endTrg; ++t) {
      // Retrieved alignments are in reversed order
      for(size_t s = 0; s < endSrc; ++s) {
        if(alignSoft[t][s] > threshold) {
          alignProbs.emplace_back(s, t, alignSoft[t][s]);
        }
      }
    }
    std::sort(alignProbs.begin(),
              alignProbs.end(),
              [](const std::tuple<size_t, size_t, float>& a, std::tuple<size_t, size_t, float>& b) {
                return std::get<2>(a) > std::get<2>(b);
              });
    std::vector<std::pair<size_t, float>> sourceMatched(alignSoft[0].size(),
                                                        std::pair<size_t, float>((size_t)-1, 0.f));
    std::vector<std::pair<size_t, float>> targetMatched(alignSoft.size(),
                                                        std::pair<size_t, float>((size_t)-1, 0.f));
    std::vector<std::pair<size_t, size_t>> align1, align2;
    // add alignments that are the best for both source and target
    // to be follwed by adding in alignments that fit into the structure created by these
    for(const auto& a : alignProbs) {
      size_t s = std::get<0>(a);
      size_t t = std::get<1>(a);
      if(sourceMatched[s].first == (size_t)-1) {
        sourceMatched[s].first = t;
        if(targetMatched[t].first == (size_t)-1) {
          targetMatched[t].first = s;
          sourceMatched[s].second = alignSoft[t][s];
          targetMatched[t].second = alignSoft[t][s];
          align1.emplace_back(s, t);
        }
      } else if(targetMatched[t].first == (size_t)-1) {
        targetMatched[t].first = s;
      }
    }
    for(size_t s = 0; s < sourceMatched.size(); ++s) {
      if(sourceMatched[s].first != (size_t)-1 && targetMatched[sourceMatched[s].first].first != s) {
        sourceMatched[s].first = (size_t)-1;
      }
    }
    for(size_t t = 0; t < targetMatched.size(); ++t) {
      if(targetMatched[t].first != (size_t)-1 && sourceMatched[targetMatched[t].first].first != t) {
        targetMatched[t].first = (size_t)-1;
      }
    }

    if(matchLastWithLast) {
      sourceMatched[alignSoft[0].size() - 1]
          = {alignSoft.size() - 1, alignSoft[alignSoft.size() - 1][alignSoft[0].size() - 1]};
      targetMatched[alignSoft.size() - 1]
          = {alignSoft[0].size() - 1, alignSoft[alignSoft.size() - 1][alignSoft[0].size() - 1]};
      align1.emplace_back(alignSoft[0].size() - 1, alignSoft.size() - 1);
    }

    align2 = align1;
    auto srcCmp = [](const std::pair<size_t, size_t>& a, const std::pair<size_t, size_t>& b) {
      if(a.first != b.first) {
        return a.first < b.first;
      }
      return a.second < b.second;
    };
    auto trgCmp = [](const std::pair<size_t, size_t>& a, const std::pair<size_t, size_t>& b) {
      if(a.second != b.second) {
        return a.second < b.second;
      }
      return a.first < b.first;
    };
    std::sort(align1.begin(), align1.end(), srcCmp);
    std::sort(align2.begin(), align2.end(), trgCmp);
    for(;;) {
      bool insert = false;
      for(const auto& a : alignProbs) {
        size_t s = std::get<0>(a);
        size_t t = std::get<1>(a);
        // sum of probabilities for a given subword not to exceed 1
        if(!((sourceMatched[s].first == (size_t)-1
              && (targetMatched[t].second < 0.5f || alignSoft[t][s] > 0.5f))
             || (targetMatched[t].first == (size_t)-1
                 && (sourceMatched[s].second < 0.5f || alignSoft[t][s] > 0.5f)))) {
          continue;
        }
        std::pair<size_t, size_t> item(s, t);
        auto lower1 = std::lower_bound(align1.begin(), align1.end(), item, srcCmp);
        // considered alignment to be adjacent to its nearest neighbouring alignment
        std::pair<size_t, size_t> floor1{0, 0},
            ceiling1{alignSoft[0].size() - 1, alignSoft.size() - 1};
        if(lower1 != align1.begin()) {
          floor1 = *(std::prev(lower1));
        }
        if(lower1 != align1.end()) {
          ceiling1 = *lower1;
        }
        //if (!(t <= ceiling1.second && t >= floor1.second)) {
        //  continue;
        //}
        if(!(((s == ceiling1.first || s + 1 == ceiling1.first)
              || (t == ceiling1.second || t + 1 == ceiling1.second))
             || ((floor1.first == s || floor1.first + 1 == s)
                 || (floor1.second == t || floor1.second + 1 == t)))) {
          continue;
        }
        if(lower1 != align1.end() && s == ceiling1.first && t + 1 != ceiling1.second) {
          continue;
        }
        if(lower1 != align1.begin() && floor1.first == s && floor1.second + 1 != t) {
          continue;
        }
        auto lower2 = std::lower_bound(align2.begin(), align2.end(), item, trgCmp);
        std::pair<size_t, size_t> floor2{0, 0},
            ceiling2{alignSoft[0].size() - 1, alignSoft.size() - 1};
        if(lower2 != align2.begin()) {
          floor2 = *(std::prev(lower2));
        }
        if(lower2 != align2.end()) {
          ceiling2 = *lower2;
        }
        //if(!(s <= ceiling2.first && s >= floor2.first)) {
        //  continue;
        //}
        if(!((((s == ceiling2.first || s + 1 == ceiling2.first)
               || (t == ceiling2.second || t + 1 == ceiling2.second))
              || ((floor2.first == s || floor2.first + 1 == s)
                  || (floor2.second == t || floor2.second + 1 == t))))) {
          continue;
        }
        if(lower2 != align2.end() && t == ceiling2.second && s + 1 != ceiling2.first) {
          continue;
        }
        if(lower2 != align2.begin() && floor2.second == t && floor2.first + 1 != s) {
          continue;
        }
        align1.insert(lower1, item);
        align2.insert(lower2, item);
        if(sourceMatched[s].first == (size_t)-1) {
          sourceMatched[s].first = t;
        }
        if(targetMatched[t].first == (size_t)-1) {
          targetMatched[t].first = s;
        }
        sourceMatched[s].second += alignSoft[t][s];
        targetMatched[t].second += alignSoft[t][s];
        insert = true;
      }
      if(!insert) {
        break;
      }
    }
    for(const auto& a : align1) {
      align.push_back(a.first, a.second, alignSoft[a.second][a.first]);
    }
  } else {
    // All alignments by greather-than-threshold
    for(size_t t = 0; t < endTrg; ++t) {
      // Retrieved alignments are in reversed order
      for(size_t s = 0; s < endSrc; ++s) {
        if(alignSoft[t][s] > threshold) {
          align.push_back(s, t, alignSoft[t][s]);
        }
      }
    }
    if(matchLastWithLast) {
      align.push_back(alignSoft[0].size() - 1,
                      alignSoft.size() - 1,
                      alignSoft[alignSoft.size() - 1][alignSoft[0].size() - 1]);
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
