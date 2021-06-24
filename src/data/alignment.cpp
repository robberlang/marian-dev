#include "data/alignment.h"
#include "common/utils.h"

#include <algorithm>
#include <tuple>
#include <cstddef>

namespace marian {
namespace data {

struct TagPosition {
  size_t id_;
  size_t pos_;
  std::ptrdiff_t span_;
  TagPosition(size_t id, size_t pos, std::ptrdiff_t span) : id_(id), pos_(pos), span_(span) {}
};

struct TagPlacement {
  std::vector<std::pair<Word, size_t>>::const_iterator lineTag_;
  TagPosition tagPosition_;
  std::vector<TagPosition> nests_;

  TagPlacement(std::vector<std::pair<Word, size_t>>::const_iterator lineTag,
               size_t id,
               size_t pos,
               std::ptrdiff_t span)
      : lineTag_(lineTag), tagPosition_(id, pos, span) {}
};

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

WordAlignment ConvertSoftAlignToHardAlign(const SoftAlignment& alignSoft,
                                          float threshold /*= 0.3f*/,
                                          bool useStrategy /*=true*/,
                                          bool matchLastWithLast /*= true*/,
                                          bool lineSpaceSymbolStart /*= false*/,
                                          bool translationSpaceSymbolStart /*= false*/) {
  WordAlignment align;
  if(alignSoft.empty()) {
    return align;
  }
  size_t startSrc = 0;
  size_t startTrg = 0;
  size_t endSrc = alignSoft[0].size();
  size_t endTrg = alignSoft.size();
  if(lineSpaceSymbolStart) {
    // don't align leading space symbol
    ++startSrc;
  }
  if(translationSpaceSymbolStart) {
    // don't align leading space symbol
    ++startTrg;
  }
  if(matchLastWithLast) {
    --endSrc;
    --endTrg;
  }
  // Alignments by maximum value
  if(useStrategy) {
    std::vector<std::tuple<size_t, size_t, float>> alignProbs;
    for(size_t t = startTrg; t < endTrg; ++t) {
      // Retrieved alignments are in reversed order
      for(size_t s = startSrc; s < endSrc; ++s) {
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
        if(!((sourceMatched[s].first == (size_t)-1
              && (targetMatched[t].second < 0.8f || alignSoft[t][s] > 0.8f))
             || (targetMatched[t].first == (size_t)-1
                 && (sourceMatched[s].second < 0.8f || alignSoft[t][s] > 0.8f)))) {
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
    for(size_t t = startTrg; t < endTrg; ++t) {
      // Retrieved alignments are in reversed order
      for(size_t s = startSrc; s < endSrc; ++s) {
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

std::string SoftAlignToString(const SoftAlignment& align) {
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

Words reinsertTags(const Words& words,
                   const SoftAlignment& align,
                   const std::vector<std::pair<Word, size_t>>& lineTags,
                   bool lineSpaceSymbolStart,
                   bool translationSpaceSymbolStart, bool entitizeTags) {
  if(lineTags.empty())
    return words;

  Words wordsWithTags;
  if(entitizeTags) {
    auto lineTag = lineTags.begin();
    for(; lineTag != lineTags.end() && lineTag->second == 0; ++lineTag) {
      wordsWithTags.emplace_back(lineTag->first);
    }

    wordsWithTags.insert(wordsWithTags.end(), words.begin(), words.end());
    for(; lineTag != lineTags.end(); ++lineTag) {
      wordsWithTags.emplace_back(lineTag->first);
    }

    return wordsWithTags;
  }


  const size_t maxSrcPos = (!align.empty() && !align[0].empty()) ? align[0].size() - 1 : 0;
  // get hard alignments, sorted by source word position, by which lineTags is also sorted
  const auto hardAlignment = ConvertSoftAlignToHardAlign(
      align, 0.3f, true, true, lineSpaceSymbolStart, translationSpaceSymbolStart);
  // vector of tags to be reinserted in the translation: tuple consists of the iterator in the
  // source, the target position, and the opening tag target position if applicable of the beginning
  // of whatever nest there may be
  std::vector<TagPlacement> translationTags;
  std::vector<std::pair<size_t, std::ptrdiff_t>> unbalancedOpenTags;
  std::vector<std::pair<size_t, size_t>> trgTagRegions;
  size_t maxOverallTgtPos = 0;
  auto curWordAlign = hardAlignment.begin();
  for(auto lineTag = lineTags.begin(); lineTag != lineTags.end(); ++lineTag) {
    const auto& markupTag = lineTag->first.getMarkupTag();
    if(!markupTag) // error
      continue;

    for(; curWordAlign != hardAlignment.end(); ++curWordAlign) {
      if(curWordAlign->srcPos >= lineTag->second) {
        break;
      }

      if(curWordAlign->tgtPos > maxOverallTgtPos)
        maxOverallTgtPos = curWordAlign->tgtPos;
    }

    if(markupTag->type() != TagType::CLOSE_TAG) {
      if(lineTag->second == 0 && markupTag->type() == TagType::EMPTY_TAG) {
        translationTags.emplace_back(lineTag, translationTags.size(), 0, 1);
      } else if(curWordAlign != hardAlignment.end() /*&& curWordAlign->srcPos == lineTag->second*/) {
        if(markupTag->type() == TagType::EMPTY_TAG
           && (markupTag->spacing() & TAGSPACING_BEFORE) == 0
           && curWordAlign != hardAlignment.begin()) {
          if(translationTags.empty() || translationTags.back().lineTag_->second != lineTag->second) {
            translationTags.emplace_back(
                lineTag, translationTags.size(), std::prev(curWordAlign)->tgtPos + 1, 1);
          } else {
            translationTags.emplace_back(lineTag,
                                         translationTags.size(),
                                         translationTags.back().tagPosition_.pos_,
                                         translationTags.back().tagPosition_.span_);
          }
        } else {
          translationTags.emplace_back(lineTag, translationTags.size(), curWordAlign->tgtPos, 1);
        }
      } else {
        // place at the end - but if nested, then later will put at end of the nest once the nest is
        // established
        translationTags.emplace_back(lineTag, translationTags.size(), words.size(), 1);
      }

      if(markupTag->type() == TagType::OPEN_TAG)
        unbalancedOpenTags.emplace_back(translationTags.size() - 1,
                                        std::distance(hardAlignment.begin(), curWordAlign));
    } else {
      if(!unbalancedOpenTags.empty()) {
        if(lineTag->second == maxSrcPos
           && translationTags[unbalancedOpenTags.back().first].lineTag_->second == 0) {
          // this is the case where the tag encloses the entire source
          translationTags[unbalancedOpenTags.back().first].tagPosition_.pos_ = 0;
          translationTags[unbalancedOpenTags.back().first].tagPosition_.span_ = words.size() + 1;
          for(size_t t = unbalancedOpenTags.back().first + 1; t < translationTags.size(); ++t) {
            translationTags[t].nests_.emplace_back(
                unbalancedOpenTags.back().first, 0, words.size() + 1);
          }
          translationTags.emplace_back(
              lineTag, unbalancedOpenTags.back().first, words.size(), -1 - words.size());
        } else if(lineTag->second == maxSrcPos
                  && translationTags[unbalancedOpenTags.back().first].lineTag_->second
                         == maxSrcPos) {
          // this is the case where the tag is at the end of the source
          for(size_t t = unbalancedOpenTags.back().first; t < translationTags.size(); ++t) {
            translationTags[t].nests_.emplace_back(
                unbalancedOpenTags.back().first, words.size() - 1, 1);
            translationTags[t].tagPosition_.pos_ = words.size() - 1;
            translationTags[t].tagPosition_.span_ = 1;
          }
          translationTags.emplace_back(
              lineTag, unbalancedOpenTags.back().first, words.size() - 1, -1);
        } else if(lineTag->second == 0) {
          // this is the case where the tag is at the beginning of the source
          for(size_t t = unbalancedOpenTags.back().first; t < translationTags.size(); ++t) {
            translationTags[t].nests_.emplace_back(unbalancedOpenTags.back().first, 0, 1);
            translationTags[t].tagPosition_.pos_ = 0;
            translationTags[t].tagPosition_.span_ = 1;
          }
          translationTags.emplace_back(
              lineTag, unbalancedOpenTags.back().first, 0, -1);
        } else {
          auto wordAlign = hardAlignment.begin() + unbalancedOpenTags.back().second;
          if(wordAlign == hardAlignment.end()) {
            // this is the case where the opening tag couldn't be placed
            translationTags.emplace_back(
                lineTag, unbalancedOpenTags.back().first, words.size(), -1);
          } else {
            // this is the normal case - opening tag appears somewhere in the middle
            // first get the boundaries of unambiguous word alignments
            // minTgtPos will be the position of the opening tag, maxTgtPos will be the position
            // of the closing tag
            size_t minTgtPos = wordAlign->tgtPos;
            size_t maxTgtPos = wordAlign->tgtPos;

            // if there are tags nested then have them remain nested
            for(size_t t = unbalancedOpenTags.back().first + 1; t < translationTags.size(); ++t) {
              // ignore tags that couldn't be placed, they will be put within the nest later
              if(translationTags[t].tagPosition_.pos_ == words.size()) {
                continue;
              }
              if(translationTags[t].tagPosition_.pos_ < minTgtPos) {
                minTgtPos = translationTags[t].tagPosition_.pos_;
              }
              if(translationTags[t].tagPosition_.pos_ > maxTgtPos) {
                maxTgtPos = translationTags[t].tagPosition_.pos_;
              }
            }

            // first loop through the clear word alignments (where source position has only one
            // corresponding target position or has target positions that are contiguous) to
            // establish a base region
            std::vector<std::pair<size_t, float>> allTgtPoses;
            std::vector<std::vector<std::pair<size_t, float>>> ambiguousTgtPoses;
            for(; wordAlign != curWordAlign;) {
              size_t srcPos = wordAlign->srcPos;
              // tgtPoses is the target positions that align with the current source position
              std::vector<std::pair<size_t, float>> tgtPoses;
              tgtPoses.emplace_back(wordAlign->tgtPos, wordAlign->prob);
              bool contiguous = true;
              for(++wordAlign; wordAlign != curWordAlign && wordAlign->srcPos == srcPos;
                  ++wordAlign) {
                if(wordAlign->tgtPos != tgtPoses.back().first + 1) {
                  contiguous = false;
                }
                tgtPoses.emplace_back(wordAlign->tgtPos, wordAlign->prob);
              }

              if(contiguous) {
                for(const auto& tgtPos : tgtPoses) {
                  if(tgtPos.first < minTgtPos) {
                    minTgtPos = tgtPos.first;
                  } else if(tgtPos.first + 1 > maxTgtPos) {
                    maxTgtPos = tgtPos.first + 1;
                  }
                }
                std::move(tgtPoses.begin(), tgtPoses.end(), std::back_inserter(allTgtPoses));
              } else {
                ambiguousTgtPoses.push_back(std::move(tgtPoses));
              }
            }

            // loop through the disjointed to-many word alignments, picking the alignment that is
            // closest to the base region
            for(const auto& tgtPoses : ambiguousTgtPoses) {
              size_t minDistance = (size_t)-1;
              auto minDistanceTgtPosIt = tgtPoses.begin();
              for(auto it = tgtPoses.begin(); it != tgtPoses.end(); ++it) {
                size_t tgtPos = it->first;
                if(tgtPos >= minTgtPos && tgtPos < maxTgtPos) {
                  minDistance = 0;
                  minDistanceTgtPosIt = it;
                  break;
                }

                size_t distance = (tgtPos < minTgtPos) ? minTgtPos - tgtPos : tgtPos - maxTgtPos + 1;
                if(distance < minDistance) {
                  minDistance = distance;
                  minDistanceTgtPosIt = it;
                }
              }

              // expand the minimum distance point to cover contiguous sequences
              if(minDistanceTgtPosIt->first < minTgtPos) {
                minTgtPos = minDistanceTgtPosIt->first;
              }

              allTgtPoses.push_back(*minDistanceTgtPosIt);

              // scan backward
              for(auto it = tgtPoses.rbegin() + std::distance(minDistanceTgtPosIt, tgtPoses.end());
                  it != tgtPoses.rend();
                  ++it) {
                if(it->first + 1 != std::prev(it)->first) {
                  break;
                }
                allTgtPoses.push_back(*it);
                if(it->first < minTgtPos) {
                  minTgtPos = it->first;
                }
              }

              if(minDistanceTgtPosIt->first + 1 > maxTgtPos) {
                maxTgtPos = minDistanceTgtPosIt->first + 1;
              }

              // scan forward
              for(auto it = std::next(minDistanceTgtPosIt); it != tgtPoses.end(); ++it) {
                if(it->first != std::prev(it)->first + 1) {
                  break;
                }
                allTgtPoses.push_back(*it);
                if(it->first + 1 > maxTgtPos) {
                  maxTgtPos = it->first + 1;
                }
              }
            }

            if(!allTgtPoses.empty()) {
              std::sort(allTgtPoses.begin(),
                        allTgtPoses.end(),
                        [](const std::pair<size_t, float>& a, const std::pair<size_t, float>& b) {
                          return a.first < b.first;
                        });
              auto bestContiguousStart = allTgtPoses.begin();
              size_t bestContiguousLength = 1;
              float bestContiguousScore = bestContiguousStart->second;
              auto curContiguousStart = bestContiguousStart;
              float curContiguousScore = bestContiguousScore;
              minTgtPos = (size_t)-1;
              maxTgtPos = (size_t)-1;
              for(auto it = std::next(allTgtPoses.begin());; ++it) {
                if(it == allTgtPoses.end() || it->first > std::prev(it)->first + 1) {
                  size_t curContiguousLength = std::distance(curContiguousStart, it);
                  if(curContiguousScore > bestContiguousScore
                     || (curContiguousScore == bestContiguousScore
                         && curContiguousLength > bestContiguousLength)) {
                    bestContiguousLength = curContiguousLength;
                    bestContiguousStart = curContiguousStart;
                    bestContiguousScore = curContiguousScore;
                  }
                  if(curContiguousScore > 0.8f || (curContiguousLength > 1
                     && curContiguousScore > 0.5f)) {
                    maxTgtPos = std::prev(it)->first + 1;
                    if(minTgtPos == (size_t)-1) {
                      minTgtPos = curContiguousStart->first;
                    }
                  }
                  if(it == allTgtPoses.end()) {
                    break;
                  }
                  curContiguousStart = it;
                  curContiguousScore = 0.f;
                }
                curContiguousScore += it->second;
              }

              size_t bestContiguousTgtPos
                  = std::next(bestContiguousStart, bestContiguousLength - 1)->first + 1;
              if(minTgtPos == (size_t)-1) {
                minTgtPos = bestContiguousStart->first;
                maxTgtPos = bestContiguousTgtPos;
              } else if(bestContiguousTgtPos > maxTgtPos) {
                maxTgtPos = bestContiguousTgtPos;
              }
            }

            // do not set maxTgtPos so that it encloses the EOS token
            if(maxTgtPos == words.size()) {
              --maxTgtPos;
            }

            // check that the chosen region doesn't overlap with other regions, causing bad syntax
            // also don't want tags nested within tags they weren't nested in on the source
            bool regionConflict = false;
            for(const auto& existingRegion : trgTagRegions) {
              if(existingRegion.first > unbalancedOpenTags.back().first) {
                // nested
                continue;
              }
              size_t existingRegionMinTgtPos
                  = translationTags[existingRegion.first].tagPosition_.pos_;
              size_t existingRegionMaxTgtPos
                  = translationTags[existingRegion.second].tagPosition_.pos_;
              // if the new region would cause an overlap and bad syntax then move one end of it so
              // it borders rather than overlaps
              if(minTgtPos <= existingRegionMinTgtPos && maxTgtPos > existingRegionMinTgtPos
                 && maxTgtPos <= existingRegionMaxTgtPos) {
                maxTgtPos = existingRegionMinTgtPos;
              } else if(minTgtPos < existingRegionMaxTgtPos && minTgtPos >= existingRegionMinTgtPos
                        && maxTgtPos >= existingRegionMaxTgtPos) {
                minTgtPos = existingRegionMaxTgtPos;
              } else if(minTgtPos < existingRegionMaxTgtPos
                        && maxTgtPos > existingRegionMinTgtPos) {  // nested or is nesting
                regionConflict = true;
                break;
              }
            }

            if(!regionConflict) {
              std::ptrdiff_t span = maxTgtPos - minTgtPos + 1;
              translationTags[unbalancedOpenTags.back().first].tagPosition_.pos_ = minTgtPos;
              translationTags[unbalancedOpenTags.back().first].tagPosition_.span_ = span;
              for(size_t t = unbalancedOpenTags.back().first + 1; t < translationTags.size(); ++t) {
                translationTags[t].nests_.emplace_back(
                    unbalancedOpenTags.back().first, minTgtPos, span);
                // put tags that were placed at the end back to within the nest (at the end of the
                // nest)
                if(translationTags[t].tagPosition_.pos_ > maxTgtPos) {
                  translationTags[t].tagPosition_.pos_ = maxTgtPos;
                }
              }
              trgTagRegions.emplace_back(unbalancedOpenTags.back().first, translationTags.size());
              translationTags.emplace_back(
                  lineTag, unbalancedOpenTags.back().first, maxTgtPos, minTgtPos - maxTgtPos - 1);
            } else {
              // place these open/close tags at the end
              // put tags that were nested on the source at the end also so they remain nested
              for(size_t t = unbalancedOpenTags.back().first; t < translationTags.size(); ++t) {
                translationTags[t].tagPosition_.pos_ = words.size();
                translationTags[t].tagPosition_.span_
                    = (translationTags[t].tagPosition_.span_ > 0) ? 1 : -1;
                if(t > unbalancedOpenTags.back().first) {
                  translationTags[t].nests_.emplace_back(
                      unbalancedOpenTags.back().first, words.size(), 1);
                }
              }
              translationTags.emplace_back(
                  lineTag, unbalancedOpenTags.back().first, words.size(), -1);
            }
          }
        }
        unbalancedOpenTags.pop_back();
      } else {
        // received unbalanced input - closing tag has no opening tag
        for(size_t t = 0; t < translationTags.size(); ++t) {
          translationTags[t].nests_.emplace_back((size_t)-1, 0, maxOverallTgtPos + 1);
        }
        translationTags.emplace_back(lineTag, (size_t)-1, maxOverallTgtPos, -1 - maxOverallTgtPos);
      }
    }
  }

  // received unbalanced input - opening tag has no closing tag - treat as if closing tags are after
  // the end of the segment
  if(!unbalancedOpenTags.empty()) {
    for(auto u = unbalancedOpenTags.rbegin(); u != unbalancedOpenTags.rend(); ++u) {
      if(translationTags[u->first].lineTag_->second == 0) {
        translationTags[u->first].tagPosition_.pos_ = 0;
      } else {
        if (translationTags[u->first].lineTag_->second == maxSrcPos) {
          translationTags[u->first].tagPosition_.pos_ = words.size();
        } else {
          for(auto a = hardAlignment.begin() + u->second; a != hardAlignment.end(); ++a) {
            if(a->tgtPos < translationTags[u->first].tagPosition_.pos_) {
              translationTags[u->first].tagPosition_.pos_ = a->tgtPos;
            }
          }
          // do not nest previous tags - put this unbalanced opening tag at the end of them if they
          // are appearing after the unbalanced tag's current assigned position
          for(size_t t = 0; t < u->first; ++t) {
            if(translationTags[t].tagPosition_.pos_ > translationTags[u->first].tagPosition_.pos_) {
              translationTags[u->first].tagPosition_.pos_ = translationTags[t].tagPosition_.pos_;
            }
          }
        }
        // if there are tags nested then have them remain nested
        for(size_t t = u->first + 1; t < translationTags.size(); ++t) {
          if(translationTags[t].tagPosition_.pos_ < translationTags[u->first].tagPosition_.pos_) {
            translationTags[u->first].tagPosition_.pos_ = translationTags[t].tagPosition_.pos_;
          }
        }
      }
      translationTags[u->first].tagPosition_.span_ = words.size() - translationTags[u->first].tagPosition_.pos_ + 1;
      for(size_t t = u->first + 1; t < translationTags.size(); ++t) {
        translationTags[t].nests_.push_back(translationTags[u->first].tagPosition_);
      }
    }
  }

  std::stable_sort(translationTags.begin(),
                   translationTags.end(),
                   [](const TagPlacement& a, const TagPlacement& b) {
                     if(a.tagPosition_.pos_ != b.tagPosition_.pos_) {
                       return a.tagPosition_.pos_ < b.tagPosition_.pos_;
                     }
                     std::ptrdiff_t i = a.nests_.size() - 1, j = b.nests_.size() - 1;
                     for(; i >= 0 && j >= 0; --i, --j) {
                       if(a.nests_[i].id_ != b.nests_[j].id_) {
                         // a and b are in different nests
                         if(a.nests_[i].pos_ != b.nests_[j].pos_) {
                           return a.nests_[i].pos_ < b.nests_[j].pos_;
                         }
                         return a.nests_[i].span_ < b.nests_[j].span_;
                       }
                     }
                     if(i >= 0) {
                       if(a.nests_[i].id_ == b.tagPosition_.id_) {
                         // b nests a
                         // if b is a closing tag then a comes first, otherwise b does
                         return (b.tagPosition_.span_ < 0);
                       }
                       if(a.nests_[i].pos_ != b.tagPosition_.pos_) {
                         return a.nests_[i].pos_ < b.tagPosition_.pos_;
                       }
                       if(b.tagPosition_.span_ > 0) {
                         // b is an opening tag
                         return a.nests_[i].span_ < b.tagPosition_.span_;
                       }
                       // b is a closing tag
                       return false;
                     }
                     if(j >= 0) {
                       if(b.nests_[j].id_ == a.tagPosition_.id_) {
                         // a nests b
                         // if a is a closing tag then b comes first, otherwise a does
                         return (a.tagPosition_.span_ > 0);
                       }
                       if(b.nests_[j].pos_ != a.tagPosition_.pos_) {
                         return a.tagPosition_.pos_ < b.nests_[j].pos_;
                       }
                       if(a.tagPosition_.span_ > 0) {
                         // a is an opening tag
                         return a.tagPosition_.span_ < b.nests_[j].span_;
                       }
                       // a is a closing tag
                       return b.nests_[j].span_ > 1 || a.tagPosition_.span_ < -1;
                     }
                     // at this point, the two tags are at the same level (i == -1 and j == -1)
                     if(a.tagPosition_.id_ != b.tagPosition_.id_) {
                       if((a.tagPosition_.span_ < 0 && b.tagPosition_.span_ < 0)
                          || (a.tagPosition_.span_ > 0 && b.tagPosition_.span_ > 0)) {
                         // both opening or both closing tags
                         return a.tagPosition_.span_ < b.tagPosition_.span_;
                       }
                       if(a.tagPosition_.span_ < 0) {
                         // a is closing, b is opening
                         return b.tagPosition_.span_ > 1 || a.tagPosition_.span_ < -1;
                       }
                       // a is opening, b is closing
                       return false;
                     }
                     // here, it is the opening and closing tag that correspond to each other
                     return a.tagPosition_.span_ > b.tagPosition_.span_;
                   });

  auto w = words.begin();
  for(auto t = translationTags.begin(); t != translationTags.end(); ++t) {
    auto x = words.begin() + t->tagPosition_.pos_;
    if(std::distance(w, x) > 0) {
      wordsWithTags.insert(wordsWithTags.end(), w, x);
      w = x;
    }
    wordsWithTags.push_back(t->lineTag_->first);
  }

  wordsWithTags.insert(wordsWithTags.end(), w, words.end());
  return wordsWithTags;
}

}  // namespace data
}  // namespace marian
