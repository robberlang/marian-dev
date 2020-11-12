#include "output_printer.h"

#include <sstream>
#include <tuple>

namespace marian {

data::SoftAlignment OutputPrinter::getSoftAlignment(const Hypothesis::PtrType& hyp) {
  data::SoftAlignment align;
  auto last = hyp;
  // get soft alignments for each target word starting from the last one
  while(last->getPrevHyp().get() != nullptr) {
    align.push_back(last->getAlignment());
    last = last->getPrevHyp();
  }

  // reverse alignments
  std::reverse(align.begin(), align.end());

  return align;
}

std::string OutputPrinter::getAlignment(const data::SoftAlignment& align) {
  if(alignment_ == "soft") {
    return data::SoftAlignToString(align);
  } else if(alignment_ == "hard") {
    return data::ConvertSoftAlignToHardAlign(align, 1.f).toString();
  } else if(alignmentThreshold_ > 0.f) {
    return data::ConvertSoftAlignToHardAlign(align, alignmentThreshold_).toString();
  } else {
    ABORT("Unrecognized word alignment type");
  }
}

std::string OutputPrinter::getWordScores(const Hypothesis::PtrType& hyp) {
  std::ostringstream scores;
  scores.precision(5);
  for(const auto& score : hyp->tracebackWordScores())
    scores << " " << std::fixed << score;
  return scores.str();
}

Words OutputPrinter::reinsertTags(const Words& words,
                                  const data::SoftAlignment& align,
                                  const std::vector<std::pair<Word, size_t>>& lineTags) {
  if(lineTags.empty())
    return words;

  Words wordsWithTags;
  if(entitizeTags_) {
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
  const auto hardAlignment = data::ConvertSoftAlignToHardAlign(align, 1.f);
  // vector of tags to be reinserted in the translation: tuple consists of the iterator in the
  // source, the target position, and the opening tag target position if applicable of the beginning
  // of whatever nest there may be
  std::vector<std::tuple<std::vector<std::pair<Word, size_t>>::const_iterator, size_t, size_t>>
      translationTags;
  std::vector<std::pair<size_t, std::ptrdiff_t>> unbalancedOpenTags;
  std::vector<std::pair<size_t, size_t>> trgTagRegions;
  size_t maxOverallTgtPos = 0;
  auto curWordAlign = hardAlignment.begin();
  auto lineTag = lineTags.begin();
  for(; lineTag != lineTags.end(); ++lineTag) {
    const auto& markupTag = lineTag->first.getMarkupTag();
    if(!markupTag) // error
      continue;

    // placeholders at end, not actually tags
    if(markupTag->getType() == TagType::NONE)
      break;

    for(; curWordAlign != hardAlignment.end(); ++curWordAlign) {
      if(curWordAlign->srcPos >= lineTag->second) {
        break;
      }

      if(curWordAlign->tgtPos > maxOverallTgtPos)
        maxOverallTgtPos = curWordAlign->tgtPos;
    }

    if(markupTag->getType() != TagType::CLOSE_TAG) {
      if(markupTag->getType() == TagType::EMPTY_TAG && lineTag->second == 0) {
        translationTags.emplace_back(lineTag, 0, 0);
      } else if(curWordAlign != hardAlignment.end() /*&& curWordAlign->srcPos == lineTag->second*/) {
        translationTags.emplace_back(
            lineTag,
            curWordAlign->tgtPos,
            (unbalancedOpenTags.empty()
                 ? curWordAlign->tgtPos
                 : std::get<2>(translationTags[unbalancedOpenTags.front().first])));
      } else {
        // place at the end - but if nested, then later will put at end of the nest once the nest is
        // established
        translationTags.emplace_back(lineTag, words.size(), words.size());
      }

      if(markupTag->getType() == TagType::OPEN_TAG)
        unbalancedOpenTags.emplace_back(translationTags.size() - 1,
                                        std::distance(hardAlignment.begin(), curWordAlign));
    } else {
      if(!unbalancedOpenTags.empty()) {
        if(lineTag->second == maxSrcPos
           && std::get<0>(translationTags[unbalancedOpenTags.back().first])->second == 0) {
          // this is the case where the tag encloses the entire source
          std::get<1>(translationTags[unbalancedOpenTags.back().first]) = 0;
          for(size_t t = unbalancedOpenTags.back().first; t < translationTags.size(); ++t) {
            std::get<2>(translationTags[t]) = 0;
          }
          translationTags.emplace_back(lineTag, words.size(), 0);
        } else {
          auto wordAlign = hardAlignment.begin() + unbalancedOpenTags.back().second;
          if(wordAlign == hardAlignment.end()) {
            // this is the case where the opening tag couldn't be placed
            translationTags.emplace_back(lineTag, words.size(), words.size());
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
              if(std::get<1>(translationTags[t]) == words.size()) {
                continue;
              }
              if(std::get<1>(translationTags[t]) < minTgtPos) {
                minTgtPos = std::get<1>(translationTags[t]);
              }
              if(std::get<1>(translationTags[t]) > maxTgtPos) {
                maxTgtPos = std::get<1>(translationTags[t]);
              }
            }

            // first loop through the clear word alignments (where source position has only one
            // corresponding target position or has target positions that are contiguous) to
            // establish a base region
            std::vector<std::vector<size_t>> ambiguousTgtPoses;
            for(; wordAlign != curWordAlign;) {
              size_t srcPos = wordAlign->srcPos;
              // tgtPoses is the target positions that align with the current source position
              std::vector<size_t> tgtPoses;
              tgtPoses.push_back(wordAlign->tgtPos);
              bool contiguous = true;
              for(++wordAlign; wordAlign != curWordAlign && wordAlign->srcPos == srcPos;
                  ++wordAlign) {
                if(wordAlign->tgtPos != tgtPoses.back() + 1) {
                  contiguous = false;
                }
                tgtPoses.push_back(wordAlign->tgtPos);
              }

              if(contiguous) {
                for(auto tgtPos : tgtPoses) {
                  if(tgtPos < minTgtPos) {
                    minTgtPos = tgtPos;
                  } else if(tgtPos + 1 > maxTgtPos) {
                    maxTgtPos = tgtPos + 1;
                  }
                }
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
                size_t tgtPos = *it;
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
              if(*minDistanceTgtPosIt < minTgtPos) {
                minTgtPos = *minDistanceTgtPosIt;
              }

              // scan backward
              for(auto it = tgtPoses.rbegin() + std::distance(minDistanceTgtPosIt, tgtPoses.end());
                  it != tgtPoses.rend();
                  ++it) {
                if(*it != *(std::prev(it)) - 1) {
                  break;
                }
                if(*it < minTgtPos) {
                  minTgtPos = *it;
                }
              }

              if(*minDistanceTgtPosIt + 1 > maxTgtPos) {
                maxTgtPos = *minDistanceTgtPosIt + 1;
              }

              // scan forward
              for(auto it = std::next(minDistanceTgtPosIt); it != tgtPoses.end(); ++it) {
                if(*it != *(std::prev(it)) + 1) {
                  break;
                }
                if(*it + 1 > maxTgtPos) {
                  maxTgtPos = *it + 1;
                }
              }
            }

            // do not set maxTgtPos so that it encloses the EOS token
            if(maxTgtPos == words.size()) {
              --maxTgtPos;
            }

            // check that the chosen region doesn't overlap with other regions, causing bad syntax
            // also don't want tags nested within tags they weren't nested in on the source
            bool regionConflict = false;
            for(const auto& trgTagRegion : trgTagRegions) {
              // if the new region would cause an overlap and bad syntax then move one end of it so
              // it borders rather than overlaps
              if(minTgtPos <= trgTagRegion.first && maxTgtPos > trgTagRegion.first
                 && maxTgtPos <= trgTagRegion.second) {
                maxTgtPos = trgTagRegion.first;
              } else if(minTgtPos < trgTagRegion.second && minTgtPos >= trgTagRegion.first
                        && maxTgtPos >= trgTagRegion.second) {
                minTgtPos = trgTagRegion.second;
              } else if(minTgtPos < trgTagRegion.second
                        && maxTgtPos > trgTagRegion.first) {  // nested or is nesting
                regionConflict = true;
                break;
              }
            }

            if(!regionConflict) {
              std::get<1>(translationTags[unbalancedOpenTags.back().first]) = minTgtPos;
              for(size_t t = unbalancedOpenTags.back().first; t < translationTags.size(); ++t) {
                std::get<2>(translationTags[t]) = minTgtPos;
                // put tags that were placed at the end back to within the nest (at the end of the
                // nest)
                if(std::get<1>(translationTags[t]) > maxTgtPos) {
                  std::get<1>(translationTags[t]) = maxTgtPos;
                }
              }
              translationTags.emplace_back(lineTag, maxTgtPos, minTgtPos);
              trgTagRegions.emplace_back(minTgtPos, maxTgtPos);
            } else {
              // place these open/close tags at the end
              // put tags that were nested on the source at the end also so they remain nested
              for(size_t t = unbalancedOpenTags.back().first; t < translationTags.size(); ++t) {
                std::get<1>(translationTags[t]) = words.size();
                std::get<2>(translationTags[t]) = words.size();
              }
              translationTags.emplace_back(lineTag, words.size(), words.size());
            }
          }
        }
        unbalancedOpenTags.pop_back();
      } else {
        // received unbalanced input - closing tag has no opening tag
        translationTags.emplace_back(lineTag, maxOverallTgtPos, 0);
      }
    }
  }

  // received unbalanced input - opening tag has no closing tag - treat as if closing tags are after
  // the end of the segment
  if(!unbalancedOpenTags.empty()) {
    for(auto u = unbalancedOpenTags.rbegin(); u != unbalancedOpenTags.rend(); ++u) {
      // if there are tags nested then have them remain nested
      for(size_t t = u->first + 1; t < translationTags.size(); ++t) {
        if(std::get<1>(translationTags[t]) < std::get<1>(translationTags[u->first])) {
          std::get<1>(translationTags[u->first]) = std::get<1>(translationTags[t]);
          std::get<2>(translationTags[u->first]) = std::get<1>(translationTags[t]);
        }
      }
      for(auto a = hardAlignment.begin() + u->second; a != hardAlignment.end(); ++a) {
        if(a->tgtPos < std::get<1>(translationTags[u->first])) {
          std::get<1>(translationTags[u->first]) = a->tgtPos;
          std::get<2>(translationTags[u->first]) = a->tgtPos;
        }
      }
    }

    for(size_t t = unbalancedOpenTags.front().first + 1; t < translationTags.size(); ++t) {
      std::get<2>(translationTags[t])
          = std::get<2>(translationTags[unbalancedOpenTags.front().first]);
    }
  }

  std::stable_sort(
      translationTags.begin(),
      translationTags.end(),
      [](const std::tuple<std::vector<std::pair<Word, size_t>>::const_iterator, size_t, size_t>& a,
         const std::tuple<std::vector<std::pair<Word, size_t>>::const_iterator, size_t, size_t>& b) {
        if(std::get<1>(a) != std::get<1>(b))
          return std::get<1>(a) < std::get<1>(b);
        return std::get<2>(a) < std::get<2>(b);
      });

  auto w = words.begin();
  for(auto t = translationTags.begin(); t != translationTags.end(); ++t) {
    auto x = words.begin() + std::get<1>(*t);
    if(std::distance(w, x) > 0) {
      wordsWithTags.insert(wordsWithTags.end(), w, x);
      w = x;
    }
    wordsWithTags.push_back(std::get<0>(*t)->first);
  }

  wordsWithTags.insert(wordsWithTags.end(), w, words.end());
  for(; lineTag != lineTags.end(); ++lineTag) {
    wordsWithTags.emplace_back(lineTag->first);
  }
  return wordsWithTags;
}

}  // namespace marian
