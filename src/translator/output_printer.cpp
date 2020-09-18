#include "output_printer.h"

#include <sstream>

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
  const auto hardAlignment = data::ConvertSoftAlignToHardAlign(align, 1.f);
  std::vector<std::pair<std::vector<std::pair<Word, size_t>>::const_iterator, size_t>>
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
      if(curWordAlign != hardAlignment.end() && curWordAlign->srcPos == lineTag->second) {
        translationTags.emplace_back(lineTag, curWordAlign->tgtPos);
      } else {
        // place at the end
        translationTags.emplace_back(lineTag, words.size());
      }

      if(markupTag->getType() == TagType::OPEN_TAG)
        unbalancedOpenTags.emplace_back(translationTags.size() - 1,
                                        std::distance(hardAlignment.begin(), curWordAlign));
    } else {
      if(!unbalancedOpenTags.empty()) {
        if(lineTag->second == maxSrcPos && translationTags[unbalancedOpenTags.back().first].first->second == 0) {
          translationTags[unbalancedOpenTags.back().first].second = 0;
          translationTags.emplace_back(lineTag, words.size());
        } else if(translationTags[unbalancedOpenTags.back().first].second == words.size()) {
          translationTags.emplace_back(lineTag, words.size());
        } else {
          auto wordAlign = hardAlignment.begin() + unbalancedOpenTags.back().second;
          // first get the boundaries of unambiguous word alignments
          size_t minTgtPos = wordAlign->tgtPos;
          size_t maxTgtPos = wordAlign->tgtPos;

          // if there are tags nested then have them remain nested
          size_t nextTagIndex = unbalancedOpenTags.back().first + 1;
          if(nextTagIndex < translationTags.size()
             && translationTags[nextTagIndex].second < minTgtPos) {
            minTgtPos = translationTags[nextTagIndex].second;
          }

          if(translationTags.back().second > maxTgtPos + 1) {
            maxTgtPos = translationTags.back().second - 1;
          }

          // first loop through the clear word alignments (where source position has only one
          // corresponding target position or has target positions that are contiguous) to establish
          // a base region
          std::vector<std::vector<size_t>> ambiguousTgtPoses;
          for(; wordAlign != curWordAlign;) {
            size_t srcPos = wordAlign->srcPos;
            std::vector<size_t> tgtPoses;
            tgtPoses.push_back(wordAlign->tgtPos);
            bool contiguous = true;
            for(++wordAlign; wordAlign != curWordAlign && wordAlign->srcPos == srcPos;
                ++wordAlign) {
              if(wordAlign->tgtPos != tgtPoses.back() + 1)
                contiguous = false;
              tgtPoses.push_back(wordAlign->tgtPos);
            }

            if(contiguous) {
              for(auto tgtPos : tgtPoses) {
                if(tgtPos < minTgtPos)
                  minTgtPos = tgtPos;
                else if(tgtPos > maxTgtPos)
                  maxTgtPos = tgtPos;
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
              if(tgtPos >= minTgtPos && tgtPos <= maxTgtPos) {
                minDistance = 0;
                minDistanceTgtPosIt = it;
                break;
              }
              size_t distance = (tgtPos < minTgtPos) ? minTgtPos - tgtPos : tgtPos - maxTgtPos;
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
            for(auto it
                = tgtPoses.rbegin() + std::distance(minDistanceTgtPosIt, tgtPoses.end());
                it != tgtPoses.rend();
                ++it) {
              if(*it != *(std::prev(it)) - 1) {
                break;
              }
              if(*it < minTgtPos) {
                minTgtPos = *it;
              }
            }

            if(*minDistanceTgtPosIt > maxTgtPos) {
              maxTgtPos = *minDistanceTgtPosIt;
            }

            // scan forward
            for(auto it = std::next(minDistanceTgtPosIt); it != tgtPoses.end(); ++it) {
              if(*it != *(std::prev(it)) + 1) {
                break;
              }
              if(*it > maxTgtPos) {
                maxTgtPos = *it;
              }
            }
          }

          // do not set maxTgtPos so that it encloses the EOS token
          if(maxTgtPos == words.size() - 1) {
            --maxTgtPos;
          }

          // check that the chosen region doesn't overlap with other regions, causing bad syntax
          // also don't want tags nested within tags they weren't nested in on the source
          bool regionConflict = false;
          for(const auto& trgTagRegion : trgTagRegions) {
            // if the new region would cause an overlap and bad syntax then move one end of it so it
            // borders rather than overlaps
            if(minTgtPos < trgTagRegion.first && maxTgtPos >= trgTagRegion.first
               && maxTgtPos <= trgTagRegion.second) {
              maxTgtPos = trgTagRegion.first - 1;
            } else if(minTgtPos <= trgTagRegion.second && minTgtPos >= trgTagRegion.first
                      && maxTgtPos > trgTagRegion.second) {
              minTgtPos = trgTagRegion.second + 1;
            } else if(minTgtPos <= trgTagRegion.second
                      && maxTgtPos >= trgTagRegion.first) {  // nested or is nesting
              regionConflict = true;
              break;
            }
          }

          if(!regionConflict) {
            translationTags[unbalancedOpenTags.back().first].second = minTgtPos;
            translationTags.emplace_back(lineTag, maxTgtPos + 1);
            trgTagRegions.emplace_back(minTgtPos, maxTgtPos);
          } else {
            // place these open/close tags at the end
            // put tags that were nested on the source at the end also so they remain nested
            for(size_t i = unbalancedOpenTags.back().first; i < translationTags.size(); ++i) {
              translationTags[i].second = words.size();
            }
            translationTags.emplace_back(lineTag, words.size());
          }
        }
        unbalancedOpenTags.pop_back();
      } else {
        translationTags.emplace_back(lineTag, maxOverallTgtPos + 1);
      }
    }
  }

  // received unbalanced input - treat as if closing tags are after the end of the segment
  if(!unbalancedOpenTags.empty()) {
    for(auto& u : unbalancedOpenTags) {
      // if there are tags nested then have them remain nested
      size_t nextTagIndex = u.first + 1;
      if(nextTagIndex < translationTags.size()
         && translationTags[nextTagIndex].second < translationTags[u.first].second) {
        translationTags[u.first].second = translationTags[nextTagIndex].second;
      }
      for(auto a = hardAlignment.begin() + u.second; a != hardAlignment.end(); ++a) {
        if(a->tgtPos < translationTags[u.first].second)
          translationTags[u.first].second = a->tgtPos;
      }
    }
  }

  std::stable_sort(
      translationTags.begin(),
      translationTags.end(),
      [](const std::pair<std::vector<std::pair<Word, size_t>>::const_iterator, size_t>& a,
         const std::pair<std::vector<std::pair<Word, size_t>>::const_iterator, size_t>& b) {
        return a.second < b.second;
      });

  auto w = words.begin();
  for(auto t = translationTags.begin(); t != translationTags.end(); ++t) {
    auto x = words.begin() + t->second;
    if(std::distance(w, x) > 0) {
      wordsWithTags.insert(wordsWithTags.end(), w, x);
      w = x;
    }
    wordsWithTags.push_back(t->first->first);
  }

  wordsWithTags.insert(wordsWithTags.end(), w, words.end());
  for(; lineTag != lineTags.end(); ++lineTag) {
    wordsWithTags.emplace_back(lineTag->first);
  }
  return wordsWithTags;
}

}  // namespace marian
