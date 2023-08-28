#include "data/alignment.h"
#include "common/utils.h"

#include <algorithm>
#include <tuple>
#include <cstddef>
#include <cmath>
#include <set>

namespace marian {
namespace data {

struct TagNest {
  size_t id_;
  Ptr<size_t> openTagPosition_;
  Ptr<size_t> closeTagPosition_;
  TagNest(size_t id,
          Ptr<size_t> openTagPosition,
          Ptr<size_t> closeTagPosition)
      : id_(id),
        openTagPosition_(openTagPosition),
        closeTagPosition_(closeTagPosition) {}
};

struct TagPlacement {
  std::pair<Word, size_t> lineTag_;
  size_t id_;
  Ptr<size_t> tagPosition_;
  Ptr<size_t> matchingTagPosition_;
  std::vector<TagNest> nests_;
  TagPlacement(std::pair<Word, size_t> lineTag,
               size_t id,
               size_t pos)
      : lineTag_(std::move(lineTag)),
        id_(id),
        tagPosition_(New<size_t>(pos)),
        matchingTagPosition_(tagPosition_) {}
  TagPlacement(std::pair<Word, size_t> lineTag,
               size_t id,
               Ptr<size_t> tagPosition,
               Ptr<size_t> matchingTagPosition)
      : lineTag_(std::move(lineTag)),
        id_(id), tagPosition_(tagPosition),
        matchingTagPosition_(matchingTagPosition) {}
  bool operator<(const TagPlacement& rhs) const {
#if 0
    bool ret = compare(rhs);
    if(*tagPosition_ == *rhs.tagPosition_) {
      LOG(info,
          "Compare lhs id {}, pos {}, open {}, rhs id {}, pos {}, open {}, lhs < rhs: {}",
          id_,
          *tagPosition_,
          lineTag_.first.getMarkupTag()->type() != TagType::CLOSE_TAG,
          rhs.id_,
          *rhs.tagPosition_,
          rhs.lineTag_.first.getMarkupTag()->type() != TagType::CLOSE_TAG,
          ret);
    }
    return ret;
  }
  bool compare(const TagPlacement& rhs) const {
#endif
    if(*tagPosition_ != *rhs.tagPosition_) {
      return *tagPosition_ < *rhs.tagPosition_;
    }
    std::ptrdiff_t i = nests_.size() - 1, j = rhs.nests_.size() - 1;
    for(; i >= 0 && j >= 0; --i, --j) {
      if(nests_[i].id_ != rhs.nests_[j].id_) {
        // a and b are in different nests
        if(*nests_[i].openTagPosition_ != *rhs.nests_[j].openTagPosition_) {
          return *nests_[i].openTagPosition_ < *rhs.nests_[j].openTagPosition_;
        }
        return *nests_[i].closeTagPosition_ < *rhs.nests_[j].closeTagPosition_;
      }
    }
    if(i >= 0) {
      if(nests_[i].id_ == rhs.id_) {
        // b nests a
        // if b is a closing tag then a comes first, otherwise b does
        return (rhs.lineTag_.first.getMarkupTag()->type() == TagType::CLOSE_TAG);
      }
      if(*nests_[i].openTagPosition_ != *rhs.tagPosition_) {
        return *nests_[i].openTagPosition_ < *rhs.tagPosition_;
      }
      if(rhs.lineTag_.first.getMarkupTag()->type() != TagType::CLOSE_TAG) {
        // b is an opening tag
        return *nests_[i].closeTagPosition_ < *rhs.matchingTagPosition_;
      }
      // b is a closing tag
      // b does not nest a and a nest start occurs at a/b position (above
      // conditions)
      // not enough info to determine order, rely on b < a and stable sort
      return false;
    }
    if(j >= 0) {
      if(rhs.nests_[j].id_ == id_) {
        // a nests b
        // if a is a not a closing tag then a comes first, otherwise b does
        return (lineTag_.first.getMarkupTag()->type() != TagType::CLOSE_TAG);
      }
      if(*rhs.nests_[j].openTagPosition_ != *tagPosition_) {
        return *tagPosition_ < *rhs.nests_[j].openTagPosition_;
      }
      if(lineTag_.first.getMarkupTag()->type() != TagType::CLOSE_TAG) {
        // a is an opening tag
        return *matchingTagPosition_ < *rhs.nests_[j].closeTagPosition_;
      }
      // a is a closing tag
      // if a does not nest b and b nest start occurs at a/b position (above
      // conditions) and b nest spans multiple positions or a spans multiple
      // positions (below condition) then a comes before b
      return *rhs.nests_[j].closeTagPosition_ > *rhs.nests_[j].openTagPosition_
             || *tagPosition_ > *matchingTagPosition_;
    }
    // at this point, the two tags are at the same level (i == -1 and j == -1)
    if(id_ != rhs.id_) {
      if((lineTag_.first.getMarkupTag()->type() == TagType::CLOSE_TAG
          && rhs.lineTag_.first.getMarkupTag()->type() == TagType::CLOSE_TAG)
         || (lineTag_.first.getMarkupTag()->type() != TagType::CLOSE_TAG
             && rhs.lineTag_.first.getMarkupTag()->type() != TagType::CLOSE_TAG)) {
        // both opening or both closing tags
        return *matchingTagPosition_ < *rhs.matchingTagPosition_;
      }
      if(lineTag_.first.getMarkupTag()->type() == TagType::CLOSE_TAG) {
        // a is closing, b is opening, if a or b spans multiple positions then a comes before b
        return *rhs.matchingTagPosition_ > *rhs.tagPosition_
               || *tagPosition_ > *matchingTagPosition_;
      }
      // b is closing, a is opening - not enough info to determine order, rely on b < a and stable
      // sort
      return false;
    }
    // here, it is the opening and closing tag that correspond to each other
    return lineTag_.first.getMarkupTag()->type() != TagType::CLOSE_TAG;
  }
};

struct TagBalancingInfo {
  size_t tagIndex_;
  size_t realTagIndex_;
  size_t alignmentIndex_;
  std::string tagIdentifier_;
  bool fake_{false};
  TagBalancingInfo(size_t tagIndex,
                   size_t alignmentIndex,
                   std::string tagIdentifier)
      : tagIndex_(tagIndex),
        realTagIndex_(tagIndex),
        alignmentIndex_(alignmentIndex),
        tagIdentifier_(std::move(tagIdentifier)) {}
  bool artificial() const { return fake_ || tagIndex_ != realTagIndex_; }
};

bool regionConflicts(const std::vector<std::pair<size_t, size_t>>& trgTagRegions,
                     const TagBalancingInfo& unbalancedOpenTag,
                     const std::vector<TagPlacement>& translationTags,
                     const size_t sentenceLength,
                     const size_t leftTgtBoundary,
                     const bool minIsFixed,
                     const bool maxIsFixed,
                     size_t& minTgtPos,
                     size_t& maxTgtPos) {
  if(minTgtPos < leftTgtBoundary) {
    return true;
  }

  for(const auto& existingRegion : trgTagRegions) {
    if(existingRegion.first >= unbalancedOpenTag.tagIndex_) {
      // nested
      continue;
    }
    size_t existingRegionMinTgtPos = (existingRegion.first < translationTags.size())
                                         ? *translationTags[existingRegion.first].tagPosition_
                                         : 0;
    size_t existingRegionMaxTgtPos = (existingRegion.second < translationTags.size())
                                         ? *translationTags[existingRegion.second].tagPosition_
                                         : sentenceLength;
    // if the new region would cause an overlap and bad syntax then move one end of it so
    // it borders rather than overlaps
    if(minTgtPos <= existingRegionMinTgtPos && maxTgtPos > existingRegionMinTgtPos
       && maxTgtPos <= existingRegionMaxTgtPos) {
      if(!maxIsFixed) {
        maxTgtPos = existingRegionMinTgtPos;
      } else if(maxTgtPos == existingRegionMaxTgtPos) {
        if(!minIsFixed) {
          minTgtPos = existingRegionMaxTgtPos;
        } else if(minTgtPos != existingRegionMaxTgtPos) {
          return true;
        }
      } else {
        return true;
      }
    } else if(minTgtPos < existingRegionMaxTgtPos && minTgtPos >= existingRegionMinTgtPos
              && maxTgtPos >= existingRegionMaxTgtPos) {
      if(!minIsFixed) {
        minTgtPos = existingRegionMaxTgtPos;
      } else if(minTgtPos == existingRegionMinTgtPos) {
        if(!maxIsFixed) {
          maxTgtPos = existingRegionMinTgtPos;
        } else if(maxTgtPos != existingRegionMinTgtPos) {
          return true;
        }
      } else {
        return true;
      }
    } else if(minTgtPos < existingRegionMaxTgtPos
              && maxTgtPos > existingRegionMinTgtPos) {  // nested or is nesting
      return true;
    }
  }
  return false;
}

void positionBalancedTags(const TagBalancingInfo& unbalancedOpenTag,
                          const size_t leftTgtBoundary,
                          const std::pair<Word, size_t>& lineTag,
                          const WordAlignment& hardAlignment,
                          const size_t curWordAlignIdx,
                          const size_t sentenceLength,
                          const size_t maxSrcPos,
                          const size_t thisElementId,
                          const bool forceEndCloseTagToEnd,
                          std::vector<TagPlacement>& translationTags,
                          Ptr<size_t> closeTagPosition,
                          std::vector<std::pair<size_t, size_t>>& trgTagRegions) {
  const auto& markupTag = lineTag.first.getMarkupTag();
  const auto curWordAlign = hardAlignment.begin() + curWordAlignIdx;
  bool checkForRegionConflict = false;
  // minTgtPos will be the position of the opening tag, maxTgtPos will be the position
  // of the closing tag
  size_t minTgtPos = *translationTags[unbalancedOpenTag.tagIndex_].tagPosition_;
  size_t maxTgtPos = minTgtPos;
  if(markupTag->type() == TagType::CLOSE_TAG) {
    if(lineTag.second == maxSrcPos
       && translationTags[unbalancedOpenTag.tagIndex_].lineTag_.second == 0
       && leftTgtBoundary == 0) {
      // this is the case where the tag encloses the entire source
      *closeTagPosition = sentenceLength - 1;
      if(!unbalancedOpenTag.artificial()) {
        *translationTags[unbalancedOpenTag.tagIndex_].tagPosition_ = 0;
      }
      for(size_t t = unbalancedOpenTag.tagIndex_ + 1; t < translationTags.size(); ++t) {
        translationTags[t].nests_.emplace_back(
            thisElementId,
            translationTags[unbalancedOpenTag.tagIndex_].tagPosition_,
            closeTagPosition);
        // put tags that were placed at the end back to within the nest (at the end of the
        // nest)
        if(*translationTags[t].tagPosition_ > sentenceLength - 1) {
          *translationTags[t].tagPosition_ = sentenceLength - 1;
        }
      }
      translationTags.emplace_back(lineTag,
                                   thisElementId,
                                   closeTagPosition,
                                   translationTags[unbalancedOpenTag.realTagIndex_].tagPosition_);
      translationTags[unbalancedOpenTag.realTagIndex_].matchingTagPosition_ = closeTagPosition;
    } else if(lineTag.second == maxSrcPos
              && translationTags[unbalancedOpenTag.tagIndex_].lineTag_.second == maxSrcPos
              && (!unbalancedOpenTag.artificial()
                  || *translationTags[unbalancedOpenTag.tagIndex_].tagPosition_
                         == sentenceLength - 1)) {
      // this is the case where the tag is at the end of the source
      *closeTagPosition = sentenceLength - 1;
      if(!unbalancedOpenTag.artificial()) {
        *translationTags[unbalancedOpenTag.tagIndex_].tagPosition_ = sentenceLength - 1;
      }
      for(size_t t = unbalancedOpenTag.tagIndex_ + 1; t < translationTags.size(); ++t) {
        translationTags[t].nests_.emplace_back(
            thisElementId,
            translationTags[unbalancedOpenTag.tagIndex_].tagPosition_,
            closeTagPosition);
        *translationTags[t].tagPosition_ = sentenceLength - 1;
      }
      translationTags.emplace_back(lineTag,
                                   thisElementId,
                                   closeTagPosition,
                                   translationTags[unbalancedOpenTag.realTagIndex_].tagPosition_);
      translationTags[unbalancedOpenTag.realTagIndex_].matchingTagPosition_ = closeTagPosition;
    } else if(lineTag.second == 0 && leftTgtBoundary == 0) {
      // this is the case where the tag is at the beginning of the source
      if(!unbalancedOpenTag.artificial()) {
        *translationTags[unbalancedOpenTag.tagIndex_].tagPosition_ = 0;
      }
      *closeTagPosition = *translationTags[unbalancedOpenTag.tagIndex_].tagPosition_;
      for(size_t t = unbalancedOpenTag.tagIndex_ + 1; t < translationTags.size(); ++t) {
        translationTags[t].nests_.emplace_back(
            thisElementId,
            translationTags[unbalancedOpenTag.tagIndex_].tagPosition_,
            closeTagPosition);
        *translationTags[t].tagPosition_
            = *translationTags[unbalancedOpenTag.tagIndex_].tagPosition_;
      }
      translationTags.emplace_back(lineTag,
                                   thisElementId, closeTagPosition,
                                   translationTags[unbalancedOpenTag.realTagIndex_].tagPosition_);
      translationTags[unbalancedOpenTag.realTagIndex_].matchingTagPosition_ = closeTagPosition;
    } else {
      checkForRegionConflict = true;
      auto wordAlign = hardAlignment.begin() + unbalancedOpenTag.alignmentIndex_;
      if(wordAlign == hardAlignment.end() && !unbalancedOpenTag.artificial()) {
        // this is the case where the opening tag couldn't be placed
        // place at the end - but if nested, then later will put at end of the nest once the
        // nest is established
        *closeTagPosition = sentenceLength;
        translationTags.emplace_back(lineTag,
                                     thisElementId,
                                     closeTagPosition,
                                     translationTags[unbalancedOpenTag.realTagIndex_].tagPosition_);
        translationTags[unbalancedOpenTag.realTagIndex_].matchingTagPosition_ = closeTagPosition;
        checkForRegionConflict = false;
      } else if(lineTag.second != translationTags[unbalancedOpenTag.tagIndex_].lineTag_.second) {
        // this is the normal case - non-empty element opening tag appears somewhere in the
        // middle; empty elements (those for which lineTag.second ==
        // translationTags[unbalancedOpenTag.tagIndex_].lineTag_.second) are treated as
        // self-closing

        // first get the boundaries of unambiguous word alignments
        if(wordAlign != hardAlignment.end()) {
          minTgtPos = wordAlign->tgtPos;
          maxTgtPos = wordAlign->tgtPos;
        }

        if(forceEndCloseTagToEnd) {
          maxTgtPos = sentenceLength - 1;
        }

        const auto openTagWordAlign = wordAlign;
        size_t minTgtPosNest = -1;
        size_t maxTgtPosNest = 0;
        // if there are tags nested then have them remain nested
        for(size_t t = unbalancedOpenTag.tagIndex_ + 1; t < translationTags.size(); ++t) {
          // ignore tags that couldn't be placed, they will be put within the nest later
          if(*translationTags[t].tagPosition_ == sentenceLength) {
            continue;
          }
          if(*translationTags[t].tagPosition_ < minTgtPosNest) {
            minTgtPosNest = *translationTags[t].tagPosition_;
          }
          if(*translationTags[t].tagPosition_ > maxTgtPosNest) {
            maxTgtPosNest = *translationTags[t].tagPosition_;
          }
        }

        if(minTgtPosNest != (size_t)-1) {
          if(minTgtPosNest < minTgtPos) {
            minTgtPos = minTgtPosNest;
          }
          if(maxTgtPosNest > maxTgtPos) {
            maxTgtPos = maxTgtPosNest;
          }
        }

        size_t minTgtPosOfAlignments = minTgtPos;
        size_t maxTgtPosOfAlignments = maxTgtPos;
        if(wordAlign != hardAlignment.end()) {
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
            if(wordAlign->tgtPos < minTgtPosOfAlignments) {
              minTgtPosOfAlignments = wordAlign->tgtPos;
            } else if(wordAlign->tgtPos + 1 > maxTgtPosOfAlignments) {
              maxTgtPosOfAlignments = wordAlign->tgtPos + 1;
            }
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

          // loop through the disjointed to-many word alignments, picking the alignment that
          // is closest to the base region
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
            size_t bestContiguousCoverage = 1;
            float bestContiguousScore = bestContiguousStart->second;
            auto curContiguousStart = bestContiguousStart;
            float curContiguousScore = bestContiguousScore;
            size_t curContiguousCoverage = 1;
            minTgtPos = (size_t)-1;
            maxTgtPos = (size_t)-1;
            for(auto it = std::next(allTgtPoses.begin());; ++it) {
              if(it == allTgtPoses.end() || it->first > std::prev(it)->first + 2) {
                if(curContiguousScore > bestContiguousScore
                   || (curContiguousScore == bestContiguousScore
                       && curContiguousCoverage > bestContiguousCoverage)) {
                  bestContiguousCoverage = curContiguousCoverage;
                  bestContiguousStart = curContiguousStart;
                  bestContiguousScore = curContiguousScore;
                  bestContiguousLength = std::distance(curContiguousStart, it);
                }
                if(curContiguousScore > 0.8f
                   || (curContiguousCoverage > 1 && curContiguousScore > 0.5f)) {
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
              curContiguousCoverage = it->first - curContiguousStart->first + 1;
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

            if(minTgtPosNest != (size_t)-1) {
              if(minTgtPosNest < minTgtPos) {
                minTgtPos = minTgtPosNest;
              }
              if(maxTgtPosNest > maxTgtPos) {
                maxTgtPos = maxTgtPosNest;
              }
            }
          }
        }

        // look at the words before and after the source region and see if they encompass the
        // determined target region - if the indication is strong, extend the target region
        // accordingly
        if(openTagWordAlign != hardAlignment.begin()) {
          auto prevWordAlign = std::prev(openTagWordAlign);
          if(prevWordAlign->srcPos + 1
                 == translationTags[unbalancedOpenTag.tagIndex_].lineTag_.second
             && prevWordAlign->tgtPos + 1 < minTgtPos
             && prevWordAlign->tgtPos + 1 >= minTgtPosOfAlignments) {
            minTgtPos = prevWordAlign->tgtPos + 1;
          }
        }

        if(curWordAlign != hardAlignment.end() && curWordAlign->srcPos == lineTag.second
           && curWordAlign->tgtPos > maxTgtPos && curWordAlign->tgtPos <= maxTgtPosOfAlignments) {
          maxTgtPos = curWordAlign->tgtPos;
        }
        if(maxTgtPos == sentenceLength || forceEndCloseTagToEnd) {
          // in the case of the former condition, ensuring that maxTgtPos encloses the EOS token
          maxTgtPos = sentenceLength - 1;
        }
      }
    }
  } else {
    checkForRegionConflict = true;
  }
  if(checkForRegionConflict) {
    // check that the chosen region doesn't overlap with other regions, causing bad syntax
    // also don't want tags nested within tags they weren't nested in on the source
    if(unbalancedOpenTag.artificial()) {
      minTgtPos = *translationTags[unbalancedOpenTag.tagIndex_].tagPosition_;
      if(maxTgtPos < minTgtPos) {
        maxTgtPos = minTgtPos;
      }
    }
    if(!regionConflicts(trgTagRegions,
                        unbalancedOpenTag,
                        translationTags,
                        sentenceLength,
                        leftTgtBoundary,
                        unbalancedOpenTag.artificial(),
                        forceEndCloseTagToEnd,
                        minTgtPos,
                        maxTgtPos)) {
      // no conflict or conflict resolved
      *translationTags[unbalancedOpenTag.tagIndex_].tagPosition_ = minTgtPos;
      *closeTagPosition = maxTgtPos;
      for(size_t t = unbalancedOpenTag.tagIndex_ + 1; t < translationTags.size(); ++t) {
        translationTags[t].nests_.emplace_back(
            thisElementId,
            translationTags[unbalancedOpenTag.tagIndex_].tagPosition_,
            closeTagPosition);
        // put tags that were placed outside the new range back to within the nest (at the end of
        // the nest)
        if(*translationTags[t].tagPosition_ > maxTgtPos
           || *translationTags[t].tagPosition_ < minTgtPos) {
          *translationTags[t].tagPosition_ = maxTgtPos;
        }
      }
      if(markupTag->type() == TagType::CLOSE_TAG) {
        trgTagRegions.emplace_back(unbalancedOpenTag.tagIndex_, translationTags.size());
        translationTags.emplace_back(lineTag,
                                     thisElementId,
                                     closeTagPosition,
                                     translationTags[unbalancedOpenTag.realTagIndex_].tagPosition_);
        translationTags[unbalancedOpenTag.realTagIndex_].matchingTagPosition_ = closeTagPosition;
      } else {
        trgTagRegions.emplace_back(unbalancedOpenTag.tagIndex_, unbalancedOpenTag.tagIndex_);
      }
    } else {
      // place these open/close tags at the end or in the case of close/close tag (misnested)
      // place as empty element at the position of the first close tag
      // also make tags that were nested on the source remain nested
      if(!unbalancedOpenTag.artificial() || forceEndCloseTagToEnd) {
        *translationTags[unbalancedOpenTag.tagIndex_].tagPosition_ = sentenceLength;
        if(unbalancedOpenTag.realTagIndex_ != unbalancedOpenTag.tagIndex_) {
          // artificial needs to be forced to end at the real start
          size_t outermostNestId = translationTags[unbalancedOpenTag.realTagIndex_].id_;
          for(const auto& n : translationTags[unbalancedOpenTag.realTagIndex_].nests_) {
            if(n.id_ < outermostNestId) {
              outermostNestId = n.id_;
            }
          }
          for(size_t t = unbalancedOpenTag.tagIndex_ - 1;
              t > 0 && translationTags[t].id_ >= outermostNestId;
              --t) {
            *translationTags[t].tagPosition_
                = *translationTags[unbalancedOpenTag.tagIndex_].tagPosition_;
          }
        }
      }
      *closeTagPosition = *translationTags[unbalancedOpenTag.tagIndex_].tagPosition_;
      for(size_t t = unbalancedOpenTag.tagIndex_ + 1; t < translationTags.size(); ++t) {
        translationTags[t].nests_.emplace_back(
            thisElementId,
            translationTags[unbalancedOpenTag.tagIndex_].tagPosition_,
            closeTagPosition);
        *translationTags[t].tagPosition_
            = *translationTags[unbalancedOpenTag.tagIndex_].tagPosition_;
      }
      if(markupTag->type() == TagType::CLOSE_TAG) {
        translationTags.emplace_back(lineTag,
                                     thisElementId,
                                     closeTagPosition,
                                     translationTags[unbalancedOpenTag.realTagIndex_].tagPosition_);
        translationTags[unbalancedOpenTag.realTagIndex_].matchingTagPosition_ = closeTagPosition;
      }
    }
  }
}

size_t getLeftTgtBoundary(std::vector<TagBalancingInfo>::const_reverse_iterator rit,
                          std::vector<TagBalancingInfo>::const_reverse_iterator rend,
                          const std::vector<TagPlacement>& translationTags) {
  size_t leftTgtBoundary = 0;
  for(; rit != rend; ++rit) {
    if(rit->tagIndex_ != rit->realTagIndex_) {
      leftTgtBoundary = *translationTags[rit->tagIndex_].tagPosition_;
      break;
    }
  }
  return leftTgtBoundary;
}

WordAlignment::WordAlignment() {}

WordAlignment::WordAlignment(const std::vector<Point>& align) : data_(align) {}

WordAlignment::WordAlignment(const std::string& line, size_t srcEosPos, size_t tgtEosPos) {
  std::vector<std::string> atok = utils::splitAny(line, " -");
  for(size_t i = 0; i < atok.size(); i += 2)
    data_.push_back(Point{ (size_t)std::stoi(atok[i]), (size_t)std::stoi(atok[i + 1]), 1.f });
  data_.push_back(Point{ srcEosPos, tgtEosPos, 1.f }); // add alignment point for both EOS symbols
}

void WordAlignment::sort() {
  std::sort(data_.begin(), data_.end(), [](const Point& a, const Point& b) {
    return (a.srcPos == b.srcPos) ? a.tgtPos < b.tgtPos : a.srcPos < b.srcPos;
  });
}

void WordAlignment::normalize(bool reverse/*=false*/) {
  std::vector<size_t> counts;
  counts.reserve(data_.size());
  
  // reverse==false : normalize target word prob by number of source words
  // reverse==true  : normalize source word prob by number of target words
  auto srcOrTgt = [](const Point& p, bool reverse) {
    return reverse ? p.srcPos : p.tgtPos;
  };

  for(const auto& a : data_) {
    size_t pos = srcOrTgt(a, reverse);
    if(counts.size() <= pos)
      counts.resize(pos + 1, 0);
    counts[pos]++;
  }
  
  // a.prob at this point is either 1 or normalized to a different value,
  // but we just set it to 1 / count, so multiple calls result in re-normalization
  // regardless of forward or reverse direction. We also set the remaining values to 1.
  for(auto& a : data_) {
    size_t pos = srcOrTgt(a, reverse);
    if(counts[pos] > 1)
      a.prob = 1.f / counts[pos];
    else 
      a.prob = 1.f;
  }
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
                                          float threshold /*= 0.1f*/,
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
                   bool translationSpaceSymbolStart,
                   bool entitizeTags,
                   InputFormat inputFormat) {
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

  size_t elementId = 0;
  const size_t maxSrcPos = (!align.empty() && !align[0].empty()) ? align[0].size() - 1 : 0;
  // get hard alignments, sorted by source word position, by which lineTags is also sorted
  const auto hardAlignment = ConvertSoftAlignToHardAlign(
      align, 0.1f, true, true, lineSpaceSymbolStart, translationSpaceSymbolStart);
  // vector of tags to be reinserted in the translation: tuple consists of the iterator in the
  // source, the target position, and the opening tag target position if applicable of the beginning
  // of whatever nest there may be
  std::vector<TagPlacement> translationTags;
  std::vector<TagBalancingInfo> unbalancedOpenTags;
  // add a dummy open tag at the beginning to use for potential close tags that have no matching
  // opening tag
  translationTags.emplace_back(
      std::make_pair<Word, size_t>(
          Word::fromWordIndexAndTag(
              0, std::string(), std::string(), TagType::OPEN_TAG, TAGSPACING_NONE),
          0),
      elementId++,
      0);
  unbalancedOpenTags.emplace_back(TagBalancingInfo(translationTags.size() - 1, 0, std::string()));
  unbalancedOpenTags.back().fake_ = true;
  std::vector<std::pair<size_t, size_t>> trgTagRegions;
  auto curWordAlign = hardAlignment.begin();
  const size_t sentenceLength = words.size();
  for(auto lineTag = lineTags.begin(); lineTag != lineTags.end(); ++lineTag) {
    const auto& markupTag = lineTag->first.getMarkupTag();
    if(!markupTag) // error
      continue;

    for(; curWordAlign != hardAlignment.end(); ++curWordAlign) {
      if(curWordAlign->srcPos >= lineTag->second) {
        break;
      }
    }

    const std::string& tagIdentifier = markupTag->identifier();
    if(markupTag->type() != TagType::CLOSE_TAG) {
      if(lineTag->second == 0 && markupTag->type() == TagType::EMPTY_TAG) {
        // empty tag at beginning of source
        translationTags.emplace_back(*lineTag, elementId++, 0);
      } else if(lineTag->second == maxSrcPos && markupTag->type() == TagType::EMPTY_TAG) {
        // empty tag at end of source
        translationTags.emplace_back(*lineTag, elementId++, sentenceLength - 1);
      } else if(curWordAlign
                != hardAlignment.end() /*&& curWordAlign->srcPos == lineTag->second*/) {
        if((markupTag->spacing() & TAGSPACING_BEFORE) == 0
           && (markupTag->spacing() & TAGSPACING_BEFORE_IMMEDIATE_PRECEDING_TAG) == 0
           && ((markupTag->spacing() & TAGSPACING_AFTER) != 0
               || (markupTag->spacing() & TAGSPACING_AFTER_IMMEDIATE_FOLLOWING_TAG) != 0)
           && curWordAlign != hardAlignment.begin()) {
          // this is for self closing tags or opening tags of empty elements
          // looks like a closing tag as it hugs the previous word (no space separation)
          if(translationTags.empty()
             || translationTags.back().lineTag_.second != lineTag->second) {
            translationTags.emplace_back(
                *lineTag, elementId++, std::prev(curWordAlign)->tgtPos + 1);
          } else {
            translationTags.emplace_back(
                *lineTag, elementId++, *translationTags.back().tagPosition_);
          }
        } else {
          translationTags.emplace_back(*lineTag, elementId++, curWordAlign->tgtPos);
        }
      } else {
        // place at the end - but if nested, then later will put at end of the nest once the nest is
        // established
        translationTags.emplace_back(*lineTag, elementId++, sentenceLength);
      }
      unbalancedOpenTags.emplace_back(
          TagBalancingInfo(translationTags.size() - 1,
                           static_cast<size_t>(std::distance(hardAlignment.begin(), curWordAlign)),
                           tagIdentifier));
      if(markupTag->type() == TagType::EMPTY_TAG) {
        auto it = std::prev(unbalancedOpenTags.end());
        size_t leftTgtBoundary = getLeftTgtBoundary(
            unbalancedOpenTags.rbegin(), unbalancedOpenTags.rend(), translationTags);
        positionBalancedTags(*it,
                             leftTgtBoundary,
                             *lineTag,
                             hardAlignment,
                             std::distance(hardAlignment.begin(), curWordAlign),
                             sentenceLength,
                             maxSrcPos,
                             translationTags[it->tagIndex_].id_,
                             false,
                             translationTags,
                             New<size_t>(0),
                             trgTagRegions);
        unbalancedOpenTags.erase(it);
      }
    } else {
      // markupTag->type() == TagType::CLOSE_TAG
      bool unbalanced = true;
      size_t tagCount = translationTags.size();
      size_t trgTagRegionCount = trgTagRegions.size();
      auto unbalancedOpenTag = unbalancedOpenTags.begin();
      // use a single close tag position ptr for matching tag position (matchingTagPosition_)
      Ptr<size_t> closeTagPosition = New<size_t>(0);
      for(auto it = unbalancedOpenTags.rbegin(); it != unbalancedOpenTags.rend(); ++it) {
        size_t leftTgtBoundary = getLeftTgtBoundary(it, unbalancedOpenTags.rend(), translationTags);
        unbalancedOpenTag = std::prev(it.base());
        bool matchingClose = (tagIdentifier == it->tagIdentifier_);
        positionBalancedTags(
            *unbalancedOpenTag,
            leftTgtBoundary,
            *lineTag,
            hardAlignment,
            std::distance(hardAlignment.begin(), curWordAlign),
            sentenceLength,
            maxSrcPos,
            ((!it->artificial() && matchingClose) ? translationTags[it->tagIndex_].id_
                                                  : elementId++),
            false,
            translationTags,
            closeTagPosition,
            trgTagRegions);
        // this is for artificial tags - keep the original placement, will move if necessary later
        translationTags.back().tagPosition_ = New<size_t>(*closeTagPosition);
        if(matchingClose || it->fake_) {
          if(!it->fake_) {
            unbalanced = false;
          }
          break;
        }
        it->alignmentIndex_
            = static_cast<size_t>(std::distance(hardAlignment.begin(), curWordAlign));
        it->tagIndex_ = tagCount;
      }
      // set the real close tag position ptr
      translationTags.back().tagPosition_ = std::move(closeTagPosition); 
      size_t addedTags = translationTags.size() - tagCount;
      if(addedTags > 1) {
        // if the artifical close tags do not match the real close tag, move the open tags and those
        // within the nest to the position of the real close tag
        // also fix the matching (close) tag position
        for(size_t i = tagCount, j = 0; i < translationTags.size() - 1; ++i, ++j) {
          if(*translationTags[i].tagPosition_ != *translationTags.back().tagPosition_) {
            auto ubt = std::next(unbalancedOpenTags.rbegin(), j);
            for(size_t t = ubt->realTagIndex_; t < tagCount; ++t) {
              *translationTags[t].tagPosition_ = *translationTags.back().tagPosition_;
            }
          }
        }
        // remove all but the last tag
        translationTags.erase(std::prev(translationTags.end(), addedTags),
                              std::prev(translationTags.end()));
      }
      if(unbalancedOpenTag->realTagIndex_ != unbalancedOpenTag->tagIndex_) {
        for(size_t t = unbalancedOpenTag->realTagIndex_ + 1; t < translationTags.size() - 1; ++t) {
          translationTags[t].nests_.emplace_back(
              translationTags[unbalancedOpenTag->realTagIndex_].id_,
              translationTags[unbalancedOpenTag->realTagIndex_].tagPosition_,
              translationTags.back().tagPosition_);
        }
      }
      if(!unbalanced) {
        unbalancedOpenTags.erase(unbalancedOpenTag);
      }
      for(size_t i = trgTagRegionCount; i < trgTagRegions.size(); ++i) {
        if(trgTagRegions[i].second > tagCount) {
          trgTagRegions[i].second = tagCount;
        }
      }
    }
  }

  if(unbalancedOpenTags.size() > 1) {
    // received unbalanced input - opening tag has no closing tag - treat as if closing tags are
    // after the end of the segment
    std::pair<Word, size_t> dummyCloseTag(
        Word::fromWordIndexAndTag(
            0, std::string(), std::string(), TagType::CLOSE_TAG, TAGSPACING_NONE),
        maxSrcPos);
    auto rend = std::prev(unbalancedOpenTags.rend());
    size_t tagCount = translationTags.size();
    for(auto it = unbalancedOpenTags.rbegin(); it != rend; ++it) {
      size_t leftTgtBoundary = getLeftTgtBoundary(it, rend, translationTags);
      auto unbalancedOpenTag = std::prev(it.base());
      positionBalancedTags(*unbalancedOpenTag,
                           leftTgtBoundary,
                           dummyCloseTag,
                           hardAlignment,
                           hardAlignment.size(),
                           sentenceLength,
                           maxSrcPos,
                           (!it->artificial()) ? translationTags[it->tagIndex_].id_ : elementId++,
                           true,
                           translationTags,
                           New<size_t>(0),
                           trgTagRegions);
      if(unbalancedOpenTag->realTagIndex_ != unbalancedOpenTag->tagIndex_) {
        for(size_t t = unbalancedOpenTag->realTagIndex_ + 1; t < translationTags.size() - 1; ++t) {
          translationTags[t].nests_.emplace_back(
              translationTags[unbalancedOpenTag->realTagIndex_].id_,
              translationTags[unbalancedOpenTag->realTagIndex_].tagPosition_,
              translationTags.back().tagPosition_);
        }
      }
    }
    // remove all the fake added close tags
    translationTags.erase(std::next(translationTags.begin(), tagCount), translationTags.end());
  }

  // erase dummy open tag
  translationTags.erase(translationTags.begin());

  // make sure positions are before EOS token; those that are past were those that couldn't be
  // placed
  for(auto& t : translationTags) {
    if(*t.tagPosition_ == sentenceLength) {
      *t.tagPosition_ = sentenceLength - 1;
    }
  }

#if 0
  LOG(info, "words size: {}", sentenceLength);
  for(auto& t : translationTags) {
    std::ostringstream oss;
    for(auto n : t.nests_) {
      oss << "{id " << n.id_ << ", open pos " << *n.openTagPosition_ << ", close pos "
          << *n.closeTagPosition_ << "}, ";
    }
    LOG(info,
        "Tag {} : id: {}, pos: {}, matching tag pos: {}, nests size: {}, nests: {}",
        t.lineTag_.first.getMarkupTag()->tag(),
        t.id_,
        *t.tagPosition_,
        *t.matchingTagPosition_,
        t.nests_.size(),
        oss.str());
  }
#endif

  std::stable_sort(translationTags.begin(), translationTags.end());

  auto w = words.begin();
  for(auto t = translationTags.begin(); t != translationTags.end(); ++t) {
    auto x = words.begin() + *t->tagPosition_;
    if(std::distance(w, x) > 0) {
      wordsWithTags.insert(wordsWithTags.end(), w, x);
      w = x;
    }
    wordsWithTags.push_back(t->lineTag_.first);
  }

  wordsWithTags.insert(wordsWithTags.end(), w, words.end());
  return wordsWithTags;
}

}  // namespace data
}  // namespace marian
