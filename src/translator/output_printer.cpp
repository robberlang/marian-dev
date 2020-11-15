#include "output_printer.h"

#include <sstream>

namespace marian {

  struct TagPosition {
  size_t id_;
  size_t pos_;
  std::ptrdiff_t span_;
  TagPosition(size_t id, size_t pos, std::ptrdiff_t span) : id_(id), pos_(pos), span_(span) {}
  };

  struct TagPlacement {
  std::vector<std::pair<Word, size_t>>::const_iterator wordAlign_;
  TagPosition tagPosition_;
  std::vector<TagPosition> nests_;

  TagPlacement(std::vector<std::pair<Word, size_t>>::const_iterator wordAlign,
               size_t id,
               size_t pos,
               std::ptrdiff_t span)
      : wordAlign_(wordAlign), tagPosition_(id, pos, span) {}
  };

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
  const auto hardAlignment = data::ConvertSoftAlignToHardAlign(align, 1.f, true);
  // vector of tags to be reinserted in the translation: tuple consists of the iterator in the
  // source, the target position, and the opening tag target position if applicable of the beginning
  // of whatever nest there may be
  std::vector<TagPlacement> translationTags;
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
        translationTags.emplace_back(lineTag, translationTags.size(), 0, 1);
      } else if(curWordAlign != hardAlignment.end() /*&& curWordAlign->srcPos == lineTag->second*/) {
        translationTags.emplace_back(lineTag, translationTags.size(), curWordAlign->tgtPos, 1);
      } else {
        // place at the end - but if nested, then later will put at end of the nest once the nest is
        // established
        translationTags.emplace_back(lineTag, translationTags.size(), words.size(), 1);
      }

      if(markupTag->getType() == TagType::OPEN_TAG)
        unbalancedOpenTags.emplace_back(translationTags.size() - 1,
                                        std::distance(hardAlignment.begin(), curWordAlign));
    } else {
      if(!unbalancedOpenTags.empty()) {
        if(lineTag->second == maxSrcPos
           && translationTags[unbalancedOpenTags.back().first].wordAlign_->second == 0) {
          // this is the case where the tag encloses the entire source
          translationTags[unbalancedOpenTags.back().first].tagPosition_.pos_ = 0;
          translationTags[unbalancedOpenTags.back().first].tagPosition_.span_ = words.size() + 1;
          for(size_t t = unbalancedOpenTags.back().first + 1; t < translationTags.size(); ++t) {
            translationTags[t].nests_.emplace_back(
                unbalancedOpenTags.back().first, 0, words.size() + 1);
          }
          translationTags.emplace_back(
              lineTag, unbalancedOpenTags.back().first, words.size(), -1 - words.size());
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
              translationTags.emplace_back(
                  lineTag, unbalancedOpenTags.back().first, maxTgtPos, minTgtPos - maxTgtPos - 1);
              trgTagRegions.emplace_back(minTgtPos, maxTgtPos);
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
      // if there are tags nested then have them remain nested
      for(size_t t = u->first + 1; t < translationTags.size(); ++t) {
        if(translationTags[t].tagPosition_.pos_ < translationTags[u->first].tagPosition_.pos_) {
          translationTags[u->first].tagPosition_.pos_ = translationTags[t].tagPosition_.pos_;
        }
      }
      for(auto a = hardAlignment.begin() + u->second; a != hardAlignment.end(); ++a) {
        if(a->tgtPos < translationTags[u->first].tagPosition_.pos_) {
          translationTags[u->first].tagPosition_.pos_ = a->tgtPos;
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
    wordsWithTags.push_back(t->wordAlign_->first);
  }

  wordsWithTags.insert(wordsWithTags.end(), w, words.end());
  for(; lineTag != lineTags.end(); ++lineTag) {
    wordsWithTags.emplace_back(lineTag->first);
  }
  return wordsWithTags;
}

}  // namespace marian
