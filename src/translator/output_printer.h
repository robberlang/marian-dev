#pragma once

#include <vector>

#include "common/options.h"
#include "common/utils.h"
#include "data/alignment.h"
#include "data/vocab.h"
#include "translator/history.h"
#include "translator/hypothesis.h"

namespace marian {

class OutputPrinter {
public:
  OutputPrinter(Ptr<const Options> options, Ptr<const Vocab> vocab)
      : vocab_(vocab),
        reverse_(options->get<bool>("right-left")),
        nbest_(options->get<bool>("n-best", false)
                   ? options->get<size_t>("beam-size")
                   : 0),
        alignment_(options->get<std::string>("alignment", "")),
        alignmentThreshold_(getAlignmentThreshold(alignment_)),
        wordScores_(options->get<bool>("word-scores")),
        entitizeTags_(options->get<bool>("entitize-tags")),
        wordCounts_(options->get<bool>("word-counts")) {}

  template <class OStream>
  void print(Ptr<const History> history, OStream& best1, OStream& bestn) {
    const auto& nbl = history->nBest(nbest_);
    const auto& lineTags = history->getLineTags();
    bool lineSpaceSymbolStart = history->getLineSpaceSymbolStart();

    // prepare n-best list output
    for(size_t i = 0; i < nbl.size(); ++i) {
      const auto& result = nbl[i];
      const auto& hypo = std::get<1>(result);
      auto words = std::get<0>(result);
      bool translationSpaceSymbolStart = vocab_->sentenceStartsWithSpaceSymbolWord(words);

      auto align = getSoftAlignment(hypo);
      Words wordsWithTags
          = reinsertTags(words, align, lineTags, lineSpaceSymbolStart, translationSpaceSymbolStart);

      if(reverse_)
        std::reverse(wordsWithTags.begin(), wordsWithTags.end());

      std::string translation = vocab_->decode(wordsWithTags);
      bestn << history->getLineNum() << " ||| " << translation;

      if(wordCounts_)
        bestn << " ||| WordCounts=" << history->getNumSrcWords() << ',' << words.size();

      if(!alignment_.empty())
        bestn << " ||| " << getAlignment(align);

      if(wordScores_)
        bestn << " ||| WordScores=" << getWordScores(hypo);

      bestn << " |||";
      if(hypo->getScoreBreakdown().empty()) {
        bestn << " F0=" << hypo->getPathScore();
      } else {
        for(size_t j = 0; j < hypo->getScoreBreakdown().size(); ++j) {
          bestn << " F" << j << "= " << hypo->getScoreBreakdown()[j];
        }
      }

      float realScore = std::get<2>(result);
      bestn << " ||| " << realScore;

      if(i < nbl.size() - 1)
        bestn << std::endl;
      else
        bestn << std::flush;
    }

    auto result = history->top();
    auto words = std::get<0>(result);
    bool translationSpaceSymbolStart = vocab_->sentenceStartsWithSpaceSymbolWord(words);

    const auto& hypo = std::get<1>(result);
    auto align = getSoftAlignment(hypo);
    Words wordsWithTags
        = reinsertTags(words, align, lineTags, lineSpaceSymbolStart, translationSpaceSymbolStart);

    if(reverse_)
      std::reverse(wordsWithTags.begin(), wordsWithTags.end());

    std::string translation = vocab_->decode(wordsWithTags);

    best1 << translation;
    if(wordCounts_)
      best1 << " ||| WordCounts=" << history->getNumSrcWords() << ',' << words.size();

    if(!alignment_.empty()) {
      best1 << " ||| " << getAlignment(align);
    }

    if(wordScores_) {
      best1 << " ||| WordScores=" << getWordScores(hypo);
    }

    best1 << std::flush;
  }

private:
  Ptr<Vocab const> vocab_;
  bool reverse_{false};            // If it is a right-to-left model that needs reversed word order
  size_t nbest_{0};                // Size of the n-best list to print
  std::string alignment_;          // A non-empty string indicates the type of word alignment
  float alignmentThreshold_{0.f};  // Threshold for converting attention into hard word alignment
  bool wordScores_{false};         // Whether to print word-level scores or not
  bool entitizeTags_{false};
  bool wordCounts_{false};

  data::SoftAlignment getSoftAlignment(const Hypothesis::PtrType& hyp);
  // Get word alignment pairs or soft alignment
  std::string getAlignment(const data::SoftAlignment& align);
  // Get word-level scores
  std::string getWordScores(const Hypothesis::PtrType& hyp);
  Words reinsertTags(const Words& words,
                     const data::SoftAlignment& align,
                     const std::vector<std::pair<Word, size_t>>& lineTags,
                     bool lineSpaceSymbolStart,
                     bool translationSpaceSymbolStart);

  float getAlignmentThreshold(const std::string& str) {
    try {
      return std::max(std::stof(str), 0.f);
    } catch(...) {
      return 0.f;
    }
  }
};
}  // namespace marian
