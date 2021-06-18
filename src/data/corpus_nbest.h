#pragma once

#include <fstream>
#include <iostream>
#include <random>

#include "common/definitions.h"
#include "common/file_stream.h"
#include "common/options.h"
#include "data/alignment.h"
#include "data/batch.h"
#include "data/corpus_base.h"
#include "data/dataset.h"
#include "data/vocab.h"

namespace marian {
namespace data {

class CorpusNBest : public CorpusBase {
private:
  std::vector<size_t> ids_;
  int lastNum_{-1};
  std::vector<std::string> lastLines_;

public:
  // @TODO: check if translate can be replaced by an option in options
  CorpusNBest(Ptr<Options> options, bool translate = false);

  CorpusNBest(std::vector<std::string> paths,
              std::vector<Ptr<Vocab>> vocabs,
              Ptr<Options> options);

  Sample next() override;

  void shuffle() override {}

  void reset() override;

  void restore(Ptr<TrainingState>) override {}

  iterator begin() override { return iterator(this); }

  iterator end() override { return iterator(); }

  std::vector<Ptr<Vocab>>& getVocabs() override { return vocabs_; }

  batch_ptr toBatch(const std::vector<Sample>& batchVector) override {
    size_t batchSize = batchVector.size();

    std::vector<size_t> sentenceIds;

    std::vector<int> maxDims;
    for(auto& ex : batchVector) {
      if(maxDims.size() < ex.size())
        maxDims.resize(ex.size(), 0);
      for(size_t i = 0; i < ex.size(); ++i) {
        int numWords
            = static_cast<int>(std::count_if(ex[i].begin(), ex[i].end(), [](const Word& w) {
                return !w.getMarkupTag().operator bool();
              }));
        if(numWords > maxDims[i])
          maxDims[i] = numWords;
      }
      sentenceIds.push_back(ex.getId());
    }

    std::vector<Ptr<SubBatch>> subBatches;
    for(size_t j = 0; j < maxDims.size(); ++j) {
      subBatches.emplace_back(New<SubBatch>(batchSize, maxDims[j], vocabs_[j]));
    }

    std::vector<std::vector<size_t>> sentenceWordCounts(maxDims.size());
    std::vector<std::vector<std::vector<std::pair<Word, size_t>>>> sentenceTags(maxDims.size());
    std::vector<std::vector<bool>> sentenceSpaceSymbolStarts(maxDims.size());
    std::vector<size_t> words(maxDims.size(), 0);
    for(size_t i = 0; i < batchSize; ++i) {
      for(size_t j = 0; j < maxDims.size(); ++j) {
        sentenceWordCounts[j].emplace_back(0);
        sentenceTags[j].emplace_back();
        sentenceSpaceSymbolStarts[j].emplace_back(false);
        bool firstWordMet = false;
        for(size_t k = 0, l = 0; k < batchVector[i][j].size(); ++k) {
          const auto& markupTag = batchVector[i][j][k].getMarkupTag();
          if(!markupTag) {
            subBatches[j]->data()[l * batchSize + i] = batchVector[i][j][k];
            subBatches[j]->mask()[l * batchSize + i] = 1.f;
            words[j]++;
            ++sentenceWordCounts[j].back();
            ++l;
            if(!firstWordMet) {
              firstWordMet = true;
              if(batchVector[i][j][k].isSpaceSymbol()) {
                sentenceSpaceSymbolStarts[j].back() = true;
              }
            }
          } else {
            sentenceTags[j].back().emplace_back(batchVector[i][j][k], l);
          }
        }
      }
    }

    for(size_t j = 0; j < maxDims.size(); ++j) {
      subBatches[j]->setWords(words[j]);
      subBatches[j]->setSentenceWordCounts(sentenceWordCounts[j]);
      subBatches[j]->setSentenceTags(sentenceTags[j]);
      subBatches[j]->setSentenceSpaceSymbolStarts(sentenceSpaceSymbolStarts[j]);
    }

    auto batch = batch_ptr(new batch_type(subBatches));
    batch->setSentenceIds(sentenceIds);

    return batch;
  }
};
}  // namespace data
}  // namespace marian
