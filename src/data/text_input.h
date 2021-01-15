#pragma once

#include "data/iterator_facade.h"
#include "data/corpus.h"

namespace marian {
namespace data {

class TextInput;

class TextIterator : public IteratorFacade<TextIterator, SentenceTuple const> {
public:
  TextIterator();
  explicit TextIterator(TextInput& corpus);

private:
  void increment() override;

  bool equal(TextIterator const& other) const override;

  const SentenceTuple& dereference() const override;

  TextInput* corpus_;

  long long int pos_;
  SentenceTuple tup_;
};

class TextInput : public DatasetBase<SentenceTuple, TextIterator, CorpusBatch> {
private:
  std::vector<UPtr<std::istringstream>> files_;
  std::vector<Ptr<Vocab>> vocabs_;

  size_t pos_{0};

public:
  typedef SentenceTuple Sample;

  TextInput(const std::vector<std::string>& inputs,
            std::vector<Ptr<Vocab>> vocabs,
            Ptr<Options> options);
  virtual ~TextInput() {}

  Sample next() override;

  void shuffle() override {}
  void reset() override {}

  iterator begin() override { return iterator(*this); }
  iterator end() override { return iterator(); }

  // TODO: There are half dozen functions called toBatch(), which are very
  // similar. Factor them.
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

    std::vector<std::vector<std::pair<Word, size_t>>> sentenceTags;
    std::vector<bool> sentenceSpaceSymbolStarts;
    std::vector<size_t> words(maxDims.size(), 0);
    for(size_t i = 0; i < batchSize; ++i) {
      sentenceTags.emplace_back();
      sentenceSpaceSymbolStarts.emplace_back(false);
      bool firstWordMet = false;
      for(size_t j = 0; j < maxDims.size(); ++j) {
        for(size_t k = 0, l = 0; k < batchVector[i][j].size(); ++k) {
          const auto& markupTag = batchVector[i][j][k].getMarkupTag();
          if(!markupTag) {
            subBatches[j]->data()[l * batchSize + i] = batchVector[i][j][k];
            subBatches[j]->mask()[l * batchSize + i] = 1.f;
            words[j]++;
            ++l;
            if(!firstWordMet) {
              firstWordMet = true;
              if(batchVector[i][j][k].isSpaceSymbol()) {
                sentenceSpaceSymbolStarts.back() = true;
              }
            }
          } else {
            sentenceTags.back().emplace_back(batchVector[i][j][k], l);
          }
        }
      }
    }

    for(size_t j = 0; j < maxDims.size(); ++j)
      subBatches[j]->setWords(words[j]);

    auto batch = batch_ptr(new batch_type(subBatches));
    batch->setSentenceIds(sentenceIds);
    batch->setSentenceTags(sentenceTags);
    batch->setSentenceSpaceSymbolStarts(sentenceSpaceSymbolStarts);

    return batch;
  }

  void prepare() override {}
};
}  // namespace data
}  // namespace marian
