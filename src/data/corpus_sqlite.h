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

#include <SQLiteCpp/SQLiteCpp.h>
#include <SQLiteCpp/sqlite3/sqlite3.h>

static void SQLiteRandomSeed(sqlite3_context* context,
                             int argc,
                             sqlite3_value** argv) {
  if(argc == 1 && sqlite3_value_type(argv[0]) == SQLITE_INTEGER) {
    const int seed = sqlite3_value_int(argv[0]);
    static std::default_random_engine eng(seed);
    std::uniform_int_distribution<> unif;
    const int result = unif(eng);
    sqlite3_result_int(context, result);
  } else {
    sqlite3_result_error(context, "Invalid", 0);
  }
}

namespace marian {
namespace data {

class CorpusSQLite : public CorpusBase {
private:
  UPtr<SQLite::Database> db_;
  UPtr<SQLite::Statement> select_;

  void fillSQLite();

  size_t seed_;

public:
  // @TODO: check if translate can be replaced by an option in options
  CorpusSQLite(Ptr<Options> options, bool translate = false);

  CorpusSQLite(const std::vector<std::string>& paths,
               const std::vector<Ptr<Vocab>>& vocabs,
               Ptr<Options> options);

  Sample next() override;

  void shuffle() override;

  void reset() override;

  void restore(Ptr<TrainingState>) override;

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

    if(options_->has("guided-alignment") && alignFileIdx_)
      addAlignmentsToBatch(batch, batchVector);
    if(options_->hasAndNotEmpty("data-weighting") && weightFileIdx_)
      addWeightsToBatch(batch, batchVector);

    return batch;
  }

private:
  void createRandomFunction() {
    sqlite3_create_function(db_->getHandle(),
                            "random_seed",
                            1,
                            SQLITE_UTF8,
                            NULL,
                            &SQLiteRandomSeed,
                            NULL,
                            NULL);
  }
};
}  // namespace data
}  // namespace marian
