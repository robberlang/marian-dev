#pragma once

#include <vector>

#include "common/definitions.h"
#include "data/types.h"

namespace marian {
namespace data {

class Batch {
public:
  virtual size_t size() const = 0;
  virtual size_t words(int /*which*/ = 0) const { return 0; };
  virtual size_t width() const { return 0; };

  virtual size_t sizeTrg() const { return 0; };
  virtual size_t wordsTrg() const { return 0; };
  virtual size_t widthTrg() const { return 0; };

  virtual void debug(bool /*printIndices*/ = false) {};

  virtual std::vector<Ptr<Batch>> split(size_t n, size_t sizeLimit = SIZE_MAX) = 0;

  const std::vector<size_t>& getSentenceIds() const { return sentenceIds_; }
  void setSentenceIds(const std::vector<size_t>& ids) { sentenceIds_ = ids; }

  const std::vector<std::vector<std::pair<Word, size_t>>>& getSentenceTags() const {
    return sentenceTags_;
  }
  void setSentenceTags(const std::vector<std::vector<std::pair<Word, size_t>>>& tags) {
    sentenceTags_ = tags;
  }

  const std::vector<bool>& getSentenceSpaceSymbolStarts() const {
    return sentenceSpaceSymbolStarts_;
  }
  void setSentenceSpaceSymbolStarts(const std::vector<bool>& sentenceSpaceSymbolStarts) {
    sentenceSpaceSymbolStarts_ = sentenceSpaceSymbolStarts;
  }

  virtual void setGuidedAlignment(std::vector<float>&&) = 0;
  virtual void setDataWeights(const std::vector<float>&) = 0;
  virtual ~Batch() {};
protected:
  std::vector<size_t> sentenceIds_;
  std::vector<std::vector<std::pair<Word, size_t>>> sentenceTags_;
  std::vector<bool> sentenceSpaceSymbolStarts_;
};
}  // namespace data
}  // namespace marian
