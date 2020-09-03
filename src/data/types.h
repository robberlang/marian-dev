#pragma once

#include "common/definitions.h"

#include <cstdint>
#include <cstdlib>
#include <string>
#include <unordered_map>
#include <vector>
#include <iterator>

namespace marian {

enum class InputFormat : char {
  PLAINTEXT = 0,
  XLIFF1 = 1,
  HTML = 2,
};

static inline InputFormat ConvertInputFormat(const std::string& inputFormat) {
  if(inputFormat == "xliff1")
    return InputFormat::XLIFF1;
  if(inputFormat == "html")
    return InputFormat::HTML;
  return InputFormat::PLAINTEXT;
}

enum class TagType : char {
  NONE = 0,
  OPEN_TAG = 1,
  CLOSE_TAG = 2,
  EMPTY_TAG = 3,
};

class MarkupTag {
  std::string tag_;
  TagType type_;

public:
  MarkupTag(std::string tag, TagType type) : tag_(std::move(tag)), type_(type) {}
  const std::string& getTag() const { return tag_; }
  const TagType& getType() const { return type_; }
};

// Type for all vocabulary items, based on IndexType
typedef IndexType WordIndex;    // WordIndex is used for words or tokens arranged in consecutive order
class Word {                    // Word is an abstraction of a unique id, not necessarily consecutive
  WordIndex wordId_;
  Ptr<MarkupTag> markupTag_;
  explicit Word(std::size_t wordId) : wordId_((WordIndex)wordId) {}
  explicit Word(std::size_t wordId, const std::string& tag, TagType tagType)
      : wordId_((WordIndex)wordId), markupTag_(New<MarkupTag>(tag, tagType)) {}

public:
  static Word fromWordIndex(std::size_t wordId) { return Word(wordId); }
  static Word fromWordIndexAndTag(std::size_t wordId, const std::string& tag, TagType tagType) {
    return Word(wordId, tag, tagType);
  }
  const WordIndex& toWordIndex() const { return wordId_; }
  void setWordIndex(WordIndex wordId) { wordId_ = wordId; }
  const Ptr<MarkupTag>& getMarkupTag() const { return markupTag_; };
  std::string toString() const { return std::to_string(wordId_); }

  // needed for STL containers
  Word() : wordId_((WordIndex)-1) {}
  bool operator==(const Word& other) const { return wordId_ == other.wordId_; }
  bool operator!=(const Word& other) const { return !(*this == other); }
  bool operator<(const Word& other) const { return wordId_ < other.wordId_; }
  std::size_t hash() const { return std::hash<WordIndex>{}(wordId_); }

  // constants
  static Word NONE; // @TODO: decide whether we need this, in additional Word()
  static Word ZERO; // an invalid word that nevertheless can safely be looked up (and then masked out)
  // EOS and UNK are placed in these positions in Marian-generated vocabs
  static Word DEFAULT_EOS_ID;
  static Word DEFAULT_UNK_ID;
};

// Sequence of vocabulary items
typedef std::vector<Word> Words;

// Helper to map a Word vector to a WordIndex vector
static inline std::vector<WordIndex> toWordIndexVector(const Words& words) {
  std::vector<WordIndex> res;
  std::transform(words.begin(), words.end(), std::back_inserter(res),
                 [](const Word& word) -> WordIndex { return word.toWordIndex(); });
  return res;
}

// names of EOS and UNK symbols
const std::string DEFAULT_EOS_STR = "</s>";
const std::string DEFAULT_UNK_STR = "<unk>";

// alternatively accepted names in Yaml dictionaries for ids 0 and 1, resp.
const std::string NEMATUS_EOS_STR = "eos";
const std::string NEMATUS_UNK_STR = "UNK";

}  // namespace marian

namespace std {
  template<> struct hash<marian::Word> {
    std::size_t  operator()(const marian::Word& s) const noexcept { return s.hash(); }
  };
}
