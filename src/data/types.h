#pragma once

#include "common/definitions.h"

/* absl::string_view is a type used within the generic vocab implementation.
 * However, currently included from within sentencepiece to avoid an extra
 * dependency. 
 */

#include "sentencepiece/third_party/absl/strings/string_view.h"

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

const char TAGSPACING_NONE = 0x0;
const char TAGSPACING_BEFORE = 0x1;
const char TAGSPACING_AFTER = 0x2;
const char TAGSPACING_WITHIN = 0x4;
const char TAGSPACING_BEFORE_IMMEDIATE_PRECEDING_TAG = 0x8;
const char TAGSPACING_AFTER_IMMEDIATE_FOLLOWING_TAG = 0x10;

class MarkupTag {
  std::string tag_;
  std::string tagIdentifier_;
  std::string elementContent_;
  TagType type_;
  char spacing_;

public:
  MarkupTag(std::string tag, std::string tagIdentifier, TagType type, char spacing)
      : tag_(std::move(tag)),
        tagIdentifier_(std::move(tagIdentifier)),
        type_(type),
        spacing_(spacing) {}
  const std::string& tag() const { return tag_; }
  std::string& tag() { return tag_; }
  const std::string& tagIdentifier() const { return tagIdentifier_; }
  std::string& tagIdentifier() { return tagIdentifier_; }
  const std::string& elementContent() const { return elementContent_; }
  std::string& elementContent() { return elementContent_; }
  const TagType& type() const { return type_; }
  TagType& type() { return type_; }
  char spacing() const { return spacing_; }
  char& spacing() { return spacing_; }
};

// Type for all vocabulary items, based on IndexType
typedef IndexType WordIndex;    // WordIndex is used for words or tokens arranged in consecutive order
class Word {                    // Word is an abstraction of a unique id, not necessarily consecutive
  WordIndex wordId_;
  Ptr<MarkupTag> markupTag_;
  Ptr<std::string> surface_;
  bool isSpaceSymbol_{false};
  bool isSpecialSymbol_{false};
  explicit Word(std::size_t wordId, bool isSpaceSymbol = false, bool isSpecialSymbol = false)
      : wordId_((WordIndex)wordId), isSpaceSymbol_(isSpaceSymbol), isSpecialSymbol_(isSpecialSymbol) {}
  explicit Word(std::size_t wordId,
                std::string surface,
                bool isSpaceSymbol = false,
                bool isSpecialSymbol = false)
      : wordId_((WordIndex)wordId),
        surface_(New<std::string>(std::move(surface))),
        isSpaceSymbol_(isSpaceSymbol),
        isSpecialSymbol_(isSpecialSymbol) {}
  explicit Word(std::size_t wordId,
                std::string tag,
                std::string tagIdentifier,
                TagType tagType,
                char tagSpacing)
      : wordId_((WordIndex)wordId),
        markupTag_(New<MarkupTag>(std::move(tag), std::move(tagIdentifier), tagType, tagSpacing)) {}

public:
  static Word fromWordIndex(std::size_t wordId,
                            bool isSpaceSymbol = false,
                            bool isSpecialSymbol = false) {
    return Word(wordId, isSpaceSymbol, isSpecialSymbol);
  }
  static Word fromWordIndex(std::size_t wordId,
                            std::string surface,
                            bool isSpaceSymbol = false,
                            bool isSpecialSymbol = false) {
    return Word(wordId, std::move(surface), isSpaceSymbol, isSpecialSymbol);
  }
  static Word fromWordIndexAndTag(std::size_t wordId,
                                  const std::string& tag,
                                  const std::string& tagIdentifier,
                                  TagType tagType,
                                  char tagSpacing) {
    return Word(wordId, tag, tagIdentifier, tagType, tagSpacing);
  }
  const WordIndex& toWordIndex() const { return wordId_; }
  void setWordIndex(WordIndex wordId) { wordId_ = wordId; }
  bool isSpaceSymbol() const { return isSpaceSymbol_; }
  void setIsSpaceSymbol(bool isSpaceSymbol) { isSpaceSymbol_ = isSpaceSymbol; }
  bool isSpecialSymbol() const { return isSpecialSymbol_; }
  void setIsSpecialSymbol(bool isSpecialSymbol) { isSpecialSymbol_ = isSpecialSymbol; }
  const Ptr<MarkupTag>& getMarkupTag() const { return markupTag_; };
  const Ptr<std::string>& getSurface() const { return surface_; };
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

typedef absl::string_view string_view;

}  // namespace marian

namespace std {
  template<> struct hash<marian::Word> {
    std::size_t  operator()(const marian::Word& s) const noexcept { return s.hash(); }
  };
}
