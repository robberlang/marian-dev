#include "data/vocab_base.h"

#ifdef USE_SENTENCEPIECE
#include "sentencepiece/src/sentencepiece_processor.h"
#include "sentencepiece/src/sentencepiece_trainer.h"
/* https://github.com/google/googletest/issues/1063#issuecomment-332518392 */
#if __GNUC__ >= 5
// Disable GCC 5's -Wsuggest-override warnings in gtest
# pragma GCC diagnostic push
# pragma GCC diagnostic ignored "-Wsuggest-override"
#endif

/* Current inclusion of SentencePiece structures assume builtin-protobuf hard. 
 * Future TODO: Make it work with standard protobuf as well.
 * */
#include "sentencepiece/src/builtin_pb/sentencepiece.pb.h"

#if __GNUC__ >= 5
# pragma GCC diagnostic pop
#endif
#endif

#include "common/config.h"
#include "common/options.h"
#include "common/logging.h"
#include "common/filesystem.h"
#include "common/regex.h"
#include "common/tag_finder.h"
#include "common/utils.h"
#include "common/char_entities.h"
#include "common/unicode_char_props.h"

#include <sstream>
#include <random>
#include <stack>
#include <tuple>
#include <functional>
#include <cctype>
#include <cassert>

namespace marian {

#ifdef USE_SENTENCEPIECE

// Wrapper around https://github.com/google/sentencepiece
class SentencePieceVocab : public IVocab {
private:
  // Actual SentencePiece processor object
  UPtr<sentencepiece::SentencePieceProcessor> spm_;

  // Sampling factor for subword regularization, disabled when 0
  float alpha_{0};

  // Allowed suffixes for SentencePiece model
  std::vector<std::string> suffixes_ = {".spm"};

  Ptr<Options> options_;
  size_t batchIndex_{0};

  std::mt19937 generator_;
  std::uniform_int_distribution<int> randInt_; // from 0 to INT_MAX

  // Keeps sentences segmented into subword units
  bool keepEncoded_{false};

  Word termTokenS_;
  Word termTokenOpenC_;
  Word termTokenCloseC_;
  Word spaceToken_;
  bool hasTerminologyConstraints_{false};

  // Contains control characters added to vocab due to byte-fallback
  std::vector<Word> controlChars_;

  // Creates the first 32 control characters as done in byte-fallback and checks if they exist in the vocab.
  // This makes sure that we do not waste computational effort on suppression if they don't actually appear.
  void populateControlChars() {
    for(int i = 0; i < 32; ++i) {
      std::string bytePiece = fmt::format("<0x{:02X}>", i); // 0 becomes <0x00>, 10 becomes <0x0A>, note uppercase A and lowercase x
      auto id = spm_->PieceToId(bytePiece);
      if(id != spm_->unk_id())
        controlChars_.push_back(Word::fromWordIndex(id));
    }
  }
  // Sample from one file, based on first algorithm from:
  // https://en.wikipedia.org/wiki/Reservoir_sampling
  void reservoirSampling(std::vector<std::string>& sample, size_t& seenLines,
                        const std::string& trainPath, size_t maxLines, size_t maxBytes) {
    ABORT_IF(maxLines == 0, "Sample needs to be larger 0");

    std::unique_ptr<std::istream> trainStrm(trainPath == "stdin"
                                                ? new std::istream(std::cin.rdbuf())
                                                : new io::InputFileStream(trainPath));

    std::string line;
    while(getline(*trainStrm, line)) {
      if(line.size() > 0 && line.size() < maxBytes) {
        if(sample.size() < maxLines) {
          sample.push_back(line);
        }
        else {
          size_t i = randInt_(generator_) % (seenLines + 1);
          if(i < maxLines)
            sample[i] = line;
        }
        seenLines++;
      }
    }
  }

  // Iterate over all input files and collect a representative sample via reservoir sampling.
  // The sample will first grow to the desired size and next keep sampling with decreasing
  // probability in the hope to get a uniform sample from the union of all files.
  size_t reservoirSamplingAll(io::TemporaryFile& temp,
                             const std::vector<std::string>& trainPaths,
                             size_t maxLines, size_t maxBytes) {
    LOG(info, "[SentencePiece] Sampling at most {} lines from {}", maxLines, utils::join(trainPaths, ", "));

    std::vector<std::string> sample;
    size_t seenLines = 0;
    for(const auto& trainPath : trainPaths)
      reservoirSampling(sample, seenLines, trainPath, maxLines, maxBytes);
    std::shuffle(sample.begin(), sample.end(), generator_);

    for(const auto& line : sample)
        temp << line << std::endl;

    LOG(info, "[SentencePiece] Selected {} lines", sample.size());
    return sample.size();
  }

  // Just concatenate all files to a temporary file so SentencePiece can consume it.
  size_t dumpAll(io::TemporaryFile& temp,
                 const std::vector<std::string>& trainPaths,
                 size_t maxBytes) {
    LOG(info, "[SentencePiece] Selecting all lines from {}", utils::join(trainPaths, ", "));

    size_t seenLines = 0;
    std::string line;
    for(const auto& trainPath : trainPaths) {
      io::InputFileStream in(trainPath);
      while(getline(in, line)) {
        if(line.size() > 0 && line.size() < maxBytes) {
          temp << line << std::endl;
          seenLines++;
        }
      }
    }

    LOG(info, "[SentencePiece] Selected {} lines", seenLines);
    return seenLines;
  }

public:
  SentencePieceVocab(Ptr<Options> options, size_t batchIndex)
      : options_(options),
        batchIndex_(batchIndex),
        generator_((uint32_t)Config::seed),
        keepEncoded_(options->get<bool>("no-spm-decode", false)) {
    if(options_->has("sentencepiece-alphas")) {
      auto alphas = options_->get<std::vector<float>>("sentencepiece-alphas");
      if(alphas.size() <= batchIndex)
        alpha_ = 0.f;
      else
        alpha_ = alphas[batchIndex_];

      if(alpha_ > 0)
        LOG(debug,
            "Setting SentencePiece vocabulary sampling factor to {} for input {}",
            alpha_,
            batchIndex_);
    }
  }

  virtual const std::string& canonicalExtension() const override { return suffixes_[0]; }
  virtual const std::vector<std::string>& suffixes() const override { return suffixes_; }

  virtual std::string suffix() { return suffixes_[0]; };

  virtual std::string type() const override { return "SentencePieceVocab"; }

  virtual Word getEosId() const override { return Word::fromWordIndex(spm_->eos_id()); }
  virtual Word getUnkId() const override { return Word::fromWordIndex(spm_->unk_id()); }

  void create(const std::string& vocabPath,
              const std::vector<std::string>& trainPaths,
              size_t maxSize) override {

    size_t defaultMaxSize = 32000;
    size_t maxLines = options_->get<size_t>("sentencepiece-max-lines");
    size_t maxBytes = 2048;

    LOG(info, "[SentencePiece] Training SentencePiece vocabulary {}", vocabPath);

    if(maxSize == 0) {
      LOG(info, "[SentencePiece] Vocabulary size is undefined (set with --dim-vocabs ...) - setting to {}", defaultMaxSize);
      maxSize = defaultMaxSize;
    }

    // Create temporary file to hold the sample for the SentencePiece trainer
    io::TemporaryFile temp(options_->get<std::string>("tempdir"), false);
    std::string tempFileName = temp.getFileName();
    LOG(info, "[SentencePiece] Creating temporary file {}", tempFileName);

    size_t seenLines = 0;
    if(maxLines == 0)
      seenLines = dumpAll(temp, trainPaths, maxBytes);
    else
      seenLines = reservoirSamplingAll(temp, trainPaths, maxLines, maxBytes);

    // Compose the SentencePiece training command from filenames and parameters0
    std::stringstream command;
    command
      << " --bos_id=-1 --eos_id=0 --unk_id=1" // these should not be changed as they match Marian defaults
      << " --input="               << tempFileName
      << " --model_prefix="        << vocabPath
      << " --vocab_size="          << maxSize
      << " --max_sentence_length=" << maxBytes
      << " --input_sentence_size=" << seenLines
      << " " << options_->get<std::string>("sentencepiece-options"); // these are SentencePiece command line options

    // Train the SentencePiece model
    const auto status = sentencepiece::SentencePieceTrainer::Train(command.str());
    ABORT_IF(!status.ok(),
             "SentencePiece vocabulary error: {}",
             status.ToString());

    LOG(info, "[SentencePiece] Removing {}", vocabPath + ".vocab");
    ABORT_IF(remove((vocabPath + ".vocab").c_str()) != 0,
             "Could not remove {}",
             vocabPath + ".vocab");

    LOG(info, "[SentencePiece] Renaming {} to {}", vocabPath + ".model", vocabPath);
    ABORT_IF(rename((vocabPath + ".model").c_str(), vocabPath.c_str()) != 0,
             "Could not rename {} to {}",
             vocabPath + ".model", vocabPath);
  }

  void createFake() override {
    ABORT("[SentencePiece] Fake SentencePiece vocabulary not supported");
  }

  Word operator[](const std::string& token) const override {
    return Word::fromWordIndex(spm_->PieceToId(token), token);
  }

  const std::string& operator[](const Word& id) const override {
    ABORT_IF(id.toWordIndex() >= size(), "Unknown word id: ", id.toWordIndex());
    return spm_->IdToPiece(id.toWordIndex());
  }

  static bool checkAndMoveToCloseTag(const std::string& line,
                                     const std::string& tagName,
                                     size_t& tagEnd,
                                     const std::string& tagNameToCompare) {
    if(!tagName.compare(tagNameToCompare)) {
      if(line[tagEnd - 1] != '/') {
        const std::string tagClose = "</" + tagName + ">";
        size_t endTagEnd = line.find(tagClose, tagEnd + 1);
        if(endTagEnd != std::string::npos) {
          tagEnd = endTagEnd + tagClose.length() - 1;
        }
      }
      return true;
    }
    return false;
  }

  static std::string encodeSpecialChars(const std::string& text) {
    std::string encoded;
    size_t p = 0, q = 0;
    for(; (p = text.find_first_of("&<>", q)) != std::string::npos; q = p + 1) {
      encoded.append(text, q, p - q);
      encoded += '&';
      if(text[p] == '&') {
        encoded.append("amp");
      } else if(text[p] == '<') {
        encoded.append("lt");
      } else {
        encoded.append("gt");
      }
      encoded += ';';
    }

    encoded.append(text, q, p - q);
    return encoded;
  }

  static std::string decodeEntities(const std::string& text, InputFormat inputFormat) {
    std::string decoded;
    size_t p = 0, q = 0;
    while((p = text.find('&', q)) != std::string::npos) {
      decoded.append(text, q, p - q);
      bool validEntity = false;
      if((q = text.find(';', p + 1)) != std::string::npos) {
        if(text[p + 1] == '#') {
          if(text[p + 2] == 'x' || (text[p + 2] == 'X' && inputFormat == InputFormat::HTML)) {
            std::string entity(text, p + 3, q - p - 3);
            validEntity = !entity.empty();
            for(auto c : entity) {
              if(!((c >= 'A' && c <= 'F') || (c >= 'a' && c <= 'f') || (c >= '0' && c <= '9'))) {
                validEntity = false;
                break;
              }
            }
            if(validEntity) {
              int numeric = std::stoi(entity, nullptr, 16);
              validEntity = (numeric < 0x110000);
              if(validEntity) {
                charentities::EntityRep ent_rep(numeric);
                decoded.append(ent_rep.str());
              }
            }
          } else {
            std::string entity(text, p + 2, q - p - 2);
            validEntity = !entity.empty();
            for(auto c : entity) {
              if(!(c >= '0' && c <= '9')) {
                validEntity = false;
                break;
              }
            }
            if(validEntity) {
              int numeric = std::stoi(entity);
              validEntity = (numeric < 0x110000);
              if(validEntity) {
                charentities::EntityRep ent_rep(numeric);
                decoded.append(ent_rep.str());
              }
            }
          }
        } else {
          std::string entity(text, p + 1, q - p - 1);
          std::string rep_char;
          if(inputFormat == InputFormat::HTML) {
            rep_char = charentities::getCharRepOfEntity(entity.c_str()).str();
          } else {
            if(entity == "amp") {
              rep_char = '&';
            } else if(entity == "lt") {
              rep_char = '<';
            } else if(entity == "gt") {
              rep_char = '>';
            } else if(entity == "quot") {
              rep_char = '"';
            } else if(entity == "apos") {
              rep_char = '\'';
            }
          }

          validEntity = !rep_char.empty();
          if(validEntity) {
            decoded.append(rep_char);
          }
        }
      }

      if(validEntity) {
        ++q;
      } else {
        decoded += '&';
        q = p + 1;
      }
    }

    decoded.append(text, q, p - q);
    return decoded;
  }

  static std::string markupTagIdentifier(const std::string& tag,
                                         TagType tagType,
                                         InputFormat inputFormat) {
    std::string tagIdentifier;
    if(tagType != TagType::EMPTY_TAG) {
      if(inputFormat == InputFormat::XLIFF1) {
        std::vector<std::pair<std::string, std::string>> attributes;
        // XLIFF open/close tags (bpt/bx, ept/ex) should have a rid attribute, but use tag name
        // otherwise (so it works for non-XLIFF tags as well)
        tagfinder::findTagEnd(tag, 0, &attributes, &tagIdentifier);
        for(const auto& att : attributes) {
          if(att.first == "rid") {
            tagIdentifier = att.second;
            break;
          }
        }
      } else {
        tagfinder::findTagEnd(
            tag, 0, (std::vector<std::pair<std::string, std::string>>*)nullptr, &tagIdentifier);
      }
    }
    return tagIdentifier;
  }

  bool isWordSpaceSymbol(const Word& word) const {
    return (word == spaceToken_);
  }

  bool wordStartsWithAlpha(const Word& word) const {
    std::u32string strU = utils::utf8ToUnicodeString((*this)[word]);
    return !strU.empty() && !unicodecharprops::isUCharNonSpacing(strU.front())
           && unicodecharprops::isUCharAlpha(strU.front());
  }

  bool wordEndsWithAlpha(const Word& word) const {
    std::u32string strU = utils::utf8ToUnicodeString((*this)[word]);
    return !strU.empty() && !unicodecharprops::isUCharNonSpacing(strU.back())
           && unicodecharprops::isUCharAlpha(strU.back());
  }

  void encodeMarkupText(const std::string& text,
                        InputFormat inputFormat,
                        bool entitizeTags,
                        unsigned char tagSpacing,
                        bool afterTag,
                        bool specialSymbols,
                        sentencepiece::normalizer::AddDummyPrefix& addDummyPrefix,
                        Words& words) const {
    if(!entitizeTags) {
      if(addDummyPrefix == sentencepiece::normalizer::AddDummyPrefix::OFF && text.front() == ' ') {
        addDummyPrefix = sentencepiece::normalizer::AddDummyPrefix::ON;
      }
    } else {
      if(addDummyPrefix != sentencepiece::normalizer::AddDummyPrefix::DEFAULT) {
        addDummyPrefix = (text.front() != ' ') ? sentencepiece::normalizer::AddDummyPrefix::OFF
                                               : sentencepiece::normalizer::AddDummyPrefix::ON;
      }
    }
    std::string textPlain = decodeEntities(text, inputFormat);
    sentencepiece::SentencePieceText spt;
    spm_->Encode(textPlain, &spt, addDummyPrefix);
    int numPieces = spt.pieces_size();
    if(numPieces > 0) {
      const auto& firstSp = spt.pieces(0);
      size_t beginOffset = 0;
      std::string firstSurface(textPlain, beginOffset, firstSp.end() - beginOffset);
      beginOffset = firstSp.end();
      Word firstWord(Word::fromWordIndex(firstSp.id(), firstSurface, false, specialSymbols));
      if(textPlain.front() == ' ' || addDummyPrefix != sentencepiece::normalizer::AddDummyPrefix::ON
         || !afterTag || !isWordSpaceSymbol(firstWord)) {
        words.push_back(firstWord);
      } else {
        // put the space before the tag since that is where it originates
        words.emplace(words.end() - 1, firstWord);
      }
      for(int k = 1; k < numPieces; ++k) {
        const auto& sp = spt.pieces(k);
        std::string surface(textPlain, beginOffset, sp.end() - beginOffset);
        beginOffset = sp.end();
        if(k == numPieces - 1 && beginOffset < textPlain.length())
          surface.append(textPlain, beginOffset);
        words.push_back(Word::fromWordIndex(sp.id(), surface, false, specialSymbols));
      }
    }
    addDummyPrefix = (!std::isspace(static_cast<unsigned char>(textPlain.back()))
                      && (tagSpacing & TAGSPACING_WITHIN) == 0)
                         ? sentencepiece::normalizer::AddDummyPrefix::OFF
                         : sentencepiece::normalizer::AddDummyPrefix::ON;
  }

  Word encodeSpecialSymbol(const std::string& symbol) const {
    int pieceId = spm_->PieceToId(symbol);
    if(pieceId != spm_->unk_id()) {
      return Word::fromWordIndex(pieceId, symbol, false, true);
    }
    return Word();
  }

  Words encode(const std::string& line,
               bool addEOS,
               bool inference,
               InputFormat inputFormat,
               bool entitizeTags,
               const std::string& srcLang,
               const std::string& trgLang) const override {
    Words words;
    std::vector<int> spmIds;
    if(!srcLang.empty()) {
      Word wrd(encodeSpecialSymbol("<" + srcLang + ">"));
      if(wrd.toWordIndex() != (WordIndex)-1) {
        words.push_back(wrd);
      } else {
        LOG(info, "[SentencePiece] Could not encode expected special symbol <{}>", srcLang);
      }
    }
    if(!trgLang.empty()) {
      Word wrd(encodeSpecialSymbol("<" + trgLang + ">"));
      if(wrd.toWordIndex() != (WordIndex)-1) {
        words.push_back(wrd);
      } else {
        LOG(info, "[SentencePiece] Could not encode expected special symbol <{}>", trgLang);
      }
    }
    if(inference || alpha_ == 0) {
      if(!inference || inputFormat == InputFormat::PLAINTEXT) {
        spm_->Encode(line, &spmIds);
      } else {
        sentencepiece::normalizer::AddDummyPrefix addDummyPrefix
            = sentencepiece::normalizer::AddDummyPrefix::DEFAULT;
        for(size_t p = 0, q = 0;;) {
          p = tagfinder::findNextTagStart(line, p);
          TagType tagType = TagType::NONE;
          unsigned char tagSpacing = TAGSPACING_NONE;
          size_t r = p;
          std::unique_ptr<std::pair<std::string, std::string>> terminologyConstraint;
          if(p != std::string::npos) {
            std::vector<std::pair<std::string, std::string>> attributes;
            std::string tagName;
            r = tagfinder::findTagEnd(line, p, &attributes, &tagName);
            // HTML comment or XML declaration?
            if((inputFormat == InputFormat::HTML && p + 3 < line.length() && line[p + 1] == '!'
                && line[p + 2] == '-' && line[p + 3] == '-')
               || (p + 1 < line.length() && line[p + 1] == '?')) {
              tagType = TagType::EMPTY_TAG;
            }
            if(r == std::string::npos) {
              // bad HTML
              if(tagType != TagType::EMPTY_TAG) {
                ++p;
                continue;
              }
              // a comment with no close - treat end of text as close
              r = line.length() - 1;
            }
            if(tagType == TagType::NONE) {
              if(line[p + 1] == '/') {
                tagType = TagType::CLOSE_TAG;
              } else {
                if(inputFormat == InputFormat::XLIFF1) {
                  if(checkAndMoveToCloseTag(line, tagName, r, "bpt")
                     || checkAndMoveToCloseTag(line, tagName, r, "bx")) {
                    tagType = TagType::OPEN_TAG;
                  } else if(checkAndMoveToCloseTag(line, tagName, r, "ept")
                            || checkAndMoveToCloseTag(line, tagName, r, "ex")) {
                    tagType = TagType::CLOSE_TAG;
                  } else if(checkAndMoveToCloseTag(line, tagName, r, "ph")
                            || checkAndMoveToCloseTag(line, tagName, r, "x")) {
                    tagType = TagType::EMPTY_TAG;
                    for(const auto& att : attributes) {
                      if(att.first == "ctype") {
                        if(att.second == "lb" || att.second == "pb" || att.second == "x-br"
                           || att.second == "x-break" || att.second == "x-linefeed"
                           || att.second == "x-nbsp") {
                          tagSpacing |= TAGSPACING_WITHIN;
                        }
                        break;
                      }
                    }
                  }
                } else if(inputFormat == InputFormat::HTML) {
                  if(!tagName.compare("img") || !tagName.compare("br") || !tagName.compare("wbr")) {
                    tagType = TagType::EMPTY_TAG;
                  }
                }

                if(tagType == TagType::NONE) {
                  static std::string dictTagName = "mn:dict";
                  tagType = (line[r - 1] != '/') ? TagType::OPEN_TAG : TagType::EMPTY_TAG;
                  if(tagType == TagType::OPEN_TAG) {
                    size_t tagContentStart = r + 1;
                    if(checkAndMoveToCloseTag(line, tagName, r, dictTagName)) {
                      for(const auto& att : attributes) {
                        if(att.first == "translation") {
                          std::string term(line,
                                           tagContentStart,
                                           r - dictTagName.length() - 2 - tagContentStart);
                          terminologyConstraint.reset(new std::pair<std::string, std::string>(
                              " " + term, " " + att.second));
                          break;
                        }
                      }
                    }
                  }
                }
              }
            }
          }
          bool emptyPrefix = true;
          std::string prefix;
          if(q < line.size() && p > q) {
            prefix.assign(line, q, p - q);
            if(prefix.find_first_not_of(" ") != std::string::npos) {
              emptyPrefix = false;
            }
          }
          if(!emptyPrefix || (!prefix.empty() && (words.empty() || p == std::string::npos))) {
            encodeMarkupText(prefix,
                             inputFormat,
                             entitizeTags,
                             tagSpacing,
                             /*afterTag=*/!words.empty(),
                             /*specialSymbols=*/false,
                             addDummyPrefix,
                             words);
          } else if((tagSpacing & TAGSPACING_WITHIN) != 0 || !prefix.empty()) {
            addDummyPrefix = sentencepiece::normalizer::AddDummyPrefix::ON;
          } else if(entitizeTags && !words.empty()) {
            addDummyPrefix = sentencepiece::normalizer::AddDummyPrefix::OFF;
          }

          if(p == std::string::npos)
            break;

          q = r;
          if(terminologyConstraint) {
            if(hasTerminologyConstraints_) {
              if(addDummyPrefix == sentencepiece::normalizer::AddDummyPrefix::ON) {
                words.push_back(spaceToken_);
              }
              words.push_back(termTokenS_);
              sentencepiece::normalizer::AddDummyPrefix adpTemp = addDummyPrefix;
              encodeMarkupText(terminologyConstraint->first,
                               inputFormat,
                               entitizeTags,
                               tagSpacing,
                               /*afterTag=*/false,
                               /*specialSymbols=*/true,
                               addDummyPrefix,
                               words);
              bool needSpace = (adpTemp != sentencepiece::normalizer::AddDummyPrefix::OFF);
              if(needSpace) {
                words.push_back(spaceToken_);
              }
              words.push_back(termTokenOpenC_);
              encodeMarkupText(terminologyConstraint->second,
                               inputFormat,
                               entitizeTags,
                               tagSpacing,
                               /*afterTag=*/false,
                               /*specialSymbols=*/false,
                               adpTemp,
                               words);
              if(needSpace) {
                words.push_back(spaceToken_);
              }
              words.push_back(termTokenCloseC_);
            } else {
              // terminology constraint specified but will be ignored since vocab model has no
              // capability for it
              encodeMarkupText(terminologyConstraint->first,
                               inputFormat,
                               entitizeTags,
                               tagSpacing,
                               /*afterTag=*/ false,
                               /*specialSymbols=*/false,
                               addDummyPrefix,
                               words);
            }
          } else {
            if(p > 0 && std::isspace(static_cast<unsigned char>(line[p - 1]))) {
              tagSpacing |= TAGSPACING_BEFORE;
            }

            if(q + 1 < line.length() && std::isspace(static_cast<unsigned char>(line[q + 1]))) {
              tagSpacing |= TAGSPACING_AFTER;
              if(emptyPrefix && !entitizeTags) {
                for(auto it = words.rbegin(); it != words.rend(); ++it) {
                  auto& markupTag = it->getMarkupTag();
                  if(!markupTag) {
                    break;
                  }
                  markupTag->spacing() |= TAGSPACING_AFTER_IMMEDIATE_FOLLOWING_TAG;
                }
              }
            }

            if(!entitizeTags) {
              std::string tag(line, p, q - p + 1);
              std::string tagIdentifier = markupTagIdentifier(tag, tagType, inputFormat);
              bool joinTagToPrev = false;
              if(emptyPrefix && !words.empty() && words.back().getMarkupTag()) {
                auto& markupTag = words.back().getMarkupTag();
                if(prefix.empty()
                   && (tagType == TagType::EMPTY_TAG || markupTag->type() == TagType::EMPTY_TAG)) {
                  joinTagToPrev = true;
                  if(tagType != TagType::EMPTY_TAG) {
                    markupTag->type() = tagType;
                    markupTag->identifier() = tagIdentifier;
                  }
                } else if(tagType == TagType::CLOSE_TAG && markupTag->type() == TagType::OPEN_TAG
                          && tagIdentifier == markupTag->identifier()) {
                  joinTagToPrev = true;
                  markupTag->type() = TagType::EMPTY_TAG;
                  if(!prefix.empty()) {
                    tagSpacing |= TAGSPACING_WITHIN;
                    if((markupTag->spacing() & TAGSPACING_AFTER) != 0) {
                      markupTag->spacing() &= ~TAGSPACING_AFTER;
                    }
                    if((tagSpacing & TAGSPACING_BEFORE) != 0) {
                      tagSpacing &= ~TAGSPACING_BEFORE;
                    }
                  }
                  if(words.size() >= 2) {
                    auto& mt = std::prev(words.end(), 2)->getMarkupTag();
                    if(mt && (mt->spacing() & TAGSPACING_AFTER) == 0) {
                      if(!prefix.empty()) {
                        for(auto it = std::next(words.rbegin()); it != words.rend(); ++it) {
                          auto& t = it->getMarkupTag();
                          if(!t) {
                            break;
                          }
                          if((t->spacing() & TAGSPACING_AFTER_IMMEDIATE_FOLLOWING_TAG) != 0) {
                            t->spacing() &= ~TAGSPACING_AFTER_IMMEDIATE_FOLLOWING_TAG;
                          }
                        }
                      }
                      mt->tag() += markupTag->tag();
                      words.pop_back();
                    }
                  }
                } else if((markupTag->spacing() & TAGSPACING_BEFORE) != 0
                          || (markupTag->spacing()
                       & TAGSPACING_BEFORE_IMMEDIATE_PRECEDING_TAG)
                          != 0) {
                  tagSpacing |= TAGSPACING_BEFORE_IMMEDIATE_PRECEDING_TAG;
                }
              }
              if(tagType != TagType::OPEN_TAG && tagSpacing == TAGSPACING_NONE
                 && addDummyPrefix != sentencepiece::normalizer::AddDummyPrefix::ON) {
                for(auto it = words.rbegin(); it != words.rend(); ++it) {
                  if(!it->getMarkupTag()) {
                    if(wordEndsWithAlpha(*it)) {
                      addDummyPrefix = sentencepiece::normalizer::AddDummyPrefix::ON;
                    }
                    break;
                  }
                }
              }
              if(!joinTagToPrev) {
                words.push_back(Word::fromWordIndexAndTag(
                    (std::size_t)-1, tag, tagIdentifier, tagType, tagSpacing));
              } else {
                auto& markupTag = words.back().getMarkupTag();
                markupTag->spacing() |= tagSpacing;
                markupTag->tag() += std::move(prefix) + std::move(tag);
              }
              if(tagType == TagType::CLOSE_TAG && !emptyPrefix) {
                auto prevMarkup
                    = std::find_if(std::next(words.rbegin()), words.rend(), [](const Word& word) {
                        return word.getMarkupTag().operator bool();
                      });
                if(prevMarkup != words.rend()
                   && prevMarkup->getMarkupTag()->type() == TagType::OPEN_TAG
                   && !std::prev(prevMarkup)->getMarkupTag()
                   && words.back().getMarkupTag()->identifier()
                          == prevMarkup->getMarkupTag()->identifier()) {
                  words.back().getMarkupTag()->elementContent() = std::move(prefix);
                }
              }
            } else {
              if(!emptyPrefix || words.empty() || !words.back().getMarkupTag()) {
                words.push_back(Word::fromWordIndexAndTag((std::size_t)addDummyPrefix,
                                                          line.substr(p, q - p + 1),
                                                          std::string(),
                                                          tagType,
                                                          tagSpacing));
              } else {
                auto& markupTag = words.back().getMarkupTag();
                markupTag->spacing() |= tagSpacing;
                markupTag->tag() += line.substr(p, q - p + 1);
              }
            }
          }
          p = ++q;
        }

        if(entitizeTags && !words.empty()) {
          size_t startWordInd = 0;
          for(; startWordInd < words.size() && words[startWordInd].getMarkupTag();
              ++startWordInd)
            ;

          size_t lastWordInd = words.size() - 1;
          if(startWordInd > 0) {
            for(; lastWordInd > startWordInd && words[lastWordInd].getMarkupTag();
                --lastWordInd)
              ;

            if(lastWordInd == words.size() - 1) {
              startWordInd = 0;
            }
          }

          if(startWordInd > 0) {
            std::stack<size_t> unbalancedTags;
            for(size_t k = 0; k < words.size(); ++k) {
              const auto& markupTag = words[k].getMarkupTag();
              if(markupTag) {
                if(markupTag->type() == TagType::OPEN_TAG) {
                  unbalancedTags.push(k);
                } else if(markupTag->type() == TagType::CLOSE_TAG) {
                  if(k < lastWordInd && unbalancedTags.top() < startWordInd) {
                    startWordInd = unbalancedTags.top();
                  } else if(k > lastWordInd && unbalancedTags.top() > startWordInd) {
                    lastWordInd = k;
                  }
                  unbalancedTags.pop();
                }
              }
            }
          }

          for(size_t k = 0; k < startWordInd; ++k) {
            if(words[k].getMarkupTag())
              words[k].setWordIndex((WordIndex)-1);
          }

          for(size_t k = lastWordInd + 1; k < words.size(); ++k) {
            if(words[k].getMarkupTag())
              words[k].setWordIndex((WordIndex)-1);
          }

          for(size_t k = startWordInd, entitizedTagId = 1; k <= lastWordInd; ++k) {
            const auto& markupTag = words[k].getMarkupTag();
            if(markupTag) {
              Word newWordTag = Word::fromWordIndexAndTag(entitizedTagId,
                                                          markupTag->tag(),
                                                          std::string(),
                                                          markupTag->type(),
                                                          markupTag->spacing());
              std::ostringstream oss;
              oss << "__ent_" << std::setfill('0') << std::setw(5) << entitizedTagId++ << "_";
              sentencepiece::normalizer::AddDummyPrefix adp
                  = (sentencepiece::normalizer::AddDummyPrefix)words[k].toWordIndex();
              sentencepiece::SentencePieceText spt;
              spm_->Encode(oss.str(), &spt, adp);
              words[k] = std::move(newWordTag);
              if(spt.pieces_size() == 0) {
                assert(false);
              } else {
                const auto& pieces = spt.pieces();
                for(auto sp = pieces.begin(); sp != pieces.end(); ++sp) {
                  words.emplace(words.begin() + ++k, Word::fromWordIndex(sp->id(), sp->surface()));
                  ++lastWordInd;
                }
              }
            }
          }
        }
      }
    } else {
      spm_->SampleEncode(line, -1, alpha_, &spmIds);
    }

    if(!spmIds.empty()) {
      words.reserve(words.size() + spmIds.size() + addEOS);
      for(auto&& spmId : spmIds)
        words.push_back(Word::fromWordIndex(spmId));
    }

    // first non-tag token is just a space? remove it - often such is added with CJK and is not
    // desirable in a non-spacing language for both translation and alignment
    for(auto it = words.begin(); it != words.end(); ++it) {
      if(!it->getMarkupTag()) {
        if(isWordSpaceSymbol(*it)) {
          it->setIsSpaceSymbol(true);
        }
      }
    }

    if(addEOS)
      words.push_back(getEosId());
#if 0
    for(const auto& word : words) {
      if(word.getMarkupTag()) {
        LOG(info,
            "Tag {}, type {}, id {}, spacing {}, elt {}",
            word.getMarkupTag()->tag(),
            static_cast<size_t>(word.getMarkupTag()->type()),
            word.getMarkupTag()->identifier(),
            static_cast<size_t>(word.getMarkupTag()->spacing()),
            word.getMarkupTag()->elementContent());
      } else {
        LOG(info, "Word id {}, word {}", word.toWordIndex(), (*this)[word]);
      }
    }
#endif
    return words;
  }

  std::string decode(const Words& sentenceIn,
                     bool ignoreEOS,
                     InputFormat inputFormat,
                     bool entitizeTags) const override {
    std::string line;
    if(keepEncoded_) {  // i.e. keep the sentence segmented into subword units
      for(const Word& id : sentenceIn)
        if(!ignoreEOS || id != getEosId())
          line += (!id.getMarkupTag() ? (*this)[id] : id.getMarkupTag()->tag()) + " ";
      line.pop_back();  // trim the trailing whitespace
    } else {
      // convert vector of Word to vector of int
      if(inputFormat == InputFormat::PLAINTEXT) {
        std::vector<int> spmSentence;
        spmSentence.reserve(sentenceIn.size());
        for(size_t i = 0; i < sentenceIn.size(); ++i) {
          const auto& word = sentenceIn[i];
          if(!ignoreEOS || word != getEosId()) {
            WordIndex wordIndex = word.toWordIndex();
            spmSentence.push_back(wordIndex);
          }
        }
        spm_->Decode(spmSentence, &line);
      } else {
        // first make sure tags do not occur within UTF-8 byte sequences, which would result in
        // breaking them up and yielding invalid UTF-8 (this can happen only if SentencePiece option
        // "byte_fallback" is set to true)
        Words sentence;
        sentence.reserve(sentenceIn.size());
        Words byteSequence;
        for(size_t i = 0; i < sentenceIn.size(); ++i) {
          const auto& word = sentenceIn[i];
          bool wordIsByte = false;
          if(!word.getMarkupTag()) {
            const std::string& curWord = (*this)[word];
            if(curWord.length() == 6 && curWord.compare(0, 3, "<0x") == 0 && curWord.back() == '>'
               && std::isxdigit(static_cast<unsigned char>(curWord[3]))
               && std::isxdigit(static_cast<unsigned char>(curWord[4]))) {
              byteSequence.push_back(word);
              wordIsByte = true;
            } else if(!byteSequence.empty()) {
                std::move(byteSequence.begin(), byteSequence.end(), std::back_inserter(sentence));
                byteSequence.clear();
            }
          }
          if(!wordIsByte) {
            sentence.push_back(word);
          }
        }
        if(!byteSequence.empty()) {
          std::move(byteSequence.begin(), byteSequence.end(), std::back_inserter(sentence));
          byteSequence.clear();
        }

        std::vector<size_t> entitizedTagIndexes;
        std::vector<std::string> spacePrefix;
        std::vector<std::string> spPieces;
        spacePrefix.reserve(sentence.size());
        spPieces.reserve(sentence.size());
        bool keepTagsOutOfWords = false;
        bool firstWordMet = false;
        bool usingSurfaces = false;
        for(size_t i = 0; i < sentence.size(); ++i) {
          spacePrefix.emplace_back();
          const auto& word = sentence[i];
          if(!word.getMarkupTag()) {
            const std::string& curWord = (*this)[word];
            if(!firstWordMet && word.getSurface() && !usingSurfaces) {
              usingSurfaces = true;
            }
            // from, in SP: const absl::string_view kSpaceSymbol = "\xe2\x96\x81";
            if(curWord.length() >= 3 && curWord[0] == (char)0xe2 && curWord[1] == (char)0x96
               && curWord[2] == (char)0x81) {
              if(!word.getSurface()) {
                spacePrefix.back() = " ";
              } else {
                std::string surface = *(word.getSurface());
                std::string curWordNoPrefix(curWord, 3);
                if(surface.length() > curWordNoPrefix.length()
                   && surface.compare(surface.length() - curWordNoPrefix.length(),
                                      curWordNoPrefix.length(),
                                      curWordNoPrefix)
                          == 0) {
                  spacePrefix.back()
                      = surface.substr(0, surface.length() - curWordNoPrefix.length());
                } else if(!surface.empty()
                          && std::isspace(static_cast<unsigned char>(surface[0]))) {
                  spacePrefix.back() = surface[0];
                }
              }
            }
            if(!keepTagsOutOfWords && word != getEosId()
               && ((firstWordMet && !spacePrefix.back().empty()) || wordStartsWithAlpha(word)
                   || wordEndsWithAlpha(word))) {
              keepTagsOutOfWords = true;
            }
            if(!firstWordMet) {
              firstWordMet = true;
            }
          }
        }

        bool lineHasTrailingSpace = true;
        Ptr<MarkupTag> prevMarkupTag;
        for(size_t i = 0; i < sentence.size();) {
          const auto& word = sentence[i];
          if(!word.getMarkupTag()) {
            if(!word.getSurface()) {
              spPieces.emplace_back((*this)[word]);
            } else {
              std::string surface = *(word.getSurface());
              // if first word following tag, remove the space prefix from the surface since it has
              // been added separately
              if(i > 0 && spPieces.empty() && !surface.empty() && !spacePrefix[i].empty()
                 && surface.compare(0, spacePrefix[i].length(), spacePrefix[i]) == 0) {
                surface.erase(0, spacePrefix[i].length());
              }
              spPieces.push_back(surface);
            }
            ++i;
          } else if(word.toWordIndex() == (WordIndex)-1) {
            // collect all adjacent tags; find next real word
            TagType tagType = word.getMarkupTag()->type();
            unsigned char tagSpacing = word.getMarkupTag()->spacing();
            size_t j = i + 1;
            for(; j < sentence.size() && sentence[j].getMarkupTag()
                  && sentence[j].toWordIndex() == (WordIndex)-1;
                ++j) {
              // need to track whether the tags are all of the same type: open or close, empty tags
              // being neutral
              if(tagType == TagType::EMPTY_TAG) {
                tagType = sentence[j].getMarkupTag()->type();
              } else if(sentence[j].getMarkupTag()->type() != tagType) {
                tagType = TagType::NONE;
              }
              tagSpacing |= sentence[j].getMarkupTag()->spacing();
            }

            bool tagSpacingRequired
                = ((tagSpacing & TAGSPACING_BEFORE) != 0 || (tagSpacing & TAGSPACING_AFTER) != 0
                   || (tagSpacing & TAGSPACING_WITHIN) != 0);
            bool specialTagHandling = false;
            std::string spaceRequiredBeforeNextWord, leftPart, rightPart, middlePart;
            std::unique_ptr<std::string> originalLeftPartBeforeClose;
            if(i > 0 && j < spacePrefix.size() && !sentence[j].getMarkupTag()
               && sentence[j] != getEosId()) {
              if(!spacePrefix[j].empty()) {
                spaceRequiredBeforeNextWord = spacePrefix[j];
              } else if(keepTagsOutOfWords && j + 1 < sentence.size()) {
                // prevent the tags from appearing in the middle of the word
                // sentence has spaces, and the adjacent tags are all open or close (possibly with
                // self-closing mixed in)
                // if open, move left, if close, move right to where there is a space
                // deal with everything here:
                if(word.getMarkupTag()->type() != TagType::CLOSE_TAG) {
                  // it is possible that the action of moving an open tag out to the beginning of
                  // a word here may be reversed later (when the closing tag is encountered and
                  // the source element content is found to be the same as the original proposed
                  // target element content)
                  if(wordStartsWithAlpha(sentence[j])) {
                    specialTagHandling = true;
                    size_t previousWordsEndIdx = spPieces.size();
                    for(size_t k = 0; k < spPieces.size(); ++k) {
                      if(!wordEndsWithAlpha(sentence[i - k - 1])) {
                        // tag comes after this word
                        previousWordsEndIdx = spPieces.size() - k;
                        break;
                      }
                      if(!spacePrefix[i - k - 1].empty()) {
                        previousWordsEndIdx = spPieces.size() - k - 1;
                        break;
                      }
                    }

                    std::string spaceRequired;
                    std::vector<std::string> spPieces2;
                    if(!spPieces.empty()) {
                      if(previousWordsEndIdx < spPieces.size()) {
                        size_t idx = i - spPieces.size() + previousWordsEndIdx;
                        if(idx > 0 && previousWordsEndIdx > 0 && !spacePrefix[idx].empty()) {
                          spaceRequired = spacePrefix[idx];
                          // remove the space prefix from the surface
                          // since it gets added separately
                          if(!spPieces[previousWordsEndIdx].empty()
                             && spPieces[previousWordsEndIdx].compare(
                                    0, spacePrefix[idx].length(), spacePrefix[idx])
                                    == 0) {
                            spPieces[previousWordsEndIdx].erase(0, spacePrefix[idx].length());
                          }
                        }
                        auto it = std::next(spPieces.begin(), previousWordsEndIdx);
                        std::move(it, spPieces.end(), std::back_inserter(spPieces2));
                        spPieces.erase(it, spPieces.end());
                      }
                      if(!spPieces.empty()) {
                        std::string detokenized;
                        spm_->Decode(spPieces, &detokenized);
                        spPieces.clear();
                        leftPart = encodeSpecialChars(detokenized);
                        lineHasTrailingSpace = false;
                      }
                    }

                    std::string spaceNeededBeforeOpenTag;
                    if(!spaceRequired.empty() && !tagSpacingRequired) {
                      spaceNeededBeforeOpenTag = spaceRequired;
                    }

                    bool emptyLine = line.empty() && leftPart.empty();
                    bool spaceAdded = false;
                    size_t m = i;
                    do {
                      const auto& markupTag = sentence[m].getMarkupTag();
                      if(!lineHasTrailingSpace) {
                        if(!spaceNeededBeforeOpenTag.empty()
                           && markupTag->type() != TagType::CLOSE_TAG) {
                          middlePart += spaceNeededBeforeOpenTag;
                          spaceNeededBeforeOpenTag.clear();
                          if(usingSurfaces) {
                            spaceRequired.clear();
                          }
                          spaceAdded = true;
                        } else if(!spaceRequired.empty()
                                  && (markupTag->spacing() & TAGSPACING_BEFORE) != 0) {
                          middlePart += spaceRequired;
                          if(usingSurfaces) {
                            spaceRequired.clear();
                            spaceNeededBeforeOpenTag.clear();
                          }
                          spaceAdded = true;
                        }
                      }
                      middlePart += markupTag->tag();
                      if((!spaceRequired.empty() || (!usingSurfaces && emptyLine))
                         && (markupTag->spacing() & TAGSPACING_AFTER) != 0) {
                        middlePart += !spaceRequired.empty() ? spaceRequired : " ";
                        if(usingSurfaces) {
                          spaceRequired.clear();
                          spaceNeededBeforeOpenTag.clear();
                        }
                        spaceAdded = true;
                        lineHasTrailingSpace = true;
                      } else {
                        lineHasTrailingSpace = false;
                      }
                    } while(++m < j);

                    if(!spPieces2.empty()) {
                      if(!spaceRequired.empty() && !spaceAdded) {
                        rightPart = spaceRequired;
                      }
                      std::string detokenized;
                      spm_->Decode(spPieces2, &detokenized);
                      rightPart += encodeSpecialChars(detokenized);
                      lineHasTrailingSpace = false;
                    }
                  }
                } else {
                  // closing tag(s), move right
                  // it is possible that the action of moving an closing tag out to the end of
                  // a word here may be reversed later (when the closing tag is encountered and
                  // the source element content is found to be the same as the original proposed
                  // target element content)
                  if(!spPieces.empty() && wordEndsWithAlpha(sentence[i - 1])) {
                    specialTagHandling = true;
                    size_t k = j;
                    for(; k < spacePrefix.size() && spacePrefix[k].empty()
                          && sentence[k] != getEosId();
                        ++k) {
                      if(sentence[k].getMarkupTag()) {
                        if(sentence[k].toWordIndex() == (WordIndex)-1
                           && sentence[k].getMarkupTag()->type() != TagType::CLOSE_TAG) {
                          // the next real word must require a leading space or eos or non-alpha
                          size_t l = k + 1;
                          for(; l < spacePrefix.size() && sentence[l].getMarkupTag(); ++l) {
                          }

                          if(l < spacePrefix.size() && spacePrefix[l].empty()
                             && sentence[l] != getEosId()) {
                            if(!wordStartsWithAlpha(sentence[l])) {
                              break;
                            }
                            // no change to the position of the tag to place
                            k = j;
                          }
                          break;
                        }
                      } else {
                        if(!wordStartsWithAlpha(sentence[k])) {
                          break;
                        }
                      }
                    }

                    if(j < k) {
                      if(!spPieces.empty()) {
                        std::string detokenized;
                        spm_->Decode(spPieces, &detokenized);
                        originalLeftPartBeforeClose.reset(
                            new std::string(encodeSpecialChars(detokenized)));
                      }

                      size_t l = j;
                      do {
                        if(!sentence[l].getMarkupTag()) {
                          if(!sentence[l].getSurface())
                            spPieces.emplace_back((*this)[sentence[l]]);
                          else
                            spPieces.emplace_back(*(sentence[l].getSurface()));
                        }
                      } while(++l < k);
                    }

                    if(!spPieces.empty()) {
                      std::string detokenized;
                      spm_->Decode(spPieces, &detokenized);
                      spPieces.clear();
                      leftPart = encodeSpecialChars(detokenized);
                      lineHasTrailingSpace = false;
                    }

                    j = k;
                    std::string spaceRequired;
                    if(j < spacePrefix.size() && !spacePrefix[j].empty())
                      spaceRequired = spacePrefix[j];
                    bool spaceAdded = false;
                    size_t m = i;
                    do {
                      const auto& markupTag = sentence[m].getMarkupTag();
                      if(markupTag && sentence[m].toWordIndex() == (WordIndex)-1) {
                        if(!spaceRequired.empty() && !lineHasTrailingSpace
                           && (markupTag->spacing() & TAGSPACING_BEFORE) != 0) {
                          middlePart += spaceRequired;
                          if(usingSurfaces) {
                            spaceRequired.clear();
                          }
                          spaceAdded = true;
                        }

                        middlePart += markupTag->tag();
                        if(!spaceRequired.empty()
                           && (markupTag->spacing() & TAGSPACING_AFTER) != 0) {
                          middlePart += spaceRequired;
                          if(usingSurfaces) {
                            spaceRequired.clear();
                          }
                          spaceAdded = true;
                          lineHasTrailingSpace = true;
                        } else {
                          lineHasTrailingSpace = false;
                        }
                      }
                    } while(++m < j);

                    if(!spaceRequired.empty() && !spaceAdded) {
                      rightPart = spaceRequired;
                      lineHasTrailingSpace = true;
                    }
                  }
                }
              }
            }

            if(!specialTagHandling) {
              if(!spPieces.empty()) {
                std::string detokenized;
                spm_->Decode(spPieces, &detokenized);
                spPieces.clear();
                leftPart = encodeSpecialChars(detokenized);
                lineHasTrailingSpace = false;
              }

              bool emptyLine = line.empty() && leftPart.empty();
              std::string spaceNeededBeforeOpenTag;
              if(!spaceRequiredBeforeNextWord.empty() && tagType != TagType::CLOSE_TAG
                 && !tagSpacingRequired) {
                spaceNeededBeforeOpenTag = spaceRequiredBeforeNextWord;
              }
              bool spaceAdded = false;
              size_t k = i;
              do {
                const auto& markupTag = sentence[k].getMarkupTag();
                if(!lineHasTrailingSpace) {
                  if(!spaceNeededBeforeOpenTag.empty() && markupTag->type() != TagType::CLOSE_TAG) {
                    middlePart += spaceNeededBeforeOpenTag;
                    spaceNeededBeforeOpenTag.clear();
                    if(usingSurfaces) {
                      spaceRequiredBeforeNextWord.clear();
                    }
                    spaceAdded = true;
                  } else if((!spaceRequiredBeforeNextWord.empty()
                             || (!usingSurfaces && j + 1 >= sentence.size()))
                            && (markupTag->spacing() & TAGSPACING_BEFORE) != 0) {
                    middlePart
                        += !spaceRequiredBeforeNextWord.empty() ? spaceRequiredBeforeNextWord : " ";
                    if(usingSurfaces) {
                      spaceRequiredBeforeNextWord.clear();
                      spaceNeededBeforeOpenTag.clear();
                    }
                    spaceAdded = true;
                  }
                }
                middlePart += markupTag->tag();
                if((!spaceRequiredBeforeNextWord.empty()
                    || (!usingSurfaces && (j + 1 >= sentence.size() || emptyLine)))
                   && (markupTag->spacing() & TAGSPACING_AFTER) != 0) {
                  middlePart
                      += !spaceRequiredBeforeNextWord.empty() ? spaceRequiredBeforeNextWord : " ";
                  if(usingSurfaces) {
                    spaceRequiredBeforeNextWord.clear();
                    spaceNeededBeforeOpenTag.clear();
                  }
                  spaceAdded = true;
                  lineHasTrailingSpace = true;
                } else {
                  lineHasTrailingSpace = false;
                }
              } while(++k < j);
              if(!spaceRequiredBeforeNextWord.empty() && !spaceAdded
                 && (usingSurfaces || (tagSpacing & TAGSPACING_WITHIN) == 0)) {
                rightPart = spaceRequiredBeforeNextWord;
                lineHasTrailingSpace = true;
              }
            }
            bool closeTagMoved = false;
            if(word.getMarkupTag()->type() == TagType::CLOSE_TAG
               && !word.getMarkupTag()->elementContent().empty() && prevMarkupTag
               && prevMarkupTag->identifier() == word.getMarkupTag()->identifier()
               && (originalLeftPartBeforeClose
                       ? (word.getMarkupTag()->elementContent() == *originalLeftPartBeforeClose)
                       : (word.getMarkupTag()->elementContent() == leftPart))) {
              size_t prevTagPos = line.rfind(prevMarkupTag->tag());
              // previous tag is expected to always be found
              if(prevTagPos != std::string::npos
                 && prevTagPos + prevMarkupTag->tag().length() < line.length()) {
                line.erase(prevTagPos, prevMarkupTag->tag().length());
                line += prevMarkupTag->tag();
              }
              if(originalLeftPartBeforeClose
                 && originalLeftPartBeforeClose->length() < leftPart.length()) {
                // note: originalLeftPartBeforeClose is a prefix of leftPart
                line += leftPart.substr(0, word.getMarkupTag()->elementContent().length());
                line += std::move(middlePart);
                line += leftPart.substr(word.getMarkupTag()->elementContent().length());
                closeTagMoved = true;
              }
            }
            if(!closeTagMoved) {
              line += std::move(leftPart) + std::move(middlePart);
            }
            line += std::move(rightPart);
            i = j;
            if(sentence[i - 1].getMarkupTag()
               && sentence[i - 1].getMarkupTag()->type() == TagType::OPEN_TAG) {
              prevMarkupTag = sentence[i - 1].getMarkupTag();
            } else {
              prevMarkupTag.reset();
            }
          } else {
            entitizedTagIndexes.push_back(i);
            ++i;
          }
        }

        if(!spPieces.empty()) {
          std::string detokenized;
          spm_->Decode(spPieces, &detokenized);
          line += encodeSpecialChars(detokenized);
        }

        if(entitizeTags && !entitizedTagIndexes.empty()) {
          for(auto i : entitizedTagIndexes) {
            WordIndex entitizedTagId = sentence[i].toWordIndex();
            std::ostringstream oss;
            oss << "__ent_" << std::setfill('0') << std::setw(5) << entitizedTagId << "_";
            std::string entity = oss.str();
            size_t p = line.find(entity);
            if(p != std::string::npos) {
              line.replace(
                  p, entity.length(), sentence[i].getMarkupTag()->tag());
            } else {
              line += sentence[i].getMarkupTag()->tag();
            }
          }

          // remove any stray tag entities
          line = regex::regex_replace(line, regex::regex("__ent_[0-9]+_"), "");
        }
      }
    }
    return line;
  }

  bool sentenceStartsWithSpaceSymbolWord(const Words& sentence) const override {
    for(auto it = sentence.begin(); it != sentence.end(); ++it) {
      if(!it->getMarkupTag()) {
        if(isWordSpaceSymbol(*it)) {
          return true;
        }
        break;
      }
    }
    return false;
  }

  std::string surfaceForm(const Words& sentence) const override {
    // with SentencePiece, decoded form and surface form are identical
    return decode(sentence,
                  /*ignoreEOS=*/true,
                  /*inputFormat=*/InputFormat::PLAINTEXT,
                  /*entitizeTags=*/false);
  }

  size_t size() const override {
    return spm_->GetPieceSize();
  }

  size_t load(const std::string& vocabPath, size_t /*maxSize*/) override {
    LOG(info, "[data] Loading SentencePiece vocabulary from file {}", vocabPath);

    ABORT_IF(!filesystem::exists(vocabPath),
             "SentencePiece vocabulary file {} does not exist",
             vocabPath);

    spm_.reset(new sentencepiece::SentencePieceProcessor());
    const auto status = spm_->Load(vocabPath);

    ABORT_IF(!status.ok(),
             "SentencePiece vocabulary error: {}",
             status.ToString());

    termTokenS_ = encodeSpecialSymbol("<S>");
    termTokenOpenC_ = encodeSpecialSymbol("<C>");
    termTokenCloseC_ = encodeSpecialSymbol("</C>");
    hasTerminologyConstraints_ = (termTokenS_.toWordIndex() != (WordIndex)-1
                                  && termTokenOpenC_.toWordIndex() != (WordIndex)-1
                                  && termTokenCloseC_.toWordIndex() != (WordIndex)-1);
    // from, in SP: const absl::string_view kSpaceSymbol = "\xe2\x96\x81";
    const char spaceSymbol[] = {(char)0xe2, (char)0x96, (char)0x81, 0x00};
    spaceToken_ = (*this)[(std::string(spaceSymbol))];
    spaceToken_.setIsSpecialSymbol(true);
    populateControlChars();
    return spm_->GetPieceSize();
  }

  std::string toUpper(const std::string& line) const override { return utils::utf8ToUpper(line); }
  std::string toEnglishTitleCase(const std::string& line) const override { return utils::toEnglishTitleCase(line); }

  // SentencePiece with byte-fallback may generate control symbols with output sampling.
  // Let's mark them as special and suppress them later on output. This is generally safe
  // for UTF-8 since control chars are not used as partial bytes in multi-byte sequences.
  // They only appear in single-byte chars as themselves and this is what we suppress.
  void addSpecialWords(std::vector<Word>& special) const override {
    special.reserve(special.size() + controlChars_.size());
    for(auto c : controlChars_)
      special.push_back(c);
  }

};
#endif // USE_SENTENCEPIECE

Ptr<IVocab> createSentencePieceVocab(const std::string& vocabPath, Ptr<Options> options, size_t batchIndex) {
  bool isSentencePiece = regex::regex_search(vocabPath, regex::regex("\\.(spm)$"));
  if(isSentencePiece) {
#ifdef USE_SENTENCEPIECE
    return New<SentencePieceVocab>(options, batchIndex);
#else
    batchIndex; options;
    ABORT("*.spm suffix in path {} reserved for SentencePiece models, "
          "but support for SentencePiece is not compiled into Marian. "
          "Try to recompile after `cmake .. -DUSE_SENTENCEPIECE=on [...]`",
          vocabPath);
#endif
  }
  // Not a SentencePiece model based on suffix;
  return nullptr;
}

}
