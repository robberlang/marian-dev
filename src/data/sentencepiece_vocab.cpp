#include "data/vocab_base.h"
#include "data/tag_finder.h"

#ifdef USE_SENTENCEPIECE
#include "sentencepiece/src/sentencepiece_processor.h"
#include "sentencepiece/src/sentencepiece_trainer.h"
#endif

#include "common/config.h"
#include "common/options.h"
#include "common/logging.h"
#include "common/filesystem.h"
#include "common/regex.h"

#include <sstream>
#include <random>
#include <stack>

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
    return Word::fromWordIndex(spm_->PieceToId(token));
  }

  const std::string& operator[](Word id) const override {
    ABORT_IF(id.toWordIndex() >= size(), "Unknown word id: ", id.toWordIndex());
    return spm_->IdToPiece(id.toWordIndex());
  }

  static bool checkAndMoveToCloseXliffTag(const std::string& line,
                                   size_t tagNameStart,
                                   size_t& tagEnd,
                                   size_t tagNameLength,
                                   const std::string& xliffTagName) {
    if(!line.compare(tagNameStart, tagNameLength, xliffTagName)) {
      if(line[tagEnd - 1] != '/') {
        const std::string tagClose = "</" + xliffTagName + ">";
        size_t endTagEnd = line.find(tagClose, tagEnd + 1);
        if(endTagEnd != std::string::npos) {
          tagEnd = endTagEnd + tagClose.length() - 1;
        }
      }
      return true;
    }
    return false;
  }

  static std::string htmlEncode(const std::string& text) {
    std::string encoded = regex::regex_replace(text, regex::regex("&"), "&amp;");
    encoded = regex::regex_replace(encoded, regex::regex("<"), "&lt;");
    encoded = regex::regex_replace(encoded, regex::regex(">"), "&gt;");
    return encoded;
  }

  static std::string htmlDecode(const std::string& text) {
    std::string decoded = regex::regex_replace(text, regex::regex("&lt;"), "<");
    decoded = regex::regex_replace(decoded, regex::regex("&gt;"), ">");
    decoded = regex::regex_replace(decoded, regex::regex("&amp;"), "&");
    return decoded;
  }

  Words encode(const std::string& line, bool addEOS, bool inference) const override {
    InputFormat inputFormat = ConvertInputFormat(options_->get<std::string>("input-format", ""));
    Words words;
    std::vector<int> spmIds;
    if(inference || alpha_ == 0) {
      if(!inference || inputFormat == InputFormat::PLAINTEXT) {
        spm_->Encode(line, &spmIds);
      } else {
        bool entitizeTags = options_->get<bool>("entitize-tags", false);
        sentencepiece::normalizer::AddDummyPrefix addDummyPrefix
            = sentencepiece::normalizer::AddDummyPrefix::DEFAULT;
        for(size_t q = (size_t)-1, p = 0;;) {
          p = tagfinder::findNextTagStart(line, ++q);
          TagType tagType = TagType::NONE;
          size_t r = p;
          if(p != std::string::npos) {
            r = tagfinder::findTagEnd(line, p);
            if(r == std::string::npos) {
              q = p;
              continue;
            }
            if(line[p + 1] == '/') {
              tagType = TagType::CLOSE_TAG;
            } else {
              size_t tagNameStart = p + 1;
              size_t t = line.find_first_of(" \t\r\n/>", tagNameStart);
              size_t tagNameLength = t - tagNameStart;
              if(inputFormat == InputFormat::XLIFF1) {
                if(checkAndMoveToCloseXliffTag(line, tagNameStart, r, tagNameLength, "bpt")
                   || checkAndMoveToCloseXliffTag(line, tagNameStart, r, tagNameLength, "bx")) {
                  tagType = TagType::OPEN_TAG;
                } else if(checkAndMoveToCloseXliffTag(line, tagNameStart, r, tagNameLength, "ept")
                          || checkAndMoveToCloseXliffTag(
                              line, tagNameStart, r, tagNameLength, "ex")) {
                  tagType = TagType::CLOSE_TAG;
                } else if(checkAndMoveToCloseXliffTag(line, tagNameStart, r, tagNameLength, "ph")
                          || checkAndMoveToCloseXliffTag(
                              line, tagNameStart, r, tagNameLength, "x")) {
                  tagType = TagType::EMPTY_TAG;
                }
              } else if(inputFormat == InputFormat::HTML) {
                if(!line.compare(tagNameStart, tagNameLength, "img")
                   || !line.compare(tagNameStart, tagNameLength, "br")
                   || !line.compare(tagNameStart, tagNameLength, "wbr")) {
                  tagType = TagType::EMPTY_TAG;
                }
              }
            }

            if(tagType == TagType::NONE) {
              tagType = (line[r - 1] != '/') ? TagType::OPEN_TAG : TagType::EMPTY_TAG;
            }
          }
          std::string prefix(line, q, p - q);
          q = r;
          if(!prefix.empty()) {
            if(!entitizeTags) {
              if(addDummyPrefix == sentencepiece::normalizer::AddDummyPrefix::OFF
                 && prefix.front() == ' ') {
                addDummyPrefix = sentencepiece::normalizer::AddDummyPrefix::ON;
              }
            } else {
              if(addDummyPrefix != sentencepiece::normalizer::AddDummyPrefix::DEFAULT) {
                addDummyPrefix = (prefix.front() != ' ')
                                     ? sentencepiece::normalizer::AddDummyPrefix::OFF
                                     : sentencepiece::normalizer::AddDummyPrefix::ON;
              }
            }
            prefix = htmlDecode(prefix);
            spm_->Encode(prefix, &spmIds, addDummyPrefix);
            for(auto&& spmId : spmIds)
              words.emplace_back(Word::fromWordIndex(spmId));

            spmIds.clear();
            addDummyPrefix = (prefix.back() != ' ')
                                 ? sentencepiece::normalizer::AddDummyPrefix::OFF
                                 : sentencepiece::normalizer::AddDummyPrefix::ON;
          } else if(entitizeTags && !words.empty()) {
            addDummyPrefix = sentencepiece::normalizer::AddDummyPrefix::OFF;
          }

          if(p == std::string::npos)
            break;

          if(!entitizeTags) {
            words.emplace_back(
                Word::fromWordIndexAndTag((WordIndex)-1, line.substr(p, q - p + 1), tagType));
          } else {
            words.emplace_back(
                Word::fromWordIndexAndTag(addDummyPrefix, line.substr(p, q - p + 1), tagType));
          }
        }

        if(entitizeTags && !words.empty()) {
          size_t startWordInd = 0;
          for(; startWordInd < words.size() && words[startWordInd].getMarkupTag(); ++startWordInd)
            ;

          size_t lastWordInd = words.size() - 1;
          if(startWordInd > 0) {
            for(; lastWordInd > startWordInd && words[lastWordInd].getMarkupTag(); --lastWordInd)
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
                if(markupTag->getType() == TagType::OPEN_TAG) {
                  unbalancedTags.push(k);
                } else if(markupTag->getType() == TagType::CLOSE_TAG) {
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

          for(size_t k = startWordInd, entitizedTagId = 0; k <= lastWordInd; ++k) {
            const auto& markupTag = words[k].getMarkupTag();
            if(markupTag) {
              Word newWordTag = Word::fromWordIndexAndTag(
                  entitizedTagId, markupTag->getTag(), markupTag->getType());
              std::ostringstream oss;
              oss << "__ent_" << std::setfill('0') << std::setw(5) << entitizedTagId++ << "_";
              sentencepiece::normalizer::AddDummyPrefix adp
                  = (sentencepiece::normalizer::AddDummyPrefix)words[k].toWordIndex();
              spm_->Encode(oss.str(), &spmIds, adp);
              if(spmIds.empty()) {
                words.erase(words.begin() + k--);
                --lastWordInd;
              } else {
                words[k] = Word::fromWordIndex(spmIds[0]);
                for(auto s = spmIds.begin() + 1; s != spmIds.end(); ++s) {
                  words.emplace(words.begin() + ++k, Word::fromWordIndex(*s));
                  ++lastWordInd;
                }
              }
              spmIds.clear();
              words.emplace_back(std::move(newWordTag));
            }
          }
        }
      }
    } else {
      spm_->SampleEncode(line, -1, alpha_, &spmIds);
    }

    if(words.empty()) {
      words.reserve(spmIds.size() + addEOS);
      for(auto&& spmId : spmIds)
        words.emplace_back(Word::fromWordIndex(spmId));
    }

    if(addEOS)
      words.emplace_back(getEosId());
    return words;
  }

  std::string decode(const Words& sentence, bool /*ignoreEOS*/) const override {
    InputFormat inputFormat = ConvertInputFormat(options_->get<std::string>("input-format", ""));
    std::string line;
    if(keepEncoded_) {  // i.e. keep the sentence segmented into subword units
      for(const Word& id : sentence)
        line += (!id.getMarkupTag() ? (*this)[id] : id.getMarkupTag()->getTag()) + " ";
      line.pop_back();  // trim the trailing whitespace
    } else {
      // convert vector of Word to vector of int
      std::vector<int> spmSentence;
      spmSentence.reserve(sentence.size());
      if(inputFormat == InputFormat::PLAINTEXT) {
        for(const Word& word : sentence)
          spmSentence.push_back(word.toWordIndex());
        spm_->Decode(spmSentence, &line);
      } else {
        bool entitizeTags = options_->get<bool>("entitize-tags", false);
        std::vector<size_t> entitizedTagIndexes;
        for(size_t i = 0; i < sentence.size();) {
          const auto& word = sentence[i];
          WordIndex wordIndex = word.toWordIndex();
          if(!word.getMarkupTag()) {
            spmSentence.push_back(wordIndex);
            ++i;
          } else if(wordIndex == (WordIndex)-1) {
            if(!spmSentence.empty()) {
              std::string detokenized;
              spm_->Decode(spmSentence, &detokenized);
              spmSentence.clear();
              line += htmlEncode(detokenized);
            }

            const auto& markupTag = word.getMarkupTag();
            std::string tag = markupTag->getTag();
            bool needSpace = false;
            size_t j = i + 1;
            for(; j < sentence.size() && sentence[j].getMarkupTag()
                  && sentence[j].toWordIndex() == (WordIndex)-1;
                ++j) {
              tag += sentence[j].getMarkupTag()->getTag();
            }

            if(i > 0 && j < sentence.size() && !sentence[j].getMarkupTag()) {
              const auto& nextWord = (*this)[sentence[j]];
              if(nextWord.size() >= 3 && nextWord[0] == (char)0xe2 && nextWord[1] == (char)0x96
                 && nextWord[2] == (char)0x81) {
                needSpace = true;
              }
            }

            if(needSpace && markupTag->getType() != TagType::CLOSE_TAG)
              line += ' ';

            line += tag;
            if(needSpace && markupTag->getType() == TagType::CLOSE_TAG)
              line += ' ';

            i = j;
          } else {
            entitizedTagIndexes.push_back(i);
            ++i;
          }
        }

        if(!spmSentence.empty()) {
          std::string detokenized;
          spm_->Decode(spmSentence, &detokenized);
          line += htmlEncode(detokenized);
        }

        if(entitizeTags && !entitizedTagIndexes.empty()) {
          for(auto entitizedTagIndex : entitizedTagIndexes) {
            WordIndex entitizedTagId = sentence[entitizedTagIndex].toWordIndex();
            std::ostringstream oss;
            oss << "__ent_" << std::setfill('0') << std::setw(5) << entitizedTagId << "_";
            std::string entity = oss.str();
            size_t p = line.find(entity);
            if(p != std::string::npos) {
              line.replace(
                  p, entity.length(), sentence[entitizedTagIndex].getMarkupTag()->getTag());
            } else {
              line += sentence[entitizedTagIndex].getMarkupTag()->getTag();
            }
          }

          // remove any stray tag entities
          line = regex::regex_replace(line, regex::regex("__ent_[0-9]+_"), "");
        }
      }
    }
    return line;
  }

  std::string surfaceForm(const Words& sentence) const override {
    // with SentencePiece, decoded form and surface form are identical
    return decode(sentence, /*ignoreEOS=*/true);
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

    return spm_->GetPieceSize();
  }

  std::string toUpper(const std::string& line) const override { return utils::utf8ToUpper(line); }
  std::string toEnglishTitleCase(const std::string& line) const override { return utils::toEnglishTitleCase(line); }
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
