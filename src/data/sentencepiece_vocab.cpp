#include "data/vocab_base.h"

#ifdef USE_SENTENCEPIECE
#include "sentencepiece/src/sentencepiece_processor.h"
#include "sentencepiece/src/sentencepiece_trainer.h"
#endif

#include "common/config.h"
#include "common/options.h"
#include "common/logging.h"
#include "common/filesystem.h"
#include "common/regex.h"
#include "common/tag_finder.h"
#include "common/utils.h"
#include "common/char_entities.h"

#include <sstream>
#include <random>
#include <stack>
#include <tuple>
#include <functional>
#include <cctype>

namespace marian {

#ifdef USE_SENTENCEPIECE

typedef std::tuple<const char*, regex::regex, std::function<bool(const std::string&)>>
    PlaceHolderProps;

static const PlaceHolderProps placeHolderList[] = {
    // URLs (from https://gist.github.com/dperini/729294
    // see also https://mathiasbynens.be/demo/url-regex)
    {"__ent_url_",
     regex::regex(
         R"((?:(?:(?:https?|ftp):)?\/\/)(?:\S+(?::\S*)?@)?(?:(?!(?:10|127)(?:\.\d{1,3}){3})(?!(?:169\.254|192\.168)(?:\.\d{1,3}){2})(?!172\.(?:1[6-9]|2\d|3[0-1])(?:\.\d{1,3}){2})(?:[1-9]\d?|1\d\d|2[01]\d|22[0-3])(?:\.(?:1?\d{1,2}|2[0-4]\d|25[0-5])){2}(?:\.(?:[1-9]\d?|1\d\d|2[0-4]\d|25[0-4]))|(?:(?:[a-z0-9\u00a1-\uffff][a-z0-9\u00a1-\uffff_-]{0,62})?[a-z0-9\u00a1-\uffff]\.)+(?:[a-z\u00a1-\uffff]{2,}\.?))(?::\d{2,5})?(?:[/?#]\S*)?)"),
     [](const std::string& str) { return str.find("//") != std::string::npos; }},
    // Email (from https://html.spec.whatwg.org/multipage/input.html#valid-e-mail-address,
    // final character '+' changed from '*')
    {"__ent_email_",
     regex::regex(
         R"([a-zA-Z0-9.!#$%&'*+\/=?^_`{|}~-]+@[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?(?:\.[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?)+)"),
     [](const std::string& str) { return str.find('@') != std::string::npos; }},
    // Credit card (based on https://stackoverflow.com/a/23231321/3832970, test:
    // https://regex101.com/r/hnvNqh/2
    {"__ent_ccard_",
     regex::regex(
         R"((?<!\d)(?:3[47][0-9]{13}|(6541|6556)[0-9]{12}|389[0-9]{11}|3(?:0[0-5]|[68][0-9])[0-9]{11}|65[4-9][0-9]{13}|64[4-9][0-9]{13}|6011[0-9]{12}|(622(?:12[6-9]|1[3-9][0-9]|[2-8][0-9][0-9]|9[01][0-9]|92[0-5])[0-9]{10})|63[7-9][0-9]{13}|(?:2131|1800|35\d{3})\d{11}|9[0-9]{15}|(6304|6706|6709|6771)[0-9]{12,15}|(5018|5020|5038|6304|6759|6761|6763)[0-9]{8,15}|(5[1-5][0-9]{14}|2(22[1-9][0-9]{12}|2[3-9][0-9]{13}|[3-6][0-9]{14}|7[0-1][0-9]{13}|720[0-9]{12}))|(6334|6767)[0-9]{12}|(6334|6767)[0-9]{14}|(6334|6767)[0-9]{15}|(4903|4905|4911|4936|6333|6759)[0-9]{12}|(4903|4905|4911|4936|6333|6759)[0-9]{14}|(4903|4905|4911|4936|6333|6759)[0-9]{15}|564182[0-9]{10}|564182[0-9]{12}|564182[0-9]{13}|633110[0-9]{10}|633110[0-9]{12}|633110[0-9]{13}|(62[0-9]{14,17})|4[0-9]{12}(?:[0-9]{3})?|(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}))(?!\d))"),
     [](const std::string& str) { return true; }},
    // Numeric Date (no leap year checking, https://regex101.com/r/dyzdkT/2)
    {"__ent_date_",
     regex::regex(
         R"((?<!\d)(?<!\d[- /.])(?:(?:19|20)?\d\d([- /.])(?:0[1-9]|1[012])\1(?:0[1-9]|[12][0-9]|3[01])|(?:0[1-9]|1[012])([- /.])(?:0[1-9]|[12][0-9]|3[01])\2(?:19|20)?\d\d|(?:0[1-9]|[12][0-9]|3[01])([- /.])(?:0[1-9]|1[012])\3(?:19|20)?\d\d)(?![- /.]?\d))"),
     [](const std::string& str) { return true; }},
    // Phone (Test: https://regex101.com/r/qmMD9w/1)
    {"__ent_phone_",
     regex::regex(
         R"((?<!\d)(?<!\d[\t\p{Zs}-])(?!\s)(?:\+?\d+[\t\p{Zs}-]?)?[\(\[\t\p{Zs}-]{0,2}\d{3,5}[]).\t\p{Zs}-]{0,2}\d{2,3}[.\t\p{Zs}-]?\d{2,4}(?:[.\t\p{Zs}-]?\d{2,4})?(?![.\t\p{Zs}-]?\d))"),
     [](const std::string& str) { return true; }},
};

static const size_t placeHolderListLength = sizeof(placeHolderList) / sizeof(PlaceHolderProps);

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

  static std::string maskURLsEmailAddysEtc(
      const std::string& text,
      std::vector<std::pair<std::size_t, std::string>>& replacedStrings) {
    std::string masked = text;

    for(size_t i = 0; i < placeHolderListLength; ++i) {
      const auto& phItem = placeHolderList[i];
      if(std::get<2>(phItem)(masked)) {
        masked = regex::regex_replace(
            masked, std::get<1>(phItem), [&replacedStrings, i](const regex::smatch& m) {
              std::string str = m.str(0);
              if(str.find("__ent_") == std::string::npos) {
                replacedStrings.emplace_back(i, str);
                return std::string(std::get<0>(placeHolderList[i]));
              }
              return str;
            });
      }
    }
    return masked;
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

  Words encode(const std::string& line, bool addEOS, bool inference) const override {
    InputFormat inputFormat = ConvertInputFormat(options_->get<std::string>("input-format", ""));
    Words words;
    std::vector<int> spmIds;
    std::vector<std::pair<std::size_t, std::string>> replacedStrings;
    if(inference || alpha_ == 0) {
      if(!inference || inputFormat == InputFormat::PLAINTEXT) {
        std::string masked = maskURLsEmailAddysEtc(line, replacedStrings);
        spm_->Encode(masked, &spmIds);
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
            prefix = decodeEntities(prefix, inputFormat);
            prefix = maskURLsEmailAddysEtc(prefix, replacedStrings);
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

          char tagSpacing = TAGSPACING_NONE;
          if(p > 0 && std::isspace(static_cast<unsigned char>(line[p - 1]))) {
            tagSpacing |= TAGSPACING_BEFORE;
          }

          if(q + 1 < line.length() && std::isspace(static_cast<unsigned char>(line[q + 1]))) {
            tagSpacing |= TAGSPACING_AFTER;
          }

          if(!entitizeTags) {
            words.emplace_back(Word::fromWordIndexAndTag(
                (std::size_t)-1, line.substr(p, q - p + 1), tagType, tagSpacing));
          } else {
            words.emplace_back(Word::fromWordIndexAndTag(
                (std::size_t)addDummyPrefix, line.substr(p, q - p + 1), tagType, tagSpacing));
          }
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

          for(size_t k = startWordInd, entitizedTagId = 1; k <= lastWordInd; ++k) {
            const auto& markupTag = words[k].getMarkupTag();
            if(markupTag) {
              Word newWordTag = Word::fromWordIndexAndTag(entitizedTagId,
                                                          markupTag->getTag(),
                                                          markupTag->getType(),
                                                          markupTag->getSpacing());
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
    for(const auto& r : replacedStrings) {
      words.push_back(
          Word::fromWordIndexAndTag(r.first, r.second, TagType::NONE, TAGSPACING_NONE));
    }
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
      std::vector<size_t> placeHolderIndexes;
      if(inputFormat == InputFormat::PLAINTEXT) {
        for(size_t i = 0; i < sentence.size(); ++i) {
          const auto& word = sentence[i];
          WordIndex wordIndex = word.toWordIndex();
          if(!word.getMarkupTag()) {
            spmSentence.push_back(wordIndex);
          } else if(word.getMarkupTag()->getType() == TagType::NONE) {
            placeHolderIndexes.push_back(i);
          }
        }
        spm_->Decode(spmSentence, &line);
      } else {
        bool entitizeTags = options_->get<bool>("entitize-tags", false);
        std::vector<size_t> entitizedTagIndexes;
        std::vector<bool> spacePrefix;
        spacePrefix.reserve(sentence.size());
        for(size_t i = 0; i < sentence.size(); ++i) {
          const auto& word = sentence[i];
          if(!word.getMarkupTag()) {
            const auto& curWord = (*this)[word];
            bool spaceRequiredBeforeWord = false;
            // from, in SP: const absl::string_view kSpaceSymbol = "\xe2\x96\x81";
            if(curWord.size() >= 3 && curWord[0] == (char)0xe2 && curWord[1] == (char)0x96
               && curWord[2] == (char)0x81) {
              spaceRequiredBeforeWord = true;
            }
            spacePrefix.push_back(spaceRequiredBeforeWord);
          } else {
            if(word.toWordIndex() != (WordIndex)-1) {
              break;
            }
            spacePrefix.push_back(false);
          }
        }

        bool sentenceHasSpaces
            = (std::find(spacePrefix.begin(), spacePrefix.end(), true) != spacePrefix.end());
        for(size_t i = 0; i < sentence.size();) {
          const auto& word = sentence[i];
          WordIndex wordIndex = word.toWordIndex();
          if(!word.getMarkupTag()) {
            spmSentence.push_back(wordIndex);
            ++i;
          } else if(wordIndex == (WordIndex)-1) {
            // collect all adjacent tags; find next real word
            TagType tt = word.getMarkupTag()->getType();
            size_t j = i + 1;
            for(; j < sentence.size() && sentence[j].getMarkupTag()
                  && sentence[j].toWordIndex() == (WordIndex)-1;
                ++j) {
              // need to track whether the tags are all of the same type: open or close, empty tags
              // being neutral
              if(sentence[j].getMarkupTag()->getType() != TagType::EMPTY_TAG) {
                if(tt == TagType::EMPTY_TAG) {
                  tt = sentence[j].getMarkupTag()->getType();
                } else if(sentence[j].getMarkupTag()->getType() != tt) {
                  tt = TagType::NONE;
                }
              }
            }

            bool done = false;
            bool spaceRequiredBeforeNextWord = false;
            if(i > 0 && j < spacePrefix.size() && !sentence[j].getMarkupTag()
               && sentence[j] != getEosId()) {
              if(spacePrefix[j]) {
                spaceRequiredBeforeNextWord = true;
              } else if(sentenceHasSpaces && tt != TagType::NONE && j + 1 < sentence.size()) {
                // prevent the tags from appearing in the middle of the word
                // sentence has spaces, and the adjacent tags are all open or close (possibly with
                // self-closing mixed in)
                // if open, move left, if close, move right to where there is a space
                // deal with everything here:
                done = true;
                if(tt != TagType::CLOSE_TAG) {
                  size_t previousWordsEndIdx = spmSentence.size();
                  for(size_t k = 0; k < spmSentence.size(); ++k) {
                    std::string wrd = (*this)[sentence[i - k - 1]];
                    if(((wrd.length() == 1 && k == i - 1) || (spacePrefix[i - k - 1] && wrd.length() == 4))
                       && std::ispunct(wrd.back())) {
                      // tag comes after the punctuation
                      previousWordsEndIdx = spmSentence.size() - k;
                      break;
                    }
                    if(spacePrefix[i - k - 1]) {
                      previousWordsEndIdx = spmSentence.size() - k - 1;
                      break;
                    }
                  }

                  bool spaceRequired = false;
                  std::vector<int> spmTwo;
                  if(!spmSentence.empty()) {
                    if(previousWordsEndIdx < spmSentence.size()) {
                      size_t idx = i - spmSentence.size() + previousWordsEndIdx;
                      spaceRequired = idx > 0 && spacePrefix[idx];
                      auto it = std::next(spmSentence.begin(), previousWordsEndIdx);
                      std::move(it, spmSentence.end(), std::back_inserter(spmTwo));
                      spmSentence.erase(it, spmSentence.end());
                    }
                    std::string detokenized;
                    spm_->Decode(spmSentence, &detokenized);
                    spmSentence.clear();
                    line += encodeSpecialChars(detokenized);
                  }

                  if(spaceRequired) {
                    line += ' ';
                  }

                  bool emptyLine = line.empty();
                  for(size_t m = i; m < j; ++m) {
                    const auto& markupTag = sentence[m].getMarkupTag();
                    if(spaceRequired && !line.empty() && line.back() != ' '
                       && (markupTag->getSpacing() & TAGSPACING_BEFORE) != 0) {
                      line += ' ';
                    }

                    line += markupTag->getTag();
                    if((spaceRequired || emptyLine)
                       && (markupTag->getSpacing() & TAGSPACING_AFTER) != 0) {
                      line += ' ';
                    }
                  }

                  if(!spmTwo.empty()) {
                    std::string detokenized;
                    spm_->Decode(spmTwo, &detokenized);
                    line += encodeSpecialChars(detokenized);
                  }
                } else {
                  // closing tag(s), move right
                  size_t k = j;
                  for(; k < spacePrefix.size() && !spacePrefix[k]; ++k) {
                    if(sentence[k].getMarkupTag()) {
                      if(sentence[k].toWordIndex() == (WordIndex)-1
                         && sentence[k].getMarkupTag()->getType() != TagType::CLOSE_TAG) {
                        // the next real word must require a leading space
                        size_t l = k + 1;
                        for(; l < spacePrefix.size() && sentence[l].getMarkupTag(); ++l) {
                        }

                        // no change to the position of the tag to place
                        if(l < spacePrefix.size() && !spacePrefix[l]) {
                          k = j;
                        }
                        break;
                      }
                    } else if((k + 1 >= spacePrefix.size() || spacePrefix[k + 1]
                               || sentence[k + 1] == getEosId())
                              && (*this)[sentence[k]].length() == 1
                              && std::ispunct((*this)[sentence[k]].front())) {
                      break;
                    }
                  }

                  for(size_t l = j; l < k; ++l) {
                    if(!sentence[l].getMarkupTag()) {
                      spmSentence.push_back(sentence[l].toWordIndex());
                    }
                  }

                 if(!spmSentence.empty()) {
                    std::string detokenized;
                    spm_->Decode(spmSentence, &detokenized);
                    spmSentence.clear();
                    line += encodeSpecialChars(detokenized);
                  }

                  j = k;
                  bool spaceRequired = (j < spacePrefix.size() && spacePrefix[j]);
                  for(size_t m = i; m < j; ++m) {
                    const auto& markupTag = sentence[m].getMarkupTag();
                    if(markupTag && sentence[m].toWordIndex() == (WordIndex)-1) {
                      if(spaceRequired && !line.empty() && line.back() != ' '
                         && (markupTag->getSpacing() & TAGSPACING_BEFORE) != 0) {
                        line += ' ';
                      }

                      line += markupTag->getTag();
                      if(spaceRequired && (markupTag->getSpacing() & TAGSPACING_AFTER) != 0) {
                        line += ' ';
                      }
                    }
                  }

                  if(spaceRequired && !line.empty() && line.back() != ' ') {
                    line += ' ';
                  }
                }
              }
            }

            if(!done) {
              if(!spmSentence.empty()) {
                std::string detokenized;
                spm_->Decode(spmSentence, &detokenized);
                spmSentence.clear();
                line += encodeSpecialChars(detokenized);
              }

              bool emptyLine = line.empty();
              bool spaceNeededBeforeOpenTag
                  = spaceRequiredBeforeNextWord && tt != TagType::CLOSE_TAG;
              if(spaceNeededBeforeOpenTag) {
                for(size_t k = i; k < j; ++k) {
                  const auto& markupTag = sentence[k].getMarkupTag();
                  if((markupTag->getSpacing() & TAGSPACING_BEFORE) != 0
                     || (markupTag->getSpacing() & TAGSPACING_AFTER) != 0) {
                    spaceNeededBeforeOpenTag = false;
                    break;
                  }
                }
              }
              bool spaceAdded = false;
              for(size_t k = i; k < j; ++k) {
                const auto& markupTag = sentence[k].getMarkupTag();
                if(!line.empty() && line.back() != ' ') {
                  if(spaceNeededBeforeOpenTag
                     && markupTag->getType() != TagType::CLOSE_TAG) {
                    line += ' ';
                    spaceNeededBeforeOpenTag = false;
                    spaceAdded = true;
                  } else if((spaceRequiredBeforeNextWord || j + 1 >= sentence.size())
                            && (markupTag->getSpacing() & TAGSPACING_BEFORE) != 0) {
                    line += ' ';
                    spaceAdded = true;
                  }
                }
                line += markupTag->getTag();
                if((spaceRequiredBeforeNextWord || j + 1 >= sentence.size() || emptyLine)
                   && (markupTag->getSpacing() & TAGSPACING_AFTER) != 0) {
                  line += ' ';
                  spaceAdded = true;
                }
              }
              if(spaceRequiredBeforeNextWord && !spaceAdded && !line.empty()
                 && tt == TagType::CLOSE_TAG) {
                line += ' ';
              }
            }
            i = j;
          } else if(word.getMarkupTag()->getType() == TagType::NONE) {
            placeHolderIndexes.push_back(i);
            ++i;
          } else {
            entitizedTagIndexes.push_back(i);
            ++i;
          }
        }

        if(!spmSentence.empty()) {
          std::string detokenized;
          spm_->Decode(spmSentence, &detokenized);
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
                  p, entity.length(), sentence[i].getMarkupTag()->getTag());
            } else {
              line += sentence[i].getMarkupTag()->getTag();
            }
          }

          // remove any stray tag entities
          line = regex::regex_replace(line, regex::regex("__ent_[0-9]+_"), "");
        }
      }

      line = regex::regex_replace(
          line,
          regex::regex("__ent_[^_]+_"),
          [&placeHolderIndexes, &sentence](const regex::smatch& m) {
            std::string str(m.str(0));
            for(size_t i = 0; i < placeHolderListLength; ++i) {
              if(str == std::get<0>(placeHolderList[i])) {
                for(auto it = placeHolderIndexes.begin(); it != placeHolderIndexes.end(); ++it) {
                  if(sentence[*it].toWordIndex() == i) {
                    size_t j = *it;
                    placeHolderIndexes.erase(it);
                    return sentence[j].getMarkupTag()->getTag();
                  }
                }
                return std::string();
              }
            }
            return std::string();
          });
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
