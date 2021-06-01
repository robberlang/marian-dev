#include "data/text_input.h"
#include "common/utils.h"

namespace marian {
namespace data {

TextIterator::TextIterator() : pos_(-1), tup_(0) {}
TextIterator::TextIterator(TextInput& corpus) : corpus_(&corpus), pos_(0), tup_(corpus_->next()) {}

void TextIterator::increment() {
  tup_ = corpus_->next();
  pos_++;
}

bool TextIterator::equal(TextIterator const& other) const {
  return this->pos_ == other.pos_ || (this->tup_.empty() && other.tup_.empty());
}

const SentenceTuple& TextIterator::dereference() const {
  return tup_;
}

TextInput::TextInput(const std::vector<std::string>& inputs,
                     std::vector<Ptr<Vocab>> vocabs,
                     Ptr<Options> options)
    : DatasetBase(inputs, options),
      vocabs_(std::move(vocabs)),
      maxLength_(options_->get<size_t>("max-length")),
      maxLengthCrop_(options_->get<bool>("max-length-crop")) {
  // Note: inputs are automatically stored in the inherited variable named paths_, but these are
  // texts not paths!
  for(const auto& text : paths_)
    files_.emplace_back(new std::istringstream(text));
}

// TextInput is mainly used for inference in the server mode, not for training, so skipping too long
// or ill-formed inputs is not necessary here
SentenceTuple TextInput::next() {
  // get index of the current sentence
  size_t curId = pos_++;

  // fill up the sentence tuple with source and/or target sentences
  SentenceTuple tup(curId);
  for(size_t i = 0; i < files_.size(); ++i) {
    std::string line;
    if(io::getline(*files_[i], line)) {
      Words words
          = vocabs_[i]->encode(line, /*addEOS =*/true, inference_, inputFormat_, entitizeTags_);
      if(maxLengthCrop_ && words.size() > maxLength_
         && static_cast<size_t>(std::count_if(words.begin(), words.end(), [](const Word& w) {
              return !w.getMarkupTag().operator bool();
            })) > maxLength_) {
        size_t maxLength = maxLength_ - 1;
        size_t wordCount = 0;
        for(auto it = words.begin(); it != words.end(); ++it) {
          if(!it->getMarkupTag() && ++wordCount >= maxLength) {
            for(++it; it != words.end();) {
              if(!it->getMarkupTag()) {
                it = words.erase(it);
              } else {
                ++it;
              }
            }
            break;
          }
        }
        words.push_back(vocabs_[i]->getEosId());
      }

      tup.push_back(words);
    }
  }

  if(tup.size() == files_.size()) // check if each input file provided an example
    return tup;
  else if(tup.size() == 0) // if no file provided examples we are done
    return SentenceTuple(0);
  else // neither all nor none => we have at least on missing entry
    ABORT("There are missing entries in the text tuples.");
}

}  // namespace data
}  // namespace marian
