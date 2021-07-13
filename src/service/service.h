#pragma once

#include <string>
#include <memory>

#include "data/batch_generator.h"
#include "data/corpus.h"
#include "data/shortlist.h"
#include "data/text_input.h"

#if USE_PTHREADS
#include "3rd_party/threadpool.h"
#endif

#include "translator/history.h"
#include "translator/output_collector.h"
#include "translator/output_printer.h"

#include "translator/scorers.h"

#include "rescorer/rescorer.h"

// currently for diagnostics only, will try to mmap files ending in *.bin suffix when enabled.
#include "3rd_party/mio/mio.hpp"

namespace marian {

template <class Search, class Model>
class TranslateService {
private:
  Ptr<Options> options_;
  std::vector<Ptr<ExpressionGraph>> graphs_;
  std::vector<std::vector<Ptr<Scorer>>> scorers_;
  std::vector<Ptr<Model>> models_;

  std::vector<Ptr<Vocab>> srcVocabs_;
  Ptr<Vocab> trgVocab_;
  Ptr<const data::ShortlistGenerator> shortlistGenerator_;
#if USE_PTHREADS
  std::unique_ptr<ThreadPool> threadPool_;
#endif

  size_t numDevices_;

  Ptr<std::vector<mio::mmap_source>> model_mmaps_;       // map
  Ptr<std::vector<std::vector<io::Item>>> model_items_;  // non-mmap

public:
  virtual ~TranslateService() {}

  TranslateService(const TranslateService& other)
      : options_(New<Options>(other.options_->clone())),
        srcVocabs_(other.srcVocabs_),
        trgVocab_(other.trgVocab_),
        shortlistGenerator_(other.shortlistGenerator_),
        model_mmaps_(other.model_mmaps_),
        model_items_(other.model_items_) {
    initScorers();
  }

  TranslateService(Ptr<Options> options)
    : options_(New<Options>(options->clone())) {
    // initialize vocabs
    options_->set("inference", true);
    options_->set("shuffle", "none");

    auto vocabPaths = options_->get<std::vector<std::string>>("vocabs");
    std::vector<int> maxVocabs = options_->get<std::vector<int>>("dim-vocabs");

    for(size_t i = 0; i < vocabPaths.size() - 1; ++i) {
      Ptr<Vocab> vocab = New<Vocab>(options_, i);
      vocab->load(vocabPaths[i], maxVocabs[i]);
      srcVocabs_.emplace_back(vocab);
    }

    trgVocab_ = New<Vocab>(options_, vocabPaths.size() - 1);
    trgVocab_->load(vocabPaths.back());

    // load lexical shortlist
    if(options_->hasAndNotEmpty("shortlist")) {
      auto slOptions = options_->get<std::vector<std::string>>("shortlist");
      ABORT_IF(slOptions.empty(), "No path to shortlist file given");
      std::string filename = slOptions[0];
      if(data::isBinaryShortlist(filename))
        shortlistGenerator_ = New<data::BinaryShortlistGenerator>(
            options_, srcVocabs_.front(), trgVocab_, 0, 1, vocabPaths.front() == vocabPaths.back());
      else
        shortlistGenerator_ = New<data::LexicalShortlistGenerator>(
            options_, srcVocabs_.front(), trgVocab_, 0, 1, vocabPaths.front() == vocabPaths.back());
    }

    auto models = options->get<std::vector<std::string>>("models");
    if(options_->get<bool>("model-mmap", false)) {
      model_mmaps_ = New<std::vector<mio::mmap_source>>();
      for(auto model : models) {
        ABORT_IF(!io::isBin(model), "Non-binarized models cannot be mmapped");
        model_mmaps_->push_back(mio::mmap_source(model));
      }
    } else {
      model_items_ = New<std::vector<std::vector<io::Item>>>();
      for(auto model : models) {
        auto items = io::loadItems(model);
        model_items_->push_back(std::move(items));
      }
    }
    initScorers();
  }

  std::string translate(const std::string& input,
                        const size_t beamSize,
                        const std::string& textFormat) {
    options_->set("beam-size", beamSize);
    options_->set("input-format", textFormat);
    // split tab-separated input into fields if necessary
    auto inputs = options_->get<bool>("tsv", false)
                      ? convertTsvToLists(input, options_->get<size_t>("tsv-fields", 1))
                      : std::vector<std::string>({input});
    auto corpus = New<data::TextInput>(inputs, srcVocabs_, options_);
#if USE_PTHREADS
    data::BatchGenerator<data::TextInput> batchGenerator(corpus, options_);
    std::vector<std::future<void>> futures;
#else
    // Set to false to run non-async mode
    data::BatchGenerator<data::TextInput> batchGenerator(corpus, options_, nullptr, false);
#endif
    batchGenerator.prepare();

    auto collector = New<StringCollector>(options_->get<bool>("quiet-translation", false));
    auto printer = New<OutputPrinter>(options_, trgVocab_);
    size_t batchId = 0;
    for(auto batch : batchGenerator) {
      auto task = [=](size_t id) {
        size_t deviceIndex = id % numDevices_;
        thread_local Ptr<ExpressionGraph> graph = graphs_[deviceIndex];
        thread_local std::vector<Ptr<Scorer>> scorers = scorers_[deviceIndex];

        auto search = New<Search>(options_, scorers, trgVocab_);
        auto histories = search->search(graph, batch);

        for(auto history : histories) {
          std::stringstream best1;
          std::stringstream bestn;
          printer->print(history, best1, bestn);
          collector->add((long)history->getLineNum(), best1.str(), bestn.str());
        }
      };

#if USE_PTHREADS
      futures.push_back(threadPool_->enqueue(task, batchId++));
#else
      task(batchId++);
#endif
    }

#if USE_PTHREADS
    for(auto& f : futures) {
      f.wait();
    }
#endif

    auto translations = collector->collect(options_->get<bool>("n-best"));
    return utils::join(translations, "\n");
  }

  std::string placeTagsInTarget(const std::string& sourceText,
                                const std::string& targetText,
                                const std::string& textFormat) {
    options_->set("input-format", textFormat);
    InputFormat inputFormat = ConvertInputFormat(textFormat);
    bool entitizeTags = options_->get<bool>("entitize-tags");
    std::vector<std::string> inputs({sourceText, targetText});
    std::vector<Ptr<Vocab>> vocabs(srcVocabs_);
    vocabs.push_back(trgVocab_);
    auto corpus = New<data::TextInput>(inputs, vocabs, options_);
#if USE_PTHREADS
    data::BatchGenerator<data::TextInput> batchGenerator(corpus, options_);
    std::vector<std::future<void>> futures;
#else
    // Set to false to run non-async mode
    data::BatchGenerator<data::TextInput> batchGenerator(corpus, options_, nullptr, false);
#endif
    batchGenerator.prepare();

    auto collector = New<StringCollector>(options_->get<bool>("quiet-translation", false));
    size_t batchId = 0;
    for(auto batch : batchGenerator) {
      auto task = [=](size_t id) {
        size_t deviceIndex = id % numDevices_;
        thread_local Ptr<ExpressionGraph> graph = graphs_[deviceIndex];
        thread_local Ptr<Model> builder = models_[deviceIndex];

        // @TODO: normalize by length as in normalize
        // Once we have Frank's concept of ce-sum with sample size by words we will return a pair
        // here which will make it trivial to report all variants.
        auto dynamicLoss = builder->build(graph, batch, inputFormat);

        graph->forward();

        // get loss
        std::vector<float> scoresForSummary;
        dynamicLoss->loss(scoresForSummary);

          // soft alignments for each sentence in the batch
        std::vector<data::SoftAlignment> aligns(
            batch->size());  // @TODO: do this resize inside getAlignmentsForBatch()
        Rescore<Model>::getAlignmentsForBatch(builder->getAlignment(), batch, aligns);
        for(size_t i = 0; i < batch->size(); ++i) {
          std::string translation = Rescore<Model>::getTargetWithTagsPlaced(
              batch, aligns, i, inputFormat, entitizeTags);
          collector->add((long)batch->getSentenceIds()[i], translation, std::string());
        }
      };

#if USE_PTHREADS
      futures.push_back(threadPool_->enqueue(task, batchId++));
#else
      task(batchId++);
#endif
    }

#if USE_PTHREADS
    for(auto& f : futures) {
      f.wait();
    }
#endif

    auto translations = collector->collect(false);
    return utils::join(translations, "\n");
  }

private:
  void initScorers() {
    // get device IDs
    auto devices = Config::getDevices(options_);
    numDevices_ = devices.size();
#if USE_PTHREADS
    threadPool_ = std::make_unique<ThreadPool>(numDevices_, numDevices_);
#endif

    // initialize scorers
    for(auto device : devices) {
      auto graph = New<ExpressionGraph>(true);

      auto prec = options_->get<std::vector<std::string>>("precision", {"float32"});
      graph->setDefaultElementType(typeFromString(prec[0]));
      graph->setDevice(device);
      graph->getBackend()->configureDevice(options_);
      graph->reserveWorkspaceMB(options_->get<size_t>("workspace"));
      graphs_.push_back(graph);

      std::vector<Ptr<Scorer>> scorers;
      if(options_->get<bool>("model-mmap", false)) {
        scorers = createScorers(options_, *model_mmaps_);
      } else {
        scorers = createScorers(options_, *model_items_);
      }

      for(auto scorer : scorers) {
        scorer->init(graph);
        if(shortlistGenerator_)
          scorer->setShortlistGenerator(shortlistGenerator_);
      }

      scorers_.push_back(scorers);
      graph->forward();
    }

    auto models = options_->get<std::vector<std::string>>("models");
    models_.resize(graphs_.size());
#if USE_PTHREADS
    ThreadPool pool(graphs_.size(), graphs_.size());
#endif
    for(size_t i = 0; i < graphs_.size(); ++i) {
      auto task = [=](size_t j) {
        models_[j] = New<Model>(options_);
        if(model_mmaps_) {
          models_[j]->mmap(graphs_[j], model_mmaps_->front().data());
        } else {
          models_[j]->load(graphs_[j], models.front());
        }
      };
#if USE_PTHREADS
      pool.enqueue(task, i);
#else
      task(i);
#endif
    }
  }

  // Converts a multi-line input with tab-separated source(s) and target sentences into separate lists
  // of sentences from source(s) and target sides, e.g.
  // "src1 \t trg1 \n src2 \t trg2" -> ["src1 \n src2", "trg1 \n trg2"]
  std::vector<std::string> convertTsvToLists(const std::string& inputText, size_t numFields) {
    std::vector<std::string> outputFields(numFields);

    std::string line;
    std::vector<std::string> lineFields(numFields);
    std::istringstream inputStream(inputText);
    bool first = true;
    while(std::getline(inputStream, line)) {
      utils::splitTsv(line, lineFields, numFields);
      for(size_t i = 0; i < numFields; ++i) {
        if(!first)
          outputFields[i] += "\n";  // join sentences with a new line sign
        outputFields[i] += lineFields[i];
      }
      if(first)
        first = false;
    }

    return outputFields;
  }
};
}  // namespace marian