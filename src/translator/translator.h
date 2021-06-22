#pragma once

#include <string>
#include <memory>

#include "data/batch_generator.h"
#include "data/corpus.h"
#include "data/shortlist.h"
#include "data/text_input.h"

#include "3rd_party/threadpool.h"

#include "translator/history.h"
#include "translator/output_collector.h"
#include "translator/output_printer.h"

#include "models/model_task.h"
#include "translator/scorers.h"

// currently for diagnostics only, will try to mmap files ending in *.bin suffix when enabled.
#include "3rd_party/mio/mio.hpp"

namespace marian {

template <class Search>
class Translate : public ModelTask {
private:
  Ptr<Options> options_;
  std::vector<Ptr<ExpressionGraph>> graphs_;
  std::vector<std::vector<Ptr<Scorer>>> scorers_;

  Ptr<data::Corpus> corpus_;
  Ptr<Vocab> trgVocab_;
  Ptr<const data::ShortlistGenerator> shortlistGenerator_;

  size_t numDevices_;

  std::vector<mio::mmap_source> mmaps_;

public:
  Translate(Ptr<Options> options)
    : options_(New<Options>(options->clone())) { // @TODO: clone should return Ptr<Options> same as "with"?
    // This is currently safe as the translator is either created stand-alone or
    // or config is created anew from Options in the validator

    options_->set("inference", true,
                  "shuffle", "none");

    corpus_ = New<data::Corpus>(options_, true);

    auto vocabs = options_->get<std::vector<std::string>>("vocabs");
    trgVocab_ = New<Vocab>(options_, vocabs.size() - 1);
    trgVocab_->load(vocabs.back());
    auto srcVocab = corpus_->getVocabs()[0];

    if(options_->hasAndNotEmpty("shortlist"))
      shortlistGenerator_ = New<data::LexicalShortlistGenerator>(
          options_, srcVocab, trgVocab_, 0, 1, vocabs.front() == vocabs.back());

    auto devices = Config::getDevices(options_);
    numDevices_ = devices.size();

    ThreadPool threadPool(numDevices_, numDevices_);
    scorers_.resize(numDevices_);
    graphs_.resize(numDevices_);

    if(options_->get<bool>("model-mmap", false)) {
      auto models = options_->get<std::vector<std::string>>("models");
      for(auto model : models) {
        ABORT_IF(!io::isBin(model), "Non-binarized models cannot be mmapped");
        mmaps_.push_back(mio::mmap_source(model));
      }
    }

    size_t id = 0;
    for(auto device : devices) {
      auto task = [&](DeviceId device, size_t id) {
        auto graph = New<ExpressionGraph>(true);
        auto prec = options_->get<std::vector<std::string>>("precision", {"float32"});
        graph->setDefaultElementType(typeFromString(prec[0]));
        graph->setDevice(device);
        graph->reserveWorkspaceMB(options_->get<size_t>("workspace"));
        graphs_[id] = graph;

        std::vector<Ptr<Scorer>> scorers;
        if(options_->get<bool>("model-mmap", false)) {
          scorers = createScorers(options_, mmaps_);
        } else {
          scorers = createScorers(options_);
        }
        for(auto scorer : scorers) {
          scorer->init(graph);
          if(shortlistGenerator_)
            scorer->setShortlistGenerator(shortlistGenerator_);
        }

        scorers_[id] = scorers;
        graph->forward();
      };

      threadPool.enqueue(task, device, id++);
    }

    if(options_->get<bool>("output-sampling", false)) {
      if(options_->get<size_t>("beam-size") > 1)
        LOG(warn,
            "[warning] Output sampling and beam search (beam-size > 1) are contradictory methods "
            "and using them together is not recommended. Set beam-size to 1");
      if(options_->get<std::vector<std::string>>("models").size() > 1)
        LOG(warn,
            "[warning] Output sampling and model ensembling are contradictory methods and using "
            "them together is not recommended. Use a single model");
    }
  }

  void run() override {
    data::BatchGenerator<data::Corpus> bg(corpus_, options_);

    ThreadPool threadPool(numDevices_, numDevices_);

    size_t batchId = 0;
    auto collector = New<OutputCollector>(options_->get<std::string>("output"));
    auto printer = New<OutputPrinter>(options_, trgVocab_);
    if(options_->get<bool>("quiet-translation"))
      collector->setPrintingStrategy(New<QuietPrinting>());

    bg.prepare();

    bool doNbest = options_->get<bool>("n-best");
    for(auto batch : bg) {
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
          collector->Write((long)history->getLineNum(),
                           best1.str(),
                           bestn.str(),
                           doNbest);
        }


        // progress heartbeat for MS-internal Philly compute cluster
        // otherwise this job may be killed prematurely if no log for 4 hrs
        if (getenv("PHILLY_JOB_ID")   // this environment variable exists when running on the cluster
            && id % 1000 == 0)  // hard beat once every 1000 batches
        {
          auto progress = 0.f; //fake progress for now
          fprintf(stderr, "PROGRESS: %.2f%%\n", progress);
          fflush(stderr);
        }
      };

      threadPool.enqueue(task, batchId++);
    }
  }
};
}  // namespace marian
