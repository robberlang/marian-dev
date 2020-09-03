#pragma once

#include <string>

namespace marian {

struct ModelTask {
  virtual ~ModelTask() {}
  virtual void run() = 0;
};

struct ModelServiceTask {
  virtual ~ModelServiceTask() {}
  virtual std::string run(const std::string& input,
                          const size_t beamSize,
                          const std::string& inputFormat)
      = 0;
};
}  // namespace marian
