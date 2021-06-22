#pragma once

#include <string>

namespace marian {

struct ModelTask {
  virtual ~ModelTask() {}
  virtual void run() = 0;
};
}  // namespace marian
