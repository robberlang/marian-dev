#include "history.h"

namespace marian {

History::History(size_t lineNo,
                 std::vector<std::pair<Word, size_t>> lineTags,
                 bool lineSpaceSymbolStart,
                 float alpha,
                 float wp)
    : lineNo_(lineNo),
      lineTags_(std::move(lineTags)),
      lineSpaceSymbolStart_(lineSpaceSymbolStart),
      alpha_(alpha),
      wp_(wp) {}
}  // namespace marian
