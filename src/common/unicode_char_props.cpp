#include "unicode_char_props.h"
#include "unicode_char_property_lists.h"

#define UNICODE32SET_HIGH 0x0110000

namespace marian {
namespace unicodecharprops {
class UnicodeChar32Set {
public:
  UnicodeChar32Set(const char32_t* list, int lenList) : list_(list), lenList_(lenList) {}

  bool contains(char32_t c) const {
    if(c >= UNICODE32SET_HIGH || !list_ || lenList_ == 0)
      return false;

    int i = findCodePoint(c);
    return (bool)(i & 1);  // return true if odd
  }

private:
  /**
   * Returns the smallest value i such that c < list[i].  Caller
   * must ensure that c is a legal value or this method will enter
   * an infinite loop.  This method performs a binary search.
   * @param c a character in the range MIN_VALUE..MAX_VALUE
   * inclusive
   * @return the smallest integer i in the range 0..len-1,
   * inclusive, such that c < list[i]
   */
  int findCodePoint(char32_t c) const {
    /* Examples:
                                       findCodePoint(c)
       set              list[]         c=0 1 3 4 7 8
       ===              ==============   ===========
       []               [110000]         0 0 0 0 0 0
       [\u0000-\u0003]  [0, 4, 110000]   1 1 1 2 2 2
       [\u0004-\u0007]  [4, 8, 110000]   0 0 0 1 1 2
       [:Any:]          [0, 110000]      1 1 1 1 1 1
     */

    // Return the smallest i such that c < list[i].  Assume
    // list[len - 1] == HIGH and that c is legal (0..HIGH-1).
    if(c < list_[0])
      return 0;
    // High runner test.  c is often after the last range, so an
    // initial check for this condition pays off.
    int lo = 0;
    int hi = lenList_ - 1;
    if(lo >= hi || c >= list_[hi - 1])
      return hi;
    // invariant: c >= list_[lo]
    // invariant: c < list_[hi]
    for(;;) {
      int i = (lo + hi) >> 1;
      if(i == lo) {
        break;  // Found!
      } else if(c < list_[i]) {
        hi = i;
      } else {
        lo = i;
      }
    }
    return hi;
  }

  const char32_t* list_;
  int lenList_;
};

static const UnicodeChar32Set g_PunctuationSet(listPunctuation, lenListPunctuation);
static const UnicodeChar32Set g_NonSpacingSet(listNonSpacing, lenListNonSpacing);

bool isUCharPunct(char32_t ch) {
  return g_PunctuationSet.contains(ch);
}
bool isUCharNonSpacing(char32_t ch) {
  return g_NonSpacingSet.contains(ch);
}
}  // namespace unicodecharprops
}  // namespace marian
