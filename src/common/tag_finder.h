#pragma once

#include <string>
#include <vector>
#include <utility>
#if(_MSC_VER >= 1800) || (__cplusplus >= 201103L)  // C++ 11
#include <type_traits>
#endif

namespace marian {
namespace tagfinder {
constexpr bool isTagWhitespace(int c) {
  return ((c == 0x20) || (c == 0x09) || (c == 0x0a) || (c == 0x0c) || (c == 0x0d));
}

template <typename T>
size_t findNextTagStart(const std::basic_string<T>& str, size_t pos) {
#if(_MSC_VER >= 1800) || (__cplusplus >= 201103L)  // C++ 11
  static_assert(std::is_integral<T>::value && sizeof(T) >= 1 && sizeof(T) <= 4,
                "This function is for 8- to 32-bit integral types.");
#endif

  size_t posFirstLT = str.find('<', pos);
  if(posFirstLT == str.npos || str.length() <= posFirstLT + 1)
    return str.npos;

  T chNext = str[posFirstLT + 1];
  if((chNext < 0x80)
     && ((chNext >= 'A' && chNext <= 'Z') || (chNext >= 'a' && chNext <= 'z') || chNext == '!'
         || chNext == '/' || chNext == ':' || chNext == '?'))
    return posFirstLT;

  return findNextTagStart<T>(str, posFirstLT + 1);
}

template <typename T>
size_t findTagEnd(const std::basic_string<T>& str,
                  size_t posLT,
                  std::vector<std::pair<std::basic_string<T>, std::basic_string<T>>>* attributes
                  = nullptr) {
#if(_MSC_VER >= 1800) || (__cplusplus >= 201103L)  // C++ 11
  static_assert(std::is_integral<T>::value && sizeof(T) >= 1 && sizeof(T) <= 4,
                "This function is for 8- to 32-bit integral types.");
#endif

  if((posLT + 3 < str.length()) && (str[posLT + 1] == '!') && (str[posLT + 2] == '-')
     && (str[posLT + 3] == '-')) {
    // a comment
    static const T commentEnd[4] = {'-', '-', '>', '\0'};
    size_t posCommentEnd = str.find(commentEnd, posLT + 4);
    if(posCommentEnd != str.npos)
      posCommentEnd += 2;
    return posCommentEnd;
  }

  size_t pos1 = posLT + 1;
  size_t k = pos1;
  bool endTag = (k < str.length()) && (str[k] == '/');
  if(endTag)
    ++k;

  // go past non-whitespace
  for(; (k < str.length()) && (str[k] != '>') && (str[k] != '/') && !isTagWhitespace(str[k]); ++k)
    ;

  // go past whitespace
  for(; (k < str.length()) && isTagWhitespace(str[k]); ++k)
    ;
  if(k >= str.length())
    return str.npos;

  if(str[k] == '/') {
    if(endTag || ((k + 1) >= str.length()) || (str[k + 1] != '>'))
      return str.npos;
  } else if(str[k] == '>') {
    return k;
  }

  if(endTag)
    return str.npos;

  do {
    // move past the attribute name
    size_t attname_start = k;
    for(; (k < str.length()) && (str[k] != '>') && (str[k] != '=') && !isTagWhitespace(str[k]); ++k)
      ;

    size_t attname_end = k;
    // go past whitespace
    for(; (k < str.length()) && isTagWhitespace(str[k]); ++k)
      ;

    // move past the attribute value
    size_t attval_start = str.length();
    size_t attval_end = attval_start;
    if((k < str.length()) && str[k] == '=') {
      // go past whitespace
      for(++k; (k < str.length()) && isTagWhitespace(str[k]); ++k)
        ;

      if((k < str.length()) && ((str[k] == '"') || (str[k] == '\''))) {
        attval_start = k + 1;
        k = str.find(str[k], attval_start);
        if(k == str.npos)
          return str.npos;
        attval_end = k;
        ++k;
      } else {
        attval_start = k;
        // go to first whitespace or end of tag, whichever is first
        for(; (k < str.length()) && (str[k] != '>') && !isTagWhitespace(str[k]); ++k)
          ;
        attval_end = k;
      }

      // go past whitespace
      for(; (k < str.length()) && isTagWhitespace(str[k]); ++k)
        ;
    }
    if(attributes) {
      attributes->emplace_back(str.substr(attname_start, attname_end - attname_start),
                               str.substr(attval_start, attval_end - attval_start));
    }
  } while((k < str.length()) && (str[k] != '>') && (str[k] != '/'));

  if(k >= str.length())
    return str.npos;

  if(str[k] == '/') {
    ++k;
    if((k >= str.length()) || (str[k] != '>'))  // invalid tag
      return str.npos;
  }

  return k;
}
}  // namespace tagfinder
}  // namespace marian
