#pragma once

#if(__cplusplus >= 201703L)  // C++ 17
#include <string_view>
#endif
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace marian {
namespace tagfinder {
constexpr bool isTagWhitespace(int c) {
  return (c == 0x20 || c == 0x09 || c == 0x0a || c == 0x0c || c == 0x0d);
}

template <typename T>
size_t findNextTagStart(
#if(__cplusplus >= 201703L)  // C++ 17
    std::basic_string_view<T> str,
#else
    const std::basic_string<T>& str,
#endif
    size_t pos) {
  static_assert(std::is_integral<T>::value && sizeof(T) >= 1 && sizeof(T) <= 4,
                "This function is for 8- to 32-bit integral types.");

  size_t posFirstLT = str.find('<', pos);
  if(posFirstLT == str.npos || str.length() <= posFirstLT + 1)
    return str.npos;

  T chNext = str[posFirstLT + 1];
  if(chNext < 0x80
     && ((chNext >= 'A' && chNext <= 'Z') || (chNext >= 'a' && chNext <= 'z') || chNext == '!'
         || chNext == '/' || chNext == ':' || chNext == '?'))
    return posFirstLT;

  return findNextTagStart<T>(str, posFirstLT + 1);
}

#if(__cplusplus >= 201703L)  // C++ 17
template <typename T>
size_t findNextTagStart(const std::basic_string<T>& str, size_t pos) {
  return findNextTagStart(std::basic_string_view<T>(str.data(), str.length()), pos);
}
#endif

template <typename T>
size_t findTagEnd(
#if(__cplusplus >= 201703L)  // C++ 17
    std::basic_string_view<T> str,
    size_t posLT,
    std::vector<std::pair<std::basic_string_view<T>, std::basic_string_view<T>>>* attributes
    = nullptr,
    std::basic_string_view<T>* name = nullptr) {
#else
    const std::basic_string<T>& str,
    size_t posLT,
    std::vector<std::pair<std::basic_string<T>, std::basic_string<T>>>* attributes = nullptr,
    std::basic_string<T>* name = nullptr) {
#endif
  static_assert(std::is_integral<T>::value && sizeof(T) >= 1 && sizeof(T) <= 4,
                "This function is for 8- to 32-bit integral types.");

  if(posLT + 3 < str.length() && str[posLT + 1] == '!' && str[posLT + 2] == '-'
     && str[posLT + 3] == '-') {
    // a comment
    static const T commentEnd[4] = {'-', '-', '>', '\0'};
    size_t posCommentEnd = str.find(commentEnd, posLT + 4);
    if(posCommentEnd != str.npos)
      posCommentEnd += 2;
    return posCommentEnd;
  }

  size_t posNameStart = posLT + 1;
  bool endTag = posNameStart < str.length() && str[posNameStart] == '/';
  if(endTag)
    ++posNameStart;

  size_t k = posNameStart;
  // go past non-whitespace
  for(; k < str.length() && str[k] != '>' && str[k] != '/' && !isTagWhitespace(str[k]); ++k)
    ;

  if(name) {
    *name = str.substr(posNameStart, k - posNameStart);
  }

  // go past whitespace
  for(; k < str.length() && isTagWhitespace(str[k]); ++k)
    ;
  if(k >= str.length())
    return str.npos;

  if(str[k] == '/') {
    return (endTag || k + 1 >= str.length() || str[k + 1] != '>') ? str.npos : k + 1;
  } else if(str[k] == '>') {
    return k;
  }

  if(endTag)
    return str.npos;

  do {
    // move past the attribute name
    size_t attname_start = k;
    for(; k < str.length() && str[k] != '>' && str[k] != '=' && !isTagWhitespace(str[k]); ++k)
      ;

    size_t attname_end = k;
    // go past whitespace
    for(; k < str.length() && isTagWhitespace(str[k]); ++k)
      ;

    // move past the attribute value
    size_t attval_start = str.length();
    size_t attval_end = attval_start;
    if(k < str.length() && str[k] == '=') {
      // go past whitespace
      for(++k; k < str.length() && isTagWhitespace(str[k]); ++k)
        ;

      if(k < str.length() && (str[k] == '"' || str[k] == '\'')) {
        attval_start = k + 1;
        k = str.find(str[k], attval_start);
        if(k == str.npos)
          return str.npos;
        attval_end = k;
        ++k;
      } else {
        attval_start = k;
        // go to first whitespace or end of tag, whichever is first
        for(; k < str.length() && str[k] != '>' && !isTagWhitespace(str[k]); ++k)
          ;
        attval_end = k;
      }

      // go past whitespace
      for(; k < str.length() && isTagWhitespace(str[k]); ++k)
        ;
    }
    if(attributes) {
      attributes->emplace_back(str.substr(attname_start, attname_end - attname_start),
                               str.substr(attval_start, attval_end - attval_start));
    }
  } while(k < str.length() && str[k] != '>' && str[k] != '/');

  if(k >= str.length())
    return str.npos;

  if(str[k] == '/') {
    ++k;
    if(k >= str.length() || str[k] != '>')  // invalid tag
      return str.npos;
  }

  return k;
}

#if(__cplusplus >= 201703L)  // C++ 17
template <typename T>
size_t findTagEnd(const std::basic_string<T>& str,
                  size_t posLT,
                  std::vector<std::pair<std::basic_string<T>, std::basic_string<T>>>* attributes
                  = nullptr,
                  std::basic_string<T>* name = nullptr) {
  if(attributes || name) {
    std::vector<std::pair<std::basic_string_view<T>, std::basic_string_view<T>>> tmpAttributes;
    std::basic_string_view<T> tmpName;
    size_t ret = findTagEnd(std::basic_string_view<T>(str.data(), str.length()),
                            posLT,
                            (attributes ? &tmpAttributes : nullptr),
                            (name ? &tmpName : nullptr));
    if(attributes) {
      for(auto&& att : tmpAttributes) {
        attributes->emplace_back(std::basic_string<T>(att.first), std::basic_string<T>(att.second));
      }
    }
    if(name) {
      name->assign(tmpName.data(), tmpName.length());
    }
    return ret;
  } else {
    return findTagEnd(std::basic_string_view<T>(str.data(), str.length()), posLT);
  }
}
#endif

}  // namespace tagfinder
}  // namespace marian
