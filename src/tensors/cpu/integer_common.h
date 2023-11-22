#pragma once

#include "tensors/tensor_allocator.h"
#include "tensors/tensor_operators.h"
#include "tensors/cpu/aligned.h"
#include "graph/expression_graph.h"
#include "graph/node.h"
#include "graph/node_operators_unary.h"
#include "common/io_item.h"
#ifdef USE_INTGEMM
#include "3rd_party/intgemm/intgemm/intgemm.h"
#else // USE_INTGEMM
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wcomment"
#include <ruy/ruy.h>
#pragma GCC diagnostic pop

#endif // USE_INTGEMM
#if defined(WASM)
#include "wasm_intgemm_interface.h"
#endif

#include <cassert>
#include <cstddef>

namespace marian {
namespace cpu {
namespace integer {

// Making sure we have access to common functions for RUY and INTGEMM
class fetchAlphaFromModelNodeOp : public UnaryNodeOp {
public:
  fetchAlphaFromModelNodeOp(Expr b)
      : UnaryNodeOp(b, Shape({1}), Type::float32) {

    std::string bname = b->name();
    std::string aQuantKey = b->name() + "_QuantMultA";
    //Very Hacky Bit. Unnamed matrix is notpart of the F0 parameter namespace
    if (aQuantKey.at(0) != 'F') {
      aQuantKey = "F0::" + aQuantKey;
    }
    set_name(aQuantKey);
  }

  NodeOps forwardOps() override {
    return {NodeOp(
      auto map = child(0)->graph()->params()->getMap();
      const auto mapiter = map.find(name());
      if (mapiter != map.end()) {
        val_ = mapiter->second->val();
      } else {
        ABORT("We did not find an alpha in the model named: {}.", name());
      }
    )};
  }

  bool equal(Expr node) override {
    if(hash() == node->hash()) return true;
    return false;
  }

  size_t hash() override {
    return std::hash<std::string>{}(name());
  }

  const std::string type() override { return "alphaNodeOp"; }
};

//Convenient function to get rows and columns of a tensor, shadowed by namespace.
inline int cols(Tensor& tensor) { return tensor->shape()[-1]; }
inline int rows(Tensor& tensor) { return tensor->shape().elements() / cols(tensor); }

inline int cols(Shape& shape) { return shape[-1]; }
inline int rows(Shape& shape) { return shape.elements() / cols(shape); }

// This operates on floats after processing so doesn't care about int8_t vs int16_t.
void AddBias(marian::Tensor C, const marian::Tensor Bias);

#ifdef USE_INTGEMM

template<Type type> struct intgemm_;
template <> struct intgemm_<Type::intgemm8> {
  using width = intgemm::Int8;
  using type = int8_t;
};

template <> struct intgemm_<Type::intgemm8ssse3> {
  using width = intgemm::SSSE3::Kernels8;
  using type = int8_t;
};

template <> struct intgemm_<Type::intgemm8avx2> {
  using width = intgemm::AVX2::Kernels8;
  using type = int8_t;
};

template <> struct intgemm_<Type::intgemm8avx512> {
  using width = intgemm::AVX512BW::Kernels8;
  using type = int8_t;
};

template <> struct intgemm_<Type::intgemm8avx512vnni> {
  using width = intgemm::AVX512VNNI::Kernels8;
  using type = int8_t;
};

template <> struct intgemm_<Type::intgemm16> {
  using width = intgemm::Int16;
  using type = int16_t;
};

template <> struct intgemm_<Type::intgemm16sse2> {
  using width = intgemm::SSE2::Kernels16;
  using type = int16_t;
};

template <> struct intgemm_<Type::intgemm16avx2> {
  using width = intgemm::AVX2::Kernels16;
  using type = int16_t;
};

template <> struct intgemm_<Type::intgemm16avx512> {
  using width = intgemm::AVX512BW::Kernels16;
  using type = int16_t;
};

template <Type vtype>
static inline float& getQuantMult(marian::Tensor val) {
#if COMPILE_CPU
  ABORT_IF(!isIntgemm(val->type()), "getQuantMult does not work for type {}", val->type());
  typedef typename intgemm_<vtype>::type Integer;
  return *(reinterpret_cast<float*>(val->data<Integer>() + val->shape().elements()));
#else
  val;
  ABORT("Using intgemm binary models is only supported when compiling marian with -DCOMPILE_CPU=ON.");
#endif
}

static inline Type getIntgemmType(Type vtype) {
#if COMPILE_CPU
  if (vtype == Type::intgemm8) {
    if (intgemm::kCPU == intgemm::CPUType::AVX512VNNI) {
      return Type::intgemm8avx512vnni;
    } else if (intgemm::kCPU == intgemm::CPUType::AVX512BW) {
      return Type::intgemm8avx512;
    } else if (intgemm::kCPU == intgemm::CPUType::AVX2) {
      return Type::intgemm8avx2;
    } else if (intgemm::kCPU == intgemm::CPUType::SSSE3) {
      return Type::intgemm8ssse3;
    } else {
      ABORT("Your CPU doesn't support SSSE3, necessary for 8bit intgemm to work.");
    }
  } else if (vtype == Type::intgemm16) {
    if (intgemm::kCPU > intgemm::CPUType::AVX2) {
      return Type::intgemm16avx512;
    } else if (intgemm::kCPU == intgemm::CPUType::AVX2) {
      return Type::intgemm16avx2;
    } else if (intgemm::kCPU >= intgemm::CPUType::SSE2) {
      return Type::intgemm16sse2;
    } else {
      ABORT("Your CPU doesn't support SSE2, necessary for 16bit intgemm to work.");
    }
  } else {
    ABORT("Unrecognised type {}.", vtype);
  }
#else
  ABORT("Using intgemm binary models is only supported when compiling marian with -DCOMPILE_CPU=ON.");
  return vtype;
#endif
}

static inline bool passOrAbort(Type vtype) {
#if COMPILE_CPU
  if (vtype == Type::intgemm8 || vtype == Type::intgemm16) {
    return true;
  } else if (vtype == Type::intgemm16sse2) {
    ABORT_IF(intgemm::kCPU < intgemm::CPUType::SSE2, "Your CPU doesn't support the architecture necessary to decode model of type {}. Try older architecture instead.", vtype);
  } else if (vtype == Type::intgemm8ssse3) {
    ABORT_IF(intgemm::kCPU < intgemm::CPUType::SSSE3, "Your CPU doesn't support the architecture necessary to decode model of type {}. Try older architecture instead.", vtype);
  } else if (vtype == Type::intgemm8avx2 || vtype == Type::intgemm16avx2) {
    ABORT_IF(intgemm::kCPU < intgemm::CPUType::AVX2, "Your CPU doesn't support the architecture necessary to decode model of type {}. Try older architecture instead.", vtype);
  } else if (vtype == Type::intgemm8avx512 || vtype == Type::intgemm16avx512) {
    ABORT_IF(intgemm::kCPU < intgemm::CPUType::AVX512BW, "Your CPU doesn't support the architecture necessary to decode model of type {}. Try older architecture instead.", vtype);
  } else if (vtype == Type::intgemm8avx512vnni) {
    ABORT_IF(intgemm::kCPU < intgemm::CPUType::AVX512VNNI, "Your CPU doesn't support the architecture necessary to decode model of type {}. Try older architecture instead.", vtype);
  }
  return true;
#else
  vtype;
  ABORT("Using intgemm binary models is only supported when compiling marian with -DCOMPILE_CPU=ON.");
  return false;
#endif
}

template <Type vtype>
static inline float computeQuantMult(marian::Tensor val) {
#if COMPILE_CPU
  if(sizeOf(vtype) == 1)
    return 127.0f / intgemm::MaxAbsolute(val->data(), val->data() + val->shape().elements());
  else if(sizeOf(vtype) == 2)
    return 1024.0f;
  else
    ABORT("Unhandled type size {}", sizeOf(vtype));
#else
  val; 
  ABORT("Using intgemm binary models is only supported when compiling marian with -DCOMPILE_CPU=ON.");
#endif
}
#else // USE_INTGEMM

struct fakeGemm {
  struct Int8 {};
  struct Int16 {};
};

template<Type type> struct intgemm_;
template <> struct intgemm_<Type::intgemm8> {
  using width = fakeGemm::Int8;
  using type = int8_t;
};

template <> struct intgemm_<Type::intgemm8ssse3> {
  using width = fakeGemm::Int8;
  using type = int8_t;
};

template <> struct intgemm_<Type::intgemm8avx2> {
  using width = fakeGemm::Int8;
  using type = int8_t;
};

template <> struct intgemm_<Type::intgemm8avx512> {
  using width = fakeGemm::Int8;
  using type = int8_t;
};

template <> struct intgemm_<Type::intgemm8avx512vnni> {
  using width = fakeGemm::Int8;
  using type = int8_t;
};

template <> struct intgemm_<Type::intgemm16> {
  using width = fakeGemm::Int16;
  using type = int16_t;
};

template <> struct intgemm_<Type::intgemm16sse2> {
  using width = fakeGemm::Int16;
  using type = int16_t;
};

template <> struct intgemm_<Type::intgemm16avx2> {
  using width = fakeGemm::Int16;
  using type = int16_t;
};

template <> struct intgemm_<Type::intgemm16avx512> {
  using width = fakeGemm::Int16;
  using type = int16_t;
};

#endif // USE_INTGEMM

// For loading architecture agnostic models. We do PrepareAndTranpose, because we already transposed
// in our binary format. Then we copy the quantizationMultiplier information at the end
template<Type vtype>
void prepareAndTransposeB(io::Item& item, const char * input) {
#ifdef COMPILE_CPU
    typedef typename intgemm_<vtype>::type Integer;
    Integer * output_tensor = reinterpret_cast<Integer *>(&(*item.bytes.begin()));
#ifdef ARM
    // On arm we do RowM * ColM. Our input already comes transposed due to the way we prepare matrices in the binary
    // So on arm ALL we need to do is just copy. No need for pre
    std::memcpy(output_tensor, reinterpret_cast<const Integer *>(input), sizeof(Integer) * item.shape.elements());
#else
    // Sometimes we will end up with misaligned intput (and output) so we can't use them directly.
    // If this is the case, we will need to temporary allocate aligned memory, copy the results, and then free it
    if (reinterpret_cast<uintptr_t>(input) % 64 == 0 && reinterpret_cast<uintptr_t>(output_tensor) % 64 == 0) {
    #if defined(WASM)
        ABORT_IF(isIntgemm(vtype) && sizeof(vtype) == 2,
                "Int16::PrepareBQuantizedTransposed is not implemented for wasm.");
        int8PrepareBFromQuantizedTransposed(reinterpret_cast<const int8_t *>(input),
                                        (Index)rows(item.shape),  //Since we only transposed, but didn't update the shape when constructing the binary 
                                        (Index)cols(item.shape), //rows here returns the columns of the transposed input matrix, and cols -> the rows
                                        (int8_t *)output_tensor);
    #elif defined(USE_INTGEMM)
        intgemm_<vtype>::width::PrepareBQuantizedTransposed(reinterpret_cast<const Integer *>(input),
                                                   output_tensor,
                                                   rows(item.shape),  //Since we only transposed, but didn't update the shape when constructing the binary, 
                                                   cols(item.shape)); //rows here returns the columns of the transposed input matrix, and cols -> the rows
    #endif
    } else {
        Integer * aligned_input = reinterpret_cast<Integer *>(genericMalloc(512, rows(item.shape)*cols(item.shape)*sizeof(Integer)));
        std::copy(input, input + rows(item.shape)*cols(item.shape), aligned_input);
        Integer * aligned_output = reinterpret_cast<Integer *>(genericMalloc(512, rows(item.shape)*cols(item.shape)*sizeof(Integer)));
    #if defined(WASM)
        ABORT_IF(isIntgemm(vtype) && sizeof(vtype) == 2,
                "Int16::PrepareBQuantizedTransposed is not implemented for wasm.");
        int8PrepareBFromQuantizedTransposed(reinterpret_cast<const int8_t *>(aligned_input),
                                        (Index)rows(item.shape),  //Since we only transposed, but didn't update the shape when constructing the binary, 
                                        (Index)cols(item.shape), //rows here returns the columns of the transposed input matrix, and cols -> the rows
                                        reinterpret_cast<int8_t *>(aligned_output));
    #elif defined(USE_INTGEMM)
        intgemm_<vtype>::width::PrepareBQuantizedTransposed(reinterpret_cast<const Integer *>(aligned_input),
                                                   reinterpret_cast<Integer *>(aligned_output),
                                                   rows(item.shape),  //Since we only transposed, but didn't update the shape when constructing the binary, 
                                                   cols(item.shape)); //rows here returns the columns of the transposed input matrix, and cols -> the rows
    #endif
        // Copy to output tensor
        std::copy(aligned_output, aligned_output + rows(item.shape)*cols(item.shape), output_tensor);
        genericFree(aligned_input);
        genericFree(aligned_output);
    }
#endif
    //Copy the quantMult
    float quantMult = *(reinterpret_cast<const float *>(reinterpret_cast<const Integer *>(input) + item.shape.elements()));
    *(reinterpret_cast<float *>(&(*(output_tensor + item.shape.elements())))) = quantMult;
#else // COMPILE_CPU
    ABORT("Using intgemm models is supported only with -DCOMPILE_CPU=on");
#endif
}

template<Type vtype>
void unquantizeWemb(io::Item& item, const char * input) {
    typedef typename intgemm_<vtype>::type Integer;
    float quantMult = *(reinterpret_cast<const float *>(reinterpret_cast<const Integer *>(input) + item.shape.elements()));
    float * output_tensor = reinterpret_cast<float *>(&(*item.bytes.begin()));
    // Explicitly calculate n once beforehand because the compiler does not pick up on its
    // static nature, and will end up calling marian::Shape::dim() a lot.
    const size_t n = rows(item.shape) * cols(item.shape);
    for (size_t i = 0; i < n; i++) {
        output_tensor[i] = reinterpret_cast<const Integer *>(input)[i]*(1/quantMult);
    }
}

} //integer
} //cpu
} //marian
