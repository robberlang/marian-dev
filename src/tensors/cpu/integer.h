#pragma once

#include "3rd_party/intgemm/intgemm.h"
#include "common/hash.h"
#include "functional/approx.h"
#include "graph/node.h"

namespace marian {
namespace cpu {

class OnlyForInferenceNodeOp : public NaryNodeOp {
public:
  OnlyForInferenceNodeOp(const std::vector<Expr>& nodes,
                         Shape shape,
                         Type value_type = Type::float32)
      : NaryNodeOp(nodes, shape, value_type) {}

  OnlyForInferenceNodeOp(const std::vector<Expr>& nodes) : NaryNodeOp(nodes) {}

  NodeOps backwardOps() override {
    ABORT("Only used for inference");
    return {NodeOp()};
  }
};

namespace integer {

template <Type Type_>
using EnableIfTypeIsSupported = typename std::enable_if<
  std::integral_constant<bool,
    (Type_ == Type::int8) ||
    (Type_ == Type::int16)
  >::value>::type;

inline int cols(Tensor& tensor) { return tensor->shape()[-1]; }
inline int rows(Tensor& tensor) { return tensor->shape().elements() / cols(tensor); }

template <Type Type_> struct backend_s;
template <> struct backend_s<Type::int8> { using backend = intgemm::Int8; };
template <> struct backend_s<Type::int16> { using backend = intgemm::Int16; };
template <Type Type_> using backend = typename backend_s<Type_>::backend;

template <Type Type_, typename = EnableIfTypeIsSupported<Type_>>
class QuantMultNodeOp : public OnlyForInferenceNodeOp {
public:
  QuantMultNodeOp(Expr input) : OnlyForInferenceNodeOp({input}, Shape()) {
    ABORT_IF(child(0) == nullptr, "Input matrix cannot be null");
  }

  NodeOps forwardOps() override {
    return {NodeOp(
      auto input = child(0)->val();

      ABORT_IF(input->type() != Type::float32, "Trying to quantize non-float");

      if (Type_ == Type::int16) {
        *val_->data() = 1024.0f;
      } else {
        *val_->data() = 127.0f / intgemm::MaxAbsolute(input->data(), input->data() + input->shape().elements());
      }
    )};
  }

  const std::string type() override { return "intQuantMult"; }
};

template <Type Type_, typename PrepareMatrixFun>
inline NodeOps prepareMatrixForwardOps(Node* node, PrepareMatrixFun prepare_matrix_fun) {
  return {NodeOp(
    using Integer = typename backend<Type_>::Integer;

    auto input = node->child(0)->val();
    auto quant_mult = node->child(1)->val();
    prepare_matrix_fun(
        input->data(),
        node->val()->data<Integer>(),
        *quant_mult->data(),
        rows(input),
        input->shape()[-1]);
  )};
}

template <Type Type_, typename = EnableIfTypeIsSupported<Type_>>
class PrepareANodeOp : public OnlyForInferenceNodeOp {
public:
  PrepareANodeOp(Expr input, Expr quant_mult)
      : OnlyForInferenceNodeOp({input, quant_mult}, input->shape(), Type_) {
    ABORT_IF(child(0) == nullptr, "A cannot be null");
    ABORT_IF(child(1) == nullptr, "Quant mult of A cannot be null");
  }

  NodeOps forwardOps() override {
    return prepareMatrixForwardOps<Type_>(this, backend<Type_>::PrepareA);
  }

  const std::string type() override { return "intPrepareA"; }
};

template <Type Type_, typename = EnableIfTypeIsSupported<Type_>>
class PrepareBNodeOp : public OnlyForInferenceNodeOp {
public:
  PrepareBNodeOp(Expr input, Expr quant_mult)
      : OnlyForInferenceNodeOp({input, quant_mult}, input->shape(), Type_) {
    ABORT_IF(child(0) == nullptr, "B cannot be null");
    ABORT_IF(child(1) == nullptr, "Quant mult of B cannot be null");
  }

  NodeOps forwardOps() override {
    return prepareMatrixForwardOps<Type_>(this, backend<Type_>::PrepareB);
  }

  const std::string type() override { return "intPrepareB"; }
};

template <Type Type_, typename = EnableIfTypeIsSupported<Type_>>
class SelectColumnsBNodeOp : public OnlyForInferenceNodeOp {
public:
  SelectColumnsBNodeOp(Expr input, const std::vector<Word> &indices)
      : OnlyForInferenceNodeOp({input}, newShape(input, indices), Type_), indices_(indices) {
    ABORT_IF(child(0) == nullptr, "B cannot be null");

    // Check number of selected columns
    assert(indices.size() % 8 == 0);
  }

  NodeOps forwardOps() override {
    return {NodeOp(
      using Integer = typename backend<Type_>::Integer;

      auto input = child(0)->val();
      backend<Type_>::SelectColumnsB(
          (const Integer*)input->data(),
          val_->data<Integer>(),
          rows(input),
          &*indices_.begin(),
          &*indices_.end());
    )};
  }

  const std::string type() override { return "intSelectColumnsB"; }

  size_t hash() override {
    if (!hash_) {
      hash_ = NaryNodeOp::hash();
      for(auto i : indices_)
        util::hash_combine(hash_, i);
    }
    return hash_;
  }

  bool equal(Expr node) override {
    if(!NaryNodeOp::equal(node)) return false;
    Ptr<SelectColumnsBNodeOp> cnode = std::dynamic_pointer_cast<SelectColumnsBNodeOp>(node);
    if (!cnode) return false;
    return indices_ == cnode->indices_;
  }

private:
  static Shape newShape(Expr a, const std::vector<Word>& indices) {
    Shape ret = a->shape();
    ret.set(1, indices.size());
    return ret;
  }

  std::vector<Word> indices_;
};

template <Type Type_, typename = EnableIfTypeIsSupported<Type_>>
class DotNodeOp : public OnlyForInferenceNodeOp {
private:
  float scalar_;

public:
  DotNodeOp(Expr a, Expr b, float scalar)
      : OnlyForInferenceNodeOp({a, b}, newShape(a, b)), scalar_(scalar) {
    ABORT_IF(child(0) == nullptr, "A cannot be null");
    ABORT_IF(child(1) == nullptr, "B cannot be null");

    // Check alignment
    assert(child(1)->shape()[-1] % 8 == 0);

    // Check dimmensions
    ABORT_IF(child(0)->shape()[-1] != child(1)->shape()[-2], "Matrices cannot be multiplied because there's a dimension mismatch");
  }

  Shape newShape(Expr a, Expr b) {
    Shape result = a->shape();
    result.set(-1, b->shape()[-1]);
    return result;
  }

  NodeOps forwardOps() override {
    return {NodeOp(
      using Integer = typename backend<Type_>::Integer;
      using intgemm::callbacks::Write;

      auto a = child(0)->val();
      auto b = child(1)->val();
      backend<Type_>::Multiply(
          (const Integer*)a->data(),
          (const Integer*)b->data(),
          rows(a),
          cols(a), // Shared dimension.
          cols(b),
          Write<int>(val_->data<int>()));
    )};
  }

  const std::string type() override { return "intDot"; }
};

template <Type Type_, typename = EnableIfTypeIsSupported<Type_>>
class AffineNodeOp : public OnlyForInferenceNodeOp {
private:
  float scalar_;

public:
  AffineNodeOp(Expr a, Expr b, Expr bias, float scalar)
      : OnlyForInferenceNodeOp({a, b, bias}, newShape(a, b)), scalar_(scalar) {
    ABORT_IF(child(0) == nullptr, "A cannot be null");
    ABORT_IF(child(1) == nullptr, "B cannot be null");
    ABORT_IF(child(2) == nullptr, "Bias cannot be null");

    // Check alignment
    assert(child(1)->shape()[-1] % 8 == 0);

    // Check dimmensions
    ABORT_IF(child(0)->shape()[-1] != child(1)->shape()[-2], "Matrices cannot be multiplied because there's a dimension mismatch");
    ABORT_IF(child(1)->shape()[-1] != child(2)->shape()[-1], "Bias cannot be added because there's a dimension mismatch");
  }

  Shape newShape(Expr a, Expr b) {
    Shape result = a->shape();
    result.set(-1, b->shape()[-1]);
    return result;
  }

  NodeOps forwardOps() override {
    return {NodeOp(
      using Integer = typename backend<Type_>::Integer;
      using intgemm::callbacks::AddBiasAndWrite;

      auto a = child(0)->val();
      auto b = child(1)->val();
      auto bias = child(2)->val();
      backend<Type_>::Multiply(
          (const Integer*)a->data(),
          (const Integer*)b->data(),
          rows(a),
          cols(a), // Shared dimension.
          cols(b),
          AddBiasAndWrite(bias->data<int>(), val_->data<int>()));
    )};
  }

  const std::string type() override { return "intAffine"; }
};

template <Type Type_, typename = EnableIfTypeIsSupported<Type_>>
class ReLUNodeOp : public OnlyForInferenceNodeOp {
public:
  ReLUNodeOp(Expr input) : OnlyForInferenceNodeOp({input}) {
    ABORT_IF(child(0) == nullptr, "Input cannot be null");
  }

  NodeOps forwardOps() override {
    return {NodeOp(kernel(val_, child(0)->val()))};
  }

  const std::string type() override { return "intReLU"; }

private:
  static inline void kernel(marian::Tensor output, const marian::Tensor input) {
    using vec_t = __m256i;

    static const auto const_zero = _mm256_setzero_si256();

    auto input_it = input->data<vec_t>();
    auto output_it = output->data<vec_t>();

    const std::size_t length = output->shape().elements() / (sizeof(vec_t) / sizeOf(Type_));
    std::size_t i = 0;

    if (Type_ == Type::int8) {
      for (auto i = 0; i < length; ++i)
        *output_it++ = _mm256_max_epi8(*input_it++, const_zero);
    } else if (Type_ == Type::int16) {
      for (auto i = 0; i < length; ++i)
        *output_it++ = _mm256_max_epi16(*input_it++, const_zero);
    }

    i *= sizeof(vec_t) / sizeOf(Type_);
    if (Type_ == Type::int8) {
      for(; i < output->shape().elements(); ++i)
        output->data<int8_t>()[i] = std::max(input->data<int8_t>()[i], int8_t(0));
    } else if (Type_ == Type::int16) {
      for(; i < output->shape().elements(); ++i)
        output->data<int16_t>()[i] = std::max(input->data<int16_t>()[i], int16_t(0));
    }
  }
};

template <Type Type_, typename = EnableIfTypeIsSupported<Type_>>
class UnquantizeNodeOp : public OnlyForInferenceNodeOp {
public:
  UnquantizeNodeOp(Expr input, Expr unquant_mult) : OnlyForInferenceNodeOp({input, unquant_mult}) {
    ABORT_IF(child(0) == nullptr, "Input cannot be null");
    ABORT_IF(child(1) == nullptr, "UnquantMult cannot be null");
  }

  NodeOps forwardOps() override {
    return {NodeOp(kernel(val_, child(0)->val(), child(1)->val()))};
  }

  const std::string type() override { return "intUnquantize"; }

private:
  static inline void kernel(Tensor output, const Tensor input, const Tensor unquant_mult) {
    using vi = __m256i;
    using vf = __m256;

    auto input_it = input->data<vi>();
    auto vunquant_mult = intgemm::set1_ps<vf>(*unquant_mult->data());
    auto output_it = output->data<vf>();

    const std::size_t length = output->shape().elements() / (sizeof(vi) / sizeOf(Type_));
    std::size_t i = 0;

    if (Type_ == Type::int8) {
      for (; i < length; ++i) {
        auto upcasted = intgemm::kernels::upcast8to32(*input_it++);
        *output_it++ = intgemm::kernels::unquantize(upcasted.first, vunquant_mult);
        *output_it++ = intgemm::kernels::unquantize(upcasted.second, vunquant_mult);
        *output_it++ = intgemm::kernels::unquantize(upcasted.third, vunquant_mult);
        *output_it++ = intgemm::kernels::unquantize(upcasted.fourth, vunquant_mult);
      }
    } else if (Type_ == Type::int16) {
      for (; i < length; ++i) {
        auto upcasted = intgemm::kernels::upcast8to32(*input_it++);
        *output_it++ = intgemm::kernels::unquantize(upcasted.first, vunquant_mult);
        *output_it++ = intgemm::kernels::unquantize(upcasted.second, vunquant_mult);
      }
    }

    // TODO: tail?
  }
};

template <Type Type_, typename = EnableIfTypeIsSupported<Type_>>
class ElemwiseAddNodeOp : public OnlyForInferenceNodeOp {
public:
  ElemwiseAddNodeOp(Expr a, Expr b) : OnlyForInferenceNodeOp({a, b}) {
    ABORT_IF(child(0) == nullptr, "A cannot be null");
    ABORT_IF(child(1) == nullptr, "B cannot be null");
  }

  NodeOps forwardOps() override {
    return {NodeOp(kernel(val_, child(0)->val(), child(1)->val()))};
  }

  const std::string type() override { return "intElemwiseAdd"; }

private:
  static inline void kernel(Tensor output, const Tensor a, const Tensor b) {
    using vec_t = __m256i;

    auto a_it = a->data<vec_t>();
    auto b_it = a->data<vec_t>();
    auto output_it = output->data<vec_t>();

    const std::size_t length = output->shape().elements() / (sizeof(vec_t) / sizeOf(Type_));
    std::size_t i = 0;

    if (Type_ == Type::int8) {
      for (; i < length; ++i)
        *output_it++ = _mm256_add_epi8(*a_it++, *b_it++);
    } else if (Type_ == Type::int16) {
      for (; i < length; ++i)
        *output_it++ = _mm256_add_epi16(*a_it++, *b_it++);
    }

    i *= sizeof(vec_t) / sizeOf(Type_);
    if (Type_ == Type::int8) {
      for(; i < output->shape().elements(); ++i)
        output->data<int8_t>()[i] = a->data<int8_t>()[i] + b->data<int8_t>()[i];
    } else if (Type_ == Type::int16) {
      for(; i < output->shape().elements(); ++i)
        output->data<int16_t>()[i] = a->data<int16_t>()[i] + b->data<int16_t>()[i];
    }
  }
};

template <Type Type_, typename = EnableIfTypeIsSupported<Type_>>
class ElemwiseMulNodeOp : public OnlyForInferenceNodeOp {
private:
  unsigned right_shift_;

public:
  ElemwiseMulNodeOp(Expr a, Expr b, unsigned right_shift) : OnlyForInferenceNodeOp({a, b}), right_shift_(right_shift) {
    ABORT_IF(child(0) == nullptr, "A cannot be null");
    ABORT_IF(child(1) == nullptr, "B cannot be null");
  }

  NodeOps forwardOps() override {
    return {NodeOp(kernel(val_, child(0)->val(), child(1)->val(), right_shift_))};
  }

  const std::string type() override { return "intElemwiseMul"; }

private:
  static inline void kernel(Tensor output, const Tensor a, const Tensor b, unsigned right_shift) {
    using vec_t = __m256i;

    auto a_it = a->data<vec_t>();
    auto b_it = a->data<vec_t>();
    auto output_it = output->data<vec_t>();

    const std::size_t length = output->shape().elements() / (sizeof(vec_t) / sizeOf(Type_));
    std::size_t i = 0;

    if (Type_ == Type::int8) {
      for (; i < length; ++i)
        *output_it++ = intgemm::kernels::multiply_sat<int8_t>(*a_it++, *b_it++, right_shift);
    } else if (Type_ == Type::int16) {
      for (; i < length; ++i)
        *output_it++ = intgemm::kernels::multiply_sat<int16_t>(*a_it++, *b_it++, right_shift);
    }

    // TODO: tail?
  }
};

template <Type Type_, typename = EnableIfTypeIsSupported<Type_>>
class ElemwiseRescaleNodeOp : public OnlyForInferenceNodeOp {
public:
  ElemwiseRescaleNodeOp(Expr input, Expr scale) : OnlyForInferenceNodeOp({input, scale}) {
    ABORT_IF(child(0) == nullptr, "Input cannot be null");
    ABORT_IF(child(1) == nullptr, "Scale cannot be null");
  }

  NodeOps forwardOps() override {
    return {NodeOp(kernel(val_, child(0)->val(), child(1)->val()))};
  }

  const std::string type() override { return "intElemwiseRescale"; }

private:
  static inline void kernel(Tensor output, const Tensor input, const Tensor scale) {
    using vi = __m256i;
    using vf = __m256;

    auto input_it = input->data<vi>();
    auto scale_it = scale->data<vf>();
    auto output_it = output->data<vi>();

    const std::size_t length = output->shape().elements() / (sizeof(vi) / sizeOf(Type_));
    std::size_t i = 0;

    if (Type_ == Type::int8) {
      for (; i < length; ++i) {
        auto upcasted = intgemm::kernels::upcast8to32(*input_it++);
        *output_it++ = intgemm::kernels::downcast32to8(
          intgemm::kernels::rescale(upcasted.first, *scale_it),
          intgemm::kernels::rescale(upcasted.second, *scale_it),
          intgemm::kernels::rescale(upcasted.third, *scale_it),
          intgemm::kernels::rescale(upcasted.fourth, *scale_it));
        ++scale_it;
      }
    } else if (Type_ == Type::int16) {
      for (; i < length; ++i) {
        auto upcasted = intgemm::kernels::upcast16to32(*input_it++);
        *output_it++ = intgemm::kernels::downcast32to16(
          intgemm::kernels::rescale(upcasted.first, *scale_it),
          intgemm::kernels::rescale(upcasted.second, *scale_it));
        ++scale_it;
      }
    }

    i *= sizeof(vi) / sizeOf(Type_);
    if (Type_ == Type::int8) {
      for(; i < output->shape().elements(); ++i)
        output->data<int8_t>()[i] = std::min<int>(std::max<int>(std::numeric_limits<int8_t>::min(), input->data<int8_t>()[i] * scale->data()[i]), std::numeric_limits<int8_t>::max());
    } else if (Type_ == Type::int16) {
      for(; i < output->shape().elements(); ++i)
        output->data<int16_t>()[i] = std::min<int>(std::max<int>(std::numeric_limits<int16_t>::min(), input->data<int16_t>()[i] * scale->data()[i]), std::numeric_limits<int16_t>::max());
    }
  }
};

// int8_t is hardcoded for now
template <Type Type_, typename = EnableIfTypeIsSupported<Type_>>
class SSRUSigmoidFNodeOp : public OnlyForInferenceNodeOp {
public:
  SSRUSigmoidFNodeOp(Expr input, Expr Wf, Expr bf, Expr scale, Expr sigmoid_lut) : OnlyForInferenceNodeOp({input, Wf, bf, scale, sigmoid_lut}) {
    ABORT_IF(child(0) == nullptr, "Input cannot be null");
    ABORT_IF(child(1) == nullptr, "Wf cannot be null");
    ABORT_IF(child(2) == nullptr, "bf cannot be null");
    ABORT_IF(child(3) == nullptr, "scale cannot be null");
    ABORT_IF(child(4) == nullptr, "Sigmoid LUT cannot be null");
  }

  NodeOps forwardOps() override {
    return {NodeOp(
      using Integer = typename backend<Type_>::Integer;
      using intgemm::callbacks::SSRUSigmoidF;

      auto input = child(0)->val();
      auto Wf = child(1)->val();
      auto bf = child(2)->val();
      auto scale = *child(3)->val()->data();
      auto sigmoid_lut = child(4)->val();
      backend<Type_>::Multiply(
          (const Integer*)input->data(),
          (const Integer*)Wf->data(),
          rows(input),
          cols(input), // Shared dimension.
          cols(Wf),
          SSRUSigmoidF<int8_t>(bf->data<int8_t>(), scale, sigmoid_lut->data<int8_t>(), val_->data<int8_t>()));
    )};
  }

  const std::string type() override { return "intSSRUSigmoidF"; }
};

// int8_t is hardcoded for now
template <Type Type_, typename = EnableIfTypeIsSupported<Type_>>
class SSRUPrecomputedPartOfHighwayNodeOp : public OnlyForInferenceNodeOp {
public:
  SSRUPrecomputedPartOfHighwayNodeOp(Expr input, Expr W, Expr sigmoid_f, Expr scale) : OnlyForInferenceNodeOp({input, W, sigmoid_f, scale}) {
    ABORT_IF(child(0) == nullptr, "Input cannot be null");
    ABORT_IF(child(1) == nullptr, "W cannot be null");
    ABORT_IF(child(2) == nullptr, "SigmoidF cannot be null");
    ABORT_IF(child(3) == nullptr, "Scale cannot be null");
  }

  NodeOps forwardOps() override {
    return {NodeOp(
      using Integer = typename backend<Type_>::Integer;
      using intgemm::callbacks::SSRUPrecomputedPartOfHighway;

      auto input = child(0)->val();
      auto W = child(1)->val();
      auto sigmoid_f = child(2)->val();
      auto scale = *child(3)->val()->data();
      backend<Type_>::Multiply(
          (const Integer*)input->data(),
          (const Integer*)W->data(),
          rows(input),
          cols(input), // Shared dimension.
          cols(W),
          SSRUPrecomputedPartOfHighway<int8_t>(sigmoid_f->data<int8_t>(), scale, val_->data<int8_t>()));
    )};
  }

  const std::string type() override { return "intSSRUPrecomputedPartOfHighway"; }
};

template <Type Type_, typename = EnableIfTypeIsSupported<Type_>>
struct ops {
  static inline Expr quantMult(Expr a) {
    return Expression<QuantMultNodeOp<Type_>>(a);
  }
  static inline Expr prepareA(Expr a, Expr quant_mult) {
    return Expression<PrepareANodeOp<Type_>>(a, quant_mult);
  }
  static inline Expr prepareB(Expr b, Expr quant_mult) {
    return Expression<PrepareBNodeOp<Type_>>(b, quant_mult);
  }
  static inline Expr selectColumnsB(Expr b, const std::vector<Word> &cols) {
    return Expression<SelectColumnsBNodeOp<Type_>>(b, cols);
  }
  static inline Expr dot(Expr a, Expr b, float scalar = 1.0f) {
    return Expression<DotNodeOp<Type_>>(a, b, scalar);
  }
  static inline Expr affine(Expr a, Expr b, Expr bias, float scalar = 1.0f) {
    return Expression<AffineNodeOp<Type_>>(a, b, bias, scalar);
  }
  static inline Expr relu(Expr input) {
    return Expression<ReLUNodeOp<Type_>>(input);
  }
  static inline Expr unquantize(Expr input, Expr unquant_mult) {
    return Expression<UnquantizeNodeOp<Type_>>(input, unquant_mult);
  }
  static inline Expr elemwiseAdd(Expr a, Expr b) {
    return Expression<ElemwiseAddNodeOp<Type_>>(a, b);
  }
  static inline Expr elemwiseMul(Expr a, Expr b, int scale = 1) {
    return Expression<ElemwiseMulNodeOp<Type_>>(a, b, scale);
  }
  static inline Expr elemwiseRescale(Expr input, Expr scale) {
    return Expression<ElemwiseRescaleNodeOp<Type_>>(input, scale);
  }
  static inline Expr SSRUSigmoidF(Expr input, Expr Wf, Expr bf, Expr scale, Expr sigmoid_lut) {
    return Expression<SSRUSigmoidFNodeOp<Type_>>(input, Wf, bf, scale, sigmoid_lut);
  }
  static inline Expr SSRUPrecomputedPartOfHighway(Expr input, Expr W, Expr sigmoid_f, Expr scale) {
    return Expression<SSRUPrecomputedPartOfHighwayNodeOp<Type_>>(input, W, sigmoid_f, scale);
  }
};

} // namespace integer

using int8 = integer::ops<Type::int8>;
using int16 = integer::ops<Type::int16>;

}
}
