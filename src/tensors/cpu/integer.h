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

namespace { // anonymous namespace

inline int cols(Tensor& tensor) { return tensor->shape()[-1]; }
inline int rows(Tensor& tensor) { return tensor->shape().elements() / cols(tensor); }

template <Type Type_> struct backend_s;
template <> struct backend_s<Type::int8> { using backend = intgemm::Int8; };
template <> struct backend_s<Type::int16> { using backend = intgemm::Int16; };
template <Type Type_> using backend = typename backend_s<Type_>::backend;

} // anonymous namespace

template <Type Type_, typename = EnableIfTypeIsSupported<Type_>>
class QuantMultNodeOp : public OnlyForInferenceNodeOp {
public:
  QuantMultNodeOp(Expr input) : OnlyForInferenceNodeOp({input}, Shape()) {
    ABORT_IF(children().size() != 1, "expected 1 child");

    // Check if arguments are not null
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

namespace { // anonymous namespace

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

} // anonymous namespace

template <Type Type_, typename = EnableIfTypeIsSupported<Type_>>
class PrepareANodeOp : public OnlyForInferenceNodeOp {
public:
  PrepareANodeOp(Expr input, Expr quant_mult, float clipValue)
      : OnlyForInferenceNodeOp({input, quant_mult}, input->shape(), Type_) {
    ABORT_IF(children().size() != 2, "expected 2 children");

    // Check if arguments are not null
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
  PrepareBNodeOp(Expr input, Expr quant_mult, float clipValue)
      : OnlyForInferenceNodeOp({input, quant_mult}, input->shape(), Type_) {
    ABORT_IF(children().size() != 2, "expected 2 children");

    // Check if arguments are not null
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
    ABORT_IF(children().size() != 1, "expected 1 child");

    // Check if arguments are not null
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
    ABORT_IF(children().size() != 2, "expected 2 children");

    // Check if arguments are not null
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
    ABORT_IF(children().size() != 3, "expected 3 children");

    // Check if arguments are not null
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
  ReLUNodeOp(Expr input)
      : OnlyForInferenceNodeOp({input}) {
    ABORT_IF(children().size() != 1, "expected 1 children");
    ABORT_IF(child(0) == nullptr, "Input cannot be null");
  }

  NodeOps forwardOps() override {
    return {NodeOp(relu(val_, child(0)->val()))};
  }

  const std::string type() override { return "intReLU"; }

private:
  static void relu(marian::Tensor output, const marian::Tensor input) {
    static const auto const_zero = _mm256_setzero_si256();

    auto input_it = input->data<__m256i>();
    auto output_it = output->data<__m256i>();
    auto lenght = input->shape().elements() / sizeof(__m256i) * sizeOf(Type_);

    if (Type_ == Type::int8) {
      for (auto i = 0; i < lenght; ++i)
        *output_it++ = _mm256_max_epi8(*input_it++, const_zero);
    } else if (Type_ == Type::int16) {
      for (auto i = 0; i < lenght; ++i)
        *output_it++ = _mm256_max_epi16(*input_it++, const_zero);
    }
  }
};

template <Type Type_, typename = EnableIfTypeIsSupported<Type_>>
class UnquantizeNodeOp : public OnlyForInferenceNodeOp {
public:
  UnquantizeNodeOp(Expr input, Expr unquant_mult) : OnlyForInferenceNodeOp({input, unquant_mult}) {
    ABORT_IF(children().size() != 2, "expected 2 children");
    ABORT_IF(child(0) == nullptr, "Input cannot be null");
    ABORT_IF(child(1) == nullptr, "UnquantMult cannot be null");
  }

  NodeOps forwardOps() override {
    return {NodeOp(kernel(val_, child(0)->val(), child(1)->val()))};
  }

  const std::string type() override { return "intUnquantize"; }

private:
  static inline void kernel(Tensor output, const Tensor input, const Tensor unquant_mult) {
    // TODO: Vectorize it.
    if (Type_ == Type::int8) {
      for (auto i = 0; i < output->shape().elements(); ++i) {
        output->data()[i] = input->data<int8_t>()[i] * *unquant_mult->data();
      }
    } else if (Type_ == Type::int16) {
      for (auto i = 0; i < output->shape().elements(); ++i) {
        output->data()[i] = input->data<int16_t>()[i] * *unquant_mult->data();
      }
    }
  }
};

template <Type Type_, typename = EnableIfTypeIsSupported<Type_>>
struct ops {
  static inline Expr quantMult(Expr a) {
    return Expression<QuantMultNodeOp<Type_>>(a);
  }
  static inline Expr prepareA(Expr a, Expr quant_mult, float clipValue) {
    return Expression<PrepareANodeOp<Type_>>(a, quant_mult, clipValue);
  }
  static inline Expr prepareB(Expr b, Expr quant_mult, float clipValue) {
    return Expression<PrepareBNodeOp<Type_>>(b, quant_mult, clipValue);
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
};

} // namespace integer

using int8 = integer::ops<Type::int8>;
using int16 = integer::ops<Type::int16>;

namespace { // anonymous namespace

class HighwayNodeOp : public OnlyForInferenceNodeOp {
public:
  HighwayNodeOp(Expr y, Expr x, Expr t) : OnlyForInferenceNodeOp({y, x, t}) {
    ABORT_IF(children().size() != 3, "expected 3 children");
    ABORT_IF(child(0) == nullptr, "Y cannot be null");
    ABORT_IF(child(1) == nullptr, "X cannot be null");
    ABORT_IF(child(2) == nullptr, "T cannot be null");
  }

  NodeOps forwardOps() override {
    return {NodeOp(highway(val_, child(0)->val(), child(1)->val(), child(2)->val()))};
  }

  const std::string type() override { return "floatHighway"; }

private:
  static inline float stableSigmoid(float x) {
    if(x >= 0) {
      float z = expf(-x);
      return 1.0f / (1.0f + z);
    }
    else {
      float z = expf(x);
      return z / (1.0f + z);
    }
  }

  static void highway(marian::Tensor out, const marian::Tensor y, const marian::Tensor x, const marian::Tensor t) {
    using vec_t = __m256;

    static functional::Approx<10, 0, 100> approxSigmoid(stableSigmoid);
    static const auto const_one = intgemm::set1_ps<vec_t>(1.f);
    static const size_t ITEMS = sizeof(vec_t) / 4;

    auto out_ptr = reinterpret_cast<vec_t*>(out->data());
    auto y_ptr = reinterpret_cast<const vec_t*>(y->data());
    auto x_ptr = reinterpret_cast<const vec_t*>(x->data());

    const size_t length = out->shape().elements() / sizeof(vec_t) * 4; // 4 bytes per float

    vec_t sigma = intgemm::setzero_ps<vec_t>();
    float* sigma_raw = reinterpret_cast<float*>(&sigma);
    for(size_t i = 0; i < length; ++i) {
      for (size_t j = 0; j < ITEMS; ++j)
        sigma_raw[j] = approxSigmoid(t->data()[i * ITEMS + j]);

      auto sigma_c = _mm256_sub_ps(const_one, sigma);
      out_ptr[i] = intgemm::add_ps(intgemm::mul_ps(sigma, y_ptr[i]), intgemm::mul_ps(sigma_c, x_ptr[i]));
    }
  }
};

} // anonymous namespace

struct float32 {
  static inline Expr highway(Expr y, Expr x, Expr t) {
    return Expression<HighwayNodeOp>(y, x, t);
  }
};

}
}
