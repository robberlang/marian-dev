#pragma once

#include "graph/node.h"
#include "graph/node_operators_unary.h"
#include "integer_common.h"

namespace marian {

namespace cpu {
namespace integer {

template<Type vtype>
struct PrepareANodeOp : public NaryNodeOp {
float clipValue_;
float quantMult_;
bool shifted_;
  PrepareANodeOp(Expr input, Expr quant_mult, float clipValue, bool shifted)
      : NaryNodeOp({input, quant_mult}, input->shape(), vtype), clipValue_(clipValue), shifted_(shifted) {

    if (!shifted) {
      set_name(input->name());
    } else {
      set_name(input->name() + "_shifted");
    }
    setMemoize(false);
    // Check if arguments are not null
    ABORT_IF(child(0) == nullptr, "A cannot be null");
    ABORT_IF(child(1) == nullptr, "Quant mult of A cannot be null");
  }

  NodeOps forwardOps() override {
#ifdef COMPILE_CPU
    return {NodeOp(
      quantMult_ = *child(1)->val()->data();
      typedef typename intgemm_<vtype>::type Integer;
      if (!shifted_) {
        intgemm_<vtype>::width::PrepareA(child(0)->val()->data(), /*input*/
                                      val_->data<Integer>(), /*output*/
                                      *child(1)->val()->data(), /*Quant Mult*/
                                      rows(child(0)->val()),
                                      cols(child(0)->val()));
      } else {
        intgemm::Int8Shift::PrepareA(child(0)->val()->data(), /*input*/
                                      val_->data<int8_t>(), /*output*/
                                      *child(1)->val()->data(), /*Quant Mult*/
                                      rows(child(0)->val()),
                                      cols(child(0)->val()));
      }
    )};
#else
    return {NodeOp()};
#endif
  }

  NodeOps backwardOps() override {
    ABORT("Only used for inference");
    return {NodeOp(0)};
  }

  const std::string type() override { return "intgemmPrepareA"; }
};

template<Type vtype>
struct PrepareBNodeOp : public NaryNodeOp {
float clipValue_;
float quantMult_;

  PrepareBNodeOp(Expr input, Expr quant_mult, float clipValue)
      : NaryNodeOp({input, quant_mult}, input->shape(), intgemm_<vtype>::intgemmType), clipValue_(clipValue) {

    set_name(input->name());
    // Check if arguments are not null
    ABORT_IF(child(0) == nullptr, "A cannot be null");
    ABORT_IF(child(1) == nullptr, "Quant mult of B cannot be null");
    ABORT_IF(input->shape()[-1] %8 != 0, "Columns of matrix: " + input->type() + " must be multiple of 8.");
  }

  NodeOps forwardOps() override {
#ifdef COMPILE_CPU
   return {NodeOp(
      quantMult_ = *child(1)->val()->data();
      typedef typename intgemm_<vtype>::type Integer;
      if (isIntgemm(child(0)->value_type())) {
        val_ = child(0)->val();
      } else {
        intgemm_<vtype>::width::PrepareB(child(0)->val()->data(), /*input*/
                                      val_->data<Integer>(), /*output*/
                                      *child(1)->val()->data(), /*Quant Mult*/
                                      rows(child(0)->val()),
                                      cols(child(0)->val()));
      }
    )};
#else
   return {NodeOp()};
#endif
  }

  NodeOps backwardOps() override {
    ABORT("Only used for inference");
    return {NodeOp(0)};
  }

  const std::string type() override { return "intgemmPrepareB"; }
};

template<Type vtype>
struct SelectColumnsBNodeOp : public UnaryNodeOp {
public:
  float clipValue_;
  float quantMult_;
  SelectColumnsBNodeOp(Expr input, const std::vector<uint_least32_t>  &indices, float clipValue)
      : UnaryNodeOp(input, newShape(input, indices), intgemm_<vtype>::intgemmType), clipValue_(clipValue), indices_(indices) {

    set_name(input->name());
    setMemoize(false); // Enabling memoization leads to a massive memory leak. Instead use special "midterm" memory.
                       // Still, I don't understand why setMemoize(true) still leaks.
    // Check if arguments are not null
    ABORT_IF(child(0) == nullptr, "B cannot be null");

    // Check number of selected columns
    ABORT_IF(indices.size() % 8 != 0, "Shortlist selected vocabulary must be a multiple of 8.");
  }

  NodeOps forwardOps() override {
#ifdef COMPILE_CPU
    return {NodeOp(
      //We get the quantization multiplier from a PrepareB or directly from the input
      if (child(0)->type() == "intgemmPrepareB") {
        auto bPreppedNode = std::static_pointer_cast<PrepareBNodeOp<vtype> >(child(0));
        quantMult_ = bPreppedNode->quantMult_;
      } else {
        typedef typename intgemm_<vtype>::type Integer;
        quantMult_ = *(reinterpret_cast<float *>(reinterpret_cast<Integer *>(child(0)->val()->data()) + child(0)->val()->shape().elements()));
      }
      auto input = child(0)->val();
      typedef typename intgemm_<vtype>::type Integer;
      intgemm_<vtype>::width::SelectColumnsB(
                    reinterpret_cast<Integer *>(input->data()),
                    val_->data<Integer>(),
                    rows(input),
                    &*indices_.begin(),
                    &*indices_.end());
    )};
#else
    return {NodeOp()};
#endif
  }

  const std::string type() override { return "intgemmSelectColumnsB"; }

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
    auto cnode = std::dynamic_pointer_cast<SelectColumnsBNodeOp<vtype>>(node);
    if (!cnode) return false;
    return indices_ == cnode->indices_;
  }

private:
  static Shape newShape(Expr a, const std::vector<uint_least32_t>& indices) {
    Shape ret = a->shape();
    ret.set(1, indices.size());
    return ret;
  }

  std::vector<uint_least32_t> indices_;
};

template<Type vtype>
struct QuantMultNodeOp : public UnaryNodeOp {
  bool isA_;
  QuantMultNodeOp(Expr input, bool isA, std::string& bname) : UnaryNodeOp(input, Shape({1}), Type::float32), isA_(isA) {
    if (isA_) {
      setMemoize(false);
      set_name(bname + "_QuantMultA");
    } else {
      set_name(input->name() + "_QuantMultB");
    }
  }
#pragma warning(push)
#pragma warning(disable: 4127) //VSCODE thinks line 222 is constant conditional expression, which it is only after the template resolution, not before.
  NodeOps forwardOps() override {
#ifdef COMPILE_CPU
    return {NodeOp(
      if (vtype == Type::int16) {
        *val_->data() = 1024.0f;
      } else if (child(0)->type() == "intgemmSelectColumnsB") {
        *val_->data() = std::static_pointer_cast<SelectColumnsBNodeOp<vtype> >(child(0))->quantMult_;
      } else if (isIntgemm(child(0)->value_type())) {                    /* So we can template*/
        typedef typename intgemm_<vtype>::type Integer;
        *val_->data() = *(reinterpret_cast<float *>(reinterpret_cast<Integer *>(child(0)->val()->data()) + child(0)->val()->shape().elements()));
      } else {
        if (child(0)->graph()->getBackend()->DumpQuantMult()) {
          intgemm::MeanStd meanstd = intgemm::GetVectorMeanStd(child(0)->val()->data(), child(0)->val()->data() + child(0)->val()->shape().elements(), true);
          intgemm::MeanStd meanstd2 = intgemm::GetVectorMeanStd(child(0)->val()->data(), child(0)->val()->data() + child(0)->val()->shape().elements());
          std::cerr << "Name: " << name() << " MeanAbs: " << meanstd.mean << " stddevAbs: " << meanstd.stddev << " Mean: " << meanstd2.mean << " stddev: "
          << meanstd2.stddev << " MaxAbs: " << intgemm::MaxAbsolute(child(0)->val()->data(), child(0)->val()->data() + child(0)->val()->shape().elements()) << std::endl;
        }
        *val_->data() = 127.0f / intgemm::MaxAbsolute(child(0)->val()->data(), child(0)->val()->data() + child(0)->val()->shape().elements());
      }
    )};
#else
    return {NodeOp()};
#endif
  }
#pragma warning(pop)
  NodeOps backwardOps() override {
    ABORT("Only used for inference");
    return {NodeOp(0)};
  }

  const std::string type() override {
    if (isA_)
      return "intgemmQuantMultA";
    else
      return "intgemmQuantMultB";
  }

  /* @TODO This is not correct in the case of none-static alphas but we are leaving it for now to battle memory leaks. */
  bool equal(Expr node) override {
    if (isA_) {
      return UnaryNodeOp::equal(node);
    }
    if(hash() == node->hash()) return true;
    return false;
  }

  size_t hash() override {
    return std::hash<std::string>{}(name());
  }

};

class PrepareBiasForBNodeOp : public NaryNodeOp {
  bool alreadyPrepared_ = false;
public:
  PrepareBiasForBNodeOp(Expr bias, Expr inputB_preppd, Expr a_quant_mult, Expr b_quant_mult)
      : NaryNodeOp({bias, inputB_preppd, a_quant_mult, b_quant_mult}, bias->shape(), Type::float32) {

    set_name(bias->name() + "_Prepared");
    if (bias->type() == "cols" && bias->graph()->getBackend()->isPrecomputedAlpha()) {
      ABORT("We shouldn't ever be here");
      alreadyPrepared_ = true;
    } else if (!bias->graph()->getBackend()->isPrecomputedAlpha()){
      setMemoize(false);
    }
  }

  NodeOps forwardOps() override {
    //std::cerr << "TrueBias: " << child(0)->name() << " type: " << child(0)->type() << " bQuantMult: " << this->child(3)->val()->data()[0] <<  " aQuantMult: " << this->child(2)->val()->data()[0] << std::endl;
    //std::cerr << "Bias name and val: " << child(0)->name() << " " << child(0)->val()->data()[0] << std::endl;
#ifdef COMPILE_CPU
    return {NodeOp(
      if (alreadyPrepared_) {
        //God Knows why trying to assign the bias tensor to this node causes a crash, the second time it's referenced
        //even though it's supposed to work fine. We use a memory copy instead.
        ABORT("We shouldn't ever be here.");
        std::memcpy(val_->data(), child(0)->val()->data(), child(0)->shape()[-1]*sizeof(float));
        //val_ = child(0)->val();
      } else {
        auto bias = this->child(0)->val();
        auto b = this->child(1)->val();
        auto quant_mult_a = this->child(2)->val();
        auto quant_mult_b = this->child(3)->val();

        float unquant_mult = (-1)*((127.0f / *quant_mult_a->data())*(127.0f / *quant_mult_b->data()))/(127.0f); //Minus one to invert add_ps later on
        intgemm::Int8Shift::PrepareBias((const int8_t *)b->data(), rows(b), cols(b), intgemm::callbacks::UnquantizeAndAddBiasAndWrite(unquant_mult, bias->data(), val_->data()));
      }
      )};
#else
    return {NodeOps()};
#endif
  }

  const std::string type() override { return "prepareBias"; }
};

class PrepareFakeBiasForBNodeOp : public NaryNodeOp {
public:
  PrepareFakeBiasForBNodeOp(Expr inputB_preppd, Expr a_quant_mult, Expr b_quant_mult)
      : NaryNodeOp({inputB_preppd, a_quant_mult, b_quant_mult}, {1, inputB_preppd->shape()[-1]}, Type::float32) {

    set_name(inputB_preppd->name() + "_FakeBias");
    if (!inputB_preppd->graph()->getBackend()->isPrecomputedAlpha()) {
      setMemoize(false);
    }
  }

  NodeOps forwardOps() override {
    //std::cerr << "FakeBias: " << child(0)->name() << " bQuantMult: " << this->child(2)->val()->data()[0] << " aQuantMult: " << this->child(1)->val()->data()[0] << std::endl;
#ifdef COMPILE_CPU
    return {NodeOp(
    auto b = this->child(0)->val();
    auto quant_mult_a = this->child(1)->val();
    auto quant_mult_b = this->child(2)->val();

    float unquant_mult = (-1)*((127.0f / *quant_mult_a->data())*(127.0f / *quant_mult_b->data()))/(127.0f); //Minus one to invert add_ps later on
    intgemm::Int8Shift::PrepareBias((const int8_t *)b->data(), rows(b), cols(b), intgemm::callbacks::UnquantizeAndWrite(unquant_mult, val_->data()));
    )};
#else
    return {NodeOp()};
#endif
  }

  const std::string type() override { return "prepareFakeBias"; }
};

template<Type vtype>
class DotNodeOp : public NaryNodeOp {
private:
float scalar_;

public:
  DotNodeOp(Expr a, Expr b, float scalar)
      : NaryNodeOp({a, b}, newShape(a, b), Type::float32), scalar_(scalar) {
        setMemoize(false); // AFAIK dot is never called with the same matrices
      }

  Shape newShape(Expr a, Expr b) {
    Shape result = a->shape();
    result.set(-1, b->shape()[-1]);
    return result;
  }

  NodeOps forwardOps() override {
#ifdef COMPILE_CPU
    return {NodeOp(
          float aQuantMult = std::static_pointer_cast<PrepareANodeOp<vtype> >(child(0))->quantMult_;
          float bQuantMult;
          if (child(1)->type() == "intgemmSelectColumnsB") {
            bQuantMult = std::static_pointer_cast<SelectColumnsBNodeOp<vtype> >(child(1))->quantMult_;
          } else if (child(1)->type() == "intgemmPrepareB") {
            bQuantMult = std::static_pointer_cast<PrepareBNodeOp<vtype> >(child(1))->quantMult_;
          } else {
            typedef typename intgemm_<vtype>::type Integer;
            bQuantMult = *(reinterpret_cast<float *>(reinterpret_cast<Integer *>(child(1)->val()->data()) + child(1)->val()->shape().elements()));
          }
          float unquant_mult = 1.0f/(aQuantMult*bQuantMult);

          unquant_mult = unquant_mult*scalar_;
          typedef typename intgemm_<vtype>::type Integer;
          intgemm_<vtype>::width::Multiply(reinterpret_cast<Integer *>(child(0)->val()->data()), /*A*/
                                           reinterpret_cast<Integer *>(child(1)->val()->data()), /*B*/
                                           rows(child(0)->val()),
                                           cols(child(0)->val()),
                                           cols(child(1)->val()),
                                           intgemm::callbacks::UnquantizeAndWrite(unquant_mult, val_->data()));
    )};
#else
    return {NodeOp()};
#endif
  }

  NodeOps backwardOps() override {
    ABORT("Only used for inference");
    return {NodeOp(0)};
  }

  const std::string type() override { return "intgemmDot"; }
};

template<Type vtype>
class AffineNodeOp : public NaryNodeOp {
private:
  float scalar_;
  bool shifted_;

public:
  AffineNodeOp(Expr a, Expr b, Expr Bias, float scalar, bool shifted=false)
      : NaryNodeOp({a, b, Bias}, newShape(a, b), Type::float32), scalar_(scalar), shifted_(shifted) {
        setMemoize(false); // AFAIK affine is never called with the same matrices
      }

  Shape newShape(Expr a, Expr b) {
    Shape result = a->shape();
    result.set(-1, b->shape()[-1]);
    return result;
  }

  NodeOps forwardOps() override {
#ifdef COMPILE_CPU
    return {NodeOp(
          float aQuantMult = std::static_pointer_cast<PrepareANodeOp<vtype> >(child(0))->quantMult_;
          float bQuantMult;
          if (child(1)->type() == "intgemmSelectColumnsB") {
            bQuantMult = std::static_pointer_cast<SelectColumnsBNodeOp<vtype> >(child(1))->quantMult_;
          } else if (child(1)->type() == "intgemmPrepareB") {
            bQuantMult = std::static_pointer_cast<PrepareBNodeOp<vtype> >(child(1))->quantMult_;
          } else {
            typedef typename intgemm_<vtype>::type Integer;
            bQuantMult = *(reinterpret_cast<float *>(reinterpret_cast<Integer *>(child(1)->val()->data()) + child(1)->val()->shape().elements()));
          }
          float unquant_mult = 1.0f/(aQuantMult*bQuantMult);

          unquant_mult = unquant_mult*scalar_;
          typedef typename intgemm_<vtype>::type Integer;
          if (!shifted_) {
            intgemm_<vtype>::width::Multiply(reinterpret_cast<Integer *>(child(0)->val()->data()), /*A*/
                                             reinterpret_cast<Integer *>(child(1)->val()->data()), /*B*/
                                             rows(child(0)->val()),
                                             cols(child(0)->val()),
                                             cols(child(1)->val()),                                          /*child(2) is bias*/
                                             intgemm::callbacks::UnquantizeAndAddBiasAndWrite(unquant_mult, child(2)->val()->data(), val_->data()));
          } else {
            intgemm::Int8Shift::Multiply(reinterpret_cast<int8_t *>(child(0)->val()->data()), /*A*/
                                  reinterpret_cast<int8_t *>(child(1)->val()->data()), /*B*/
                                  rows(child(0)->val()),
                                  cols(child(0)->val()),
                                  cols(child(1)->val()),                                          /*child(2) is bias*/
                                  intgemm::callbacks::UnquantizeAndAddBiasAndWrite(unquant_mult, child(2)->val()->data(), val_->data()));
          }
    )};
#else
    return {NodeOp()};
#endif
  }

  NodeOps backwardOps() override {
    ABORT("Only used for inference");
    return {NodeOp(0)};
  }

  const std::string type() override { return "intgemmAffine"; }
};

template<Type vtype>
static inline Expr quantMult(Expr a, bool isA=false, std::string Bname="") {
  return Expression<QuantMultNodeOp<vtype> >(a, isA, Bname);
}

template<Type vtype>
static inline Expr prepareA(Expr a, Expr quantMult, float clipValue, bool shifted=false) {
  return Expression<PrepareANodeOp<vtype> >(a, quantMult, clipValue, shifted);
}

template<Type vtype>
static inline Expr prepareB(Expr b, Expr quantMult, float clipValue) {
  return Expression<PrepareBNodeOp<vtype> >(b, quantMult, clipValue);
}

template<Type vtype>
static inline Expr selectColumnsB(Expr b, const std::vector<uint_least32_t> &cols, float clipValue) {
  return Expression<SelectColumnsBNodeOp<vtype > >(b, cols, clipValue);
}

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

template<Type vtype>
static inline Expr affine(Expr a, Expr b, Expr bias, bool transA, bool transB, float scale, float /* clipValue currently unused */ = 0.0f, bool shiftedBias=false) {
  Type bElementType = b->value_type();
  Expr aQuantMult = nullptr;
  static bool precomputedAlphas = b->graph()->getBackend()->isPrecomputedAlpha();
  if (precomputedAlphas) { //Shifting here maybe should check?
    aQuantMult = Expression<fetchAlphaFromModelNodeOp>(b);
  } else {
    aQuantMult = quantMult<vtype>(a, true, b->name()); /*@TODO Do something about b->name() here*/
  }
  auto aQuant = prepareA<vtype>(transA ? transpose(a) : a, aQuantMult, scale, shiftedBias);
  Expr bQuantMult = quantMult<vtype>(b);
  Expr bQuant = nullptr;
  if (isIntgemm(bElementType)) {
    //This is the case where we already run SelectColumnB or we loaded a prepacked model.
    //We ignore a transpose argument here, because we do not support it.
    ABORT_IF(transB, "Transpose on prepareB not currently supported");
    bQuant = b;
  } else {
    bQuant = prepareB<vtype>(transB ? transpose(b) : b, bQuantMult, scale);
  }
  if (bias && precomputedAlphas && bias->name() == "none") {
    // This is the case of the preprocessed bias. It's hacky but otherwise node caching is broken.
    // The bias node is the shortlisted bias and it has been prepared before index_select was run
  } else if (shiftedBias && bias) {
    bias = Expression<PrepareBiasForBNodeOp>(bias, bQuant, aQuantMult, bQuantMult);
  } else if (shiftedBias) {
    bias = Expression<PrepareFakeBiasForBNodeOp>(bQuant, aQuantMult, bQuantMult);
  }

  if (bias) {
    return Expression<AffineNodeOp<vtype> >(aQuant, bQuant, bias, scale, shiftedBias);
  } else {
    return Expression<DotNodeOp<vtype> >(aQuant, bQuant, scale);
  }
}

template<Type vtype>
static inline Expr dot(Expr a, Expr b, bool transA, bool transB, float scale, bool shiftedBias=false) {
  return affine<vtype>(a, b, nullptr, transA, transB, scale, 0 /*currently unused clipValue*/, shiftedBias);
}

}  // namespace integer
}  // namespace cpu
}  // namespace marian
