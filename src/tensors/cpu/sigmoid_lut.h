#pragma once

#include "3rd_party/intgemm/aligned.h"
#include "3rd_party/intgemm/intrinsics.h"
#include "3rd_party/intgemm/kernels.h"
#include "tensors/tensor.h"
#include "tensors/tensor_allocator.h"
#include "tensors/tensor_operators.h"

namespace marian {
namespace cpu {

static void SigmoidLUT(float min, float max, float quant_mult, marian::Tensor output) {
  using vf = __m256;

  intgemm::AlignedVector<float> input(256);
  for (int i = 0; i < input.size(); ++i)
    input[i] = i * (max - min) / 255.f + min;

  auto vquant_mult = set1_ps<vf>(quant_mult);

  auto input_it = input.as<vf>();
  auto output_it = output->data<int8_t>();

  for (int i = 0; i < input.size() / sizeof(vf); ++i) {
    *output_it++ = intgemm::downcast32to8(
      intgemm::kernels::quantize(intgemm::kernels::sigmoid(*input_it++), vquant_mult),
      intgemm::kernels::quantize(intgemm::kernels::sigmoid(*input_it++), vquant_mult),
      intgemm::kernels::quantize(intgemm::kernels::sigmoid(*input_it++), vquant_mult),
      intgemm::kernels::quantize(intgemm::kernels::sigmoid(*input_it++), vquant_mult));
  }
}

}
}
