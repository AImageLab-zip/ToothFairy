#ifndef FPS_H
#define FPS_H

#include <tuple>

#include <torch/extension.h>


std::tuple<torch::Tensor, torch::Tensor> farthestPointSampling(
    torch::Tensor coordinates,
    torch::Tensor batchCounts,
    const float ratio,
    const int maxPoints
);


#endif
