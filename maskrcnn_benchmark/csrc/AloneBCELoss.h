#pragma once

#include "cpu/vision.h"

#ifdef WITH_CUDA
#include "cuda/vision.h"
#endif

// Interface for Python
at::Tensor AloneBCELoss_forward(
		const at::Tensor& logits,
                const at::Tensor& targets,
		const int num_classes) {
    return AloneBCELoss_forward_cuda(logits, targets, num_classes);
}

at::Tensor AloneBCELoss_backward(
			     const at::Tensor& logits,
                             const at::Tensor& targets,
			     const at::Tensor& d_losses,
			     const int num_classes) {
    return AloneBCELoss_backward_cuda(logits, targets, d_losses, num_classes);
}
