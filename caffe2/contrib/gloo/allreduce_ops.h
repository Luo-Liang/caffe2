#pragma once

#include <algorithm>

#include "caffe2/contrib/gloo/common.h"
#include "caffe2/core/operator.h"
#include "caffe2/utils/math.h"

#include <gloo/algorithm.h>
#include <gloo/common/error.h>
#include <gloo/context.h>
#include <cassert>
#include <cstdlib>
#include <iostream>

#include <PHub/Integration.h>

namespace caffe2 {
namespace gloo {

template <class Context>
class AllreduceOp final : public Operator<Context> {
  enum Mode { RING_FULL, RING_CHUNKED, HALVING_DOUBLING, PHUB };

 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  AllreduceOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        ws_(ws),
        status_blob_(
            OperatorBase::GetSingleArgument<std::string>("status_blob", "")),
        gpu_direct_(
            OperatorBase::GetSingleArgument<bool>("gpu_direct", false)) {
    if (status_blob_ != "") {
      ws_->CreateBlob(status_blob_);
    }
    CHECK(status_blob_ != "");
    // build phub incrementally
    // const char* algo = std::getenv("GLOO_ALGORITHM");
    // if (algo != NULL && strcmp(algo, "PHUB") == 0) {
     
    // }
  }

  virtual ~AllreduceOp() {}

  bool RunOnDevice() override {
    std::call_once(once_, [&] { initialize(); });

    // If any parameter has changed in between runs, the initialized
    // algorithm is invalid and cannot be used.
    update(current_);
    CAFFE_ENFORCE(current_ == init_, "Inputs/outputs have changed");

    try {
      algorithm_->run();
    } catch (::gloo::IoException& ioe) {
      LOG(ERROR) << "Caught gloo IO exception: " << ioe.what();
      if (status_blob_ != "") {
        signalFailure(ws_->GetBlob(status_blob_), ioe);
        return false;
      } else {
        throw ioe;
      }
    }
    return true;
  }

 protected:
  void initialize() {
    const char* algo = std::getenv("GLOO_ALGORITHM");
    Mode mode = HALVING_DOUBLING;
    if (algo != NULL) {
      if (strcmp(algo, "RING_FULL") == 0) {
        mode = RING_FULL;
      } else if (strcmp(algo, "RING_CHUNKED") == 0) {
        mode = RING_CHUNKED;
      } else if (strcmp(algo, "HALVING_DOUBLING") == 0) {
        mode = HALVING_DOUBLING;
      } else if (strcmp(algo, "PHUB") == 0) {
        mode = PHUB;
      } else {
        CAFFE_ENFORCE(false);
      }
    }
    auto bytes = Input(1).nbytes();

    // Store which inputs/outputs this instance initialized with
    update(init_);

    // Verify inputs == ouputs
    CAFFE_ENFORCE_EQ(init_.inputs.size(), init_.outputs.size());
    for (auto i = 0; i < init_.inputs.size(); i++) {
      CAFFE_ENFORCE_EQ(init_.inputs[i], init_.outputs[i]);
    }

    // Verify tensors all have same size
    size_t size = Input(1).size();
    for (auto i = 2; i < InputSize(); i++) {
      CAFFE_ENFORCE_EQ(Input(i).size(), size);
    }

    // Verify tensors all have same type
    TypeMeta meta = Input(1).meta();
    for (auto i = 2; i < InputSize(); i++) {
      CAFFE_ENFORCE(Input(i).meta() == meta);
    }

    switch (mode) {
      case RING_FULL:
        initializeRingFull();
        return;
      case RING_CHUNKED:
        initializeRingChunked();
        return;
      case HALVING_DOUBLING:
        initializeHalvingDoubling();
        return;
      case PHUB:
        initializePHub();
        return;
    }
    CAFFE_ENFORCE(false, "Unreachable code");
  }

  void initializeHalvingDoubling();
  void initializeRingFull();
  void initializeRingChunked();
  void initializePHub();

  std::once_flag once_;
  std::unique_ptr<::gloo::Algorithm> algorithm_;

  // Captures the parameters passed to Gloo when first initialized.
  // An instance is updated every time this op runs and is compared
  // to the reference instance for equality. If any parameter has
  // changed from run to run, the initialized algorithm is invalid.
  void update(GlooParameters& params) {
    params.context = OperatorBase::Input<std::shared_ptr<::gloo::Context>>(0);
    params.inputs.resize(InputSize() - 1);
    params.outputs.resize(OutputSize());
    for (auto i = 0; i < params.inputs.size(); i++) {
      params.inputs[i] = Input(i + 1).template raw_data();
      params.outputs[i] = Output(i)->template raw_mutable_data();
    }
    params.size = Output(0)->size();
    params.meta = Output(0)->meta();
  }

  GlooParameters init_;
  GlooParameters current_;
  Workspace* ws_;
  std::string status_blob_;
  const bool gpu_direct_;
};

} // namespace gloo
} // namespace caffe2
