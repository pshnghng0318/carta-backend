// Copyright 2020 The TensorStore Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Extracts a slice of a volumetric dataset, outputting it as a 2d image.
//
// extract_slice --output_file=/tmp/foo.jpg --input_spec=...

#include <stdint.h>

#include <iostream>
#include <memory>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>
#include <chrono>
#include <fstream>

#include "absl/flags/flag.h"
#include "absl/status/status.h"
#include "absl/strings/match.h"
#include "absl/strings/str_join.h"
#include <nlohmann/json.hpp>
#include "riegeli/bytes/fd_writer.h"
#include "riegeli/bytes/std_io.h"
#include "riegeli/bytes/writer.h"
#include "tensorstore/array.h"
#include "tensorstore/context.h"
#include "tensorstore/data_type.h"
#include "absl/flags/parse.h"
#include "tensorstore/index.h"
#include "tensorstore/index_space/dim_expression.h"
#include "tensorstore/index_space/index_transform.h"
#include "tensorstore/internal/image/avif_writer.h"
#include "tensorstore/internal/image/image_info.h"
#include "tensorstore/internal/image/image_writer.h"
#include "tensorstore/internal/image/jpeg_writer.h"
#include "tensorstore/internal/image/png_writer.h"
#include "tensorstore/internal/image/webp_writer.h"
#include "tensorstore/open.h"
#include "tensorstore/open_mode.h"
#include "tensorstore/spec.h"
#include "tensorstore/strided_layout.h"
#include "tensorstore/tensorstore.h"
#include "tensorstore/util/json_absl_flag.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/span.h"
#include "tensorstore/util/status.h"
#include "tensorstore/util/str_cat.h"

namespace {

using ::tensorstore::Context;
using ::tensorstore::Index;
using ::tensorstore::internal_image::AvifWriter;
using ::tensorstore::internal_image::ImageInfo;
using ::tensorstore::internal_image::ImageWriter;
using ::tensorstore::internal_image::JpegWriter;
using ::tensorstore::internal_image::PngWriter;
using ::tensorstore::internal_image::WebPWriter;

template <typename InputArray>
absl::Status Validate(const InputArray& input) {
  std::vector<std::string> errors;
  if (input.rank() != 5) {
    errors.push_back(tensorstore::StrCat("expected input of rank 5, not ",
                                         input.rank()));
  }

  // Validate data types
  if (input.dtype() != tensorstore::dtype_v<tensorstore::dtypes::float32_t>) {
    errors.push_back("expected input.dtype of float32");
  }

  // Validate shapes
  auto input_shape = input.domain().shape();
  auto c = input.rank() - 1;

  if (!errors.empty()) {
    return absl::InvalidArgumentError(tensorstore::StrCat(
        "tensorstore validation failed: ", absl::StrJoin(errors, ", ")));
  }
  return absl::OkStatus();
}

/// Load a 2d tensorstore volume slice and render it as an image.
absl::Status Run(tensorstore::Spec input_spec, std::string output_filename) {
  auto context = Context::Default();

  double open_time = 0.0;
  auto open_start = std::chrono::steady_clock::now();

  // Open input tensorstore and resolve the bounds.
  TENSORSTORE_ASSIGN_OR_RETURN(
      
      auto input,
      tensorstore::Open(input_spec, context, tensorstore::OpenMode::open,
                        tensorstore::ReadWriteMode::read)
          .result());

  auto open_end = std::chrono::steady_clock::now();
  open_time = std::chrono::duration<double>(open_end - open_start).count();
  std::cout << "Open time: " << open_time << std::endl;

  /// To render something other than the top-layer, A spec should
  /// include a transform.
  tensorstore::Result<tensorstore::IndexTransform<>> transform(
      std::in_place, tensorstore::IdentityTransform(input.domain()));

  std::cerr << std::endl << "Before: " << *transform << std::endl;

  // DimRange(...).IndexSlice(0) transform below assumes that extra dimensions
  // are 0-based; so make all dimensions 0-based here.
  transform = transform | tensorstore::AllDims().TranslateTo(0);

  // By convention, assume that the first dimension is Y, and the second is X,
  // and the third is C. The C++ api could use some help with labelling missing
  // dimensions, actually...
  // bool has_x = false;
  // bool has_y = false;
  // bool has_c = false;
  // for (auto& l : input.domain().labels()) {
  //   has_x = has_x || l == "x";
  //   has_y = has_y || l == "y";
  //   has_c = has_c || l == "c";
  // }

  // if (has_y) {
  //   std::cerr << "Transforming Y dimension" << std::endl;
  //   transform = transform | tensorstore::Dims("y").MoveTo(0);
  // }
  // if (has_x) {
  //   std::cerr << "Transforming X dimension" << std::endl;
  //   transform = transform | tensorstore::Dims("x").MoveTo(1);
  // }
  // if (has_c) {
  //   std::cerr << "Transforming C dimension" << std::endl;
  //   transform = transform | tensorstore::Dims("c").MoveToBack();
  // }
  

// transform = tensorstore::IdentityTransform(input.domain())
//     | tensorstore::Dims(4).BoxSlice(tensorstore::Box<>({0}, {1500}))  // x=4742
//     | tensorstore::Dims(3).BoxSlice(tensorstore::Box<>({0}, {1000}))  // y=7763
//     | tensorstore::Dims(2).IndexSlice(0)  // stokes
//     | tensorstore::Dims(0).IndexSlice(0); // time

  transform = tensorstore::IdentityTransform(input.domain())
    | tensorstore::Dims(0).IndexSlice(0)
    | tensorstore::Dims(0).IndexSlice(0)  // After previous slice, next is at 0
    | tensorstore::Dims(0).IndexSlice(0);  // After previous slice, next is at 0
    // | tensorstore::Dims(0).BoxSlice(tensorstore::Box<>({0}, {7763}))
    // | tensorstore::Dims(1).BoxSlice(tensorstore::Box<>({0}, {4742}));

std::cerr << std::endl << "After: " << *transform << std::endl;

  TENSORSTORE_RETURN_IF_ERROR(Validate(input));
  auto constrained_input = input | *transform;
  TENSORSTORE_RETURN_IF_ERROR(constrained_input);

  auto transposed_transform = constrained_input| tensorstore::Dims(1,0).Transpose({0,1});
  

  //std::cerr << "Spec: " << *(constrained_input->spec()) << std::endl;

  double read_time = 0.0;
  auto read_start = std::chrono::steady_clock::now();

  TENSORSTORE_ASSIGN_OR_RETURN(
      auto slice,
      // tensorstore::Read<tensorstore::zero_origin>(constrained_input).result());
      tensorstore::Read<tensorstore::zero_origin>(transposed_transform).result());

  auto read_end = std::chrono::steady_clock::now();
  read_time = std::chrono::duration<double>(read_end - read_start).count();
  std::cout << "Read time: " << read_time << std::endl;

  // auto slice_result = sliceYX | tensorstore::Dims(0,1).Transpose({1,0});
  // auto slice = std::move(slice_result).value();  // 或使用 slice_transposed_result.value()

  ///// sum slice.data to make sure read happened
  auto shape = slice.shape();
  std::cout << "Slice shape: ";
  for (auto s : shape) std::cout << s << " ";
    std::cout << std::endl;
    
  // auto n_elements = slice.num_elements();
  // auto data_ptr = reinterpret_cast<float*>(slice.data());  // float32 對應 float
  // int num_p[shape[1]] = {0};
  // double total[shape[1]] = {0.0};

  // std::ofstream outfile("output.dat");

  // for (size_t j = 0; j < shape[1]; ++j) {
  //   for (size_t i = 0; i < shape[0]; ++i) {
  //     if (!std::isnan(data_ptr[j * shape[0] + i])) {
  //       total[j] += static_cast<float>(data_ptr[j * shape[0] + i]);
  //       num_p[j] += 1;
  //     }
  //   }
  //   outfile << "Row " << j << ": " << total[j] << " " << num_p[j] << std::endl;
  // }
  // outfile.close();
  /////
  
  

  /////////////// test without output ///////////////

  auto shape_yxc = slice.shape();
  std::cout << "rank = " << slice.rank() << std::endl;
  // std::cout << "slice.rank = " << slice.rank() << std::endl;
  ImageInfo info{/*height=*/static_cast<int32_t>(shape_yxc[0]),
                 /*width=*/static_cast<int32_t>(shape_yxc[1]),
                 /*num_components=*/slice.rank() == 2
                     ? 1
                     : static_cast<int32_t>(shape_yxc[2])};

  std::unique_ptr<ImageWriter> writer;
  std::unique_ptr<riegeli::Writer> output;

  // Select the image format.
  if (absl::EndsWith(output_filename, ".jpg") ||
      absl::EndsWith(output_filename, ".jpeg")) {
    writer = std::make_unique<JpegWriter>();
  } else if (absl::EndsWith(output_filename, ".avif")) {
    writer = std::make_unique<AvifWriter>();
  } else if (absl::EndsWith(output_filename, ".webp")) {
    writer = std::make_unique<WebPWriter>();
  } else if (absl::EndsWith(output_filename, ".png") ||
             output_filename == "-") {
    writer = std::make_unique<PngWriter>();
  } else {
    return absl::InvalidArgumentError(
        "Only .jpeg, .webp, .avif, and .png output formats permitted");
  }

  // Maybe output to stdout.
  if (output_filename == "-" || absl::StartsWith(output_filename, "-.")) {
    // TODO: Also check istty.
    output = std::make_unique<riegeli::StdOut>();
  } else {
    output = std::make_unique<riegeli::FdWriter<>>(output_filename);
  }
  if (!output->ok()) return output->status();

  // And encode the image.
  TENSORSTORE_RETURN_IF_ERROR(writer->Initialize(output.get()));

  // float slice.data to 0-255
  size_t num_elements = info.width * info.height * info.num_components;
  std::cout << "Num elements: " << num_elements << std::endl;

  // Mapping values of slice.data() to 0-255 and upside down for imaging
  std::vector<unsigned char> buffer(num_elements);

  const float* src = reinterpret_cast<const float*>(slice.data());
  const float min_val = -1e-5f;
  const float max_val =  1e-5f;
  const float scale   = 255.0f / (max_val - min_val);  // 255 / 2e-5

  for (size_t j = 0; j < info.height ; j++)
    for (size_t i = 0; i < info.width ; i++) {
    
    float v = src[j * info.width + i];
    if (std::isnan(v)) {
        buffer[(info.height - 1 - j) * info.width + i] = 0;
        continue;
    }

    float norm = (v - min_val) * scale;

    if (norm < 0.0f) norm = 0.0f;
    if (norm > 255.0f) norm = 255.0f;

    buffer[(info.height - 1 - j) * info.width + i] = static_cast<unsigned char>(norm);
  }

  TENSORSTORE_RETURN_IF_ERROR(writer->Encode(
      info,
      // tensorstore::span(reinterpret_cast<const unsigned char*>(slice.data()),      
      //                info.width * info.height * info.num_components)));
      tensorstore::span<const unsigned char>(buffer.data(), buffer.size())
  ));

  std::cout << "buffer size = " << buffer.size() << std::endl;
  return writer->Done();

  /////////////// test without output ///////////////

  return absl::OkStatus();
}

}  // namespace

tensorstore::Spec DefaultInputSpec() {
  return tensorstore::Spec::FromJson(
             {
                 {"open", true},
                 {"driver", "n5"},
                 {"kvstore", {{"driver", "memory"}}},
                 {"path", "input"},
                 {"metadata",
                  {
                      {"compression", {{"type", "raw"}}},
                      {"dataType", "uint8"},
                      {"blockSize", {16, 16, 1}},
                      {"dimensions", {64, 64, 1}},
                  }},
             })
      .value();
}

/// Required. The DefaultInputSpec() renders a 64x64 black square.
///
/// Specify a transform along with the spec to select a specific region.
/// For example, this renders a 512x512 region from the middle of the H01
/// dataset release.
///
///   --input_spec='{
///     "driver":"neuroglancer_precomputed",
///     "kvstore":{"bucket":"h01-release","driver":"gcs"},
///     "path":"data/20210601/4nm_raw",
///     "scale_metadata":{ "resolution":[8,8,33] },
///     "transform":{
///         "input_labels": ["x", "y"],
///         "input_inclusive_min":[320553,177054],
///         "input_shape":[512,512],
///         "output":[{"input_dimension":0},
///                   {"input_dimension":1},
///                   {"offset":3667},{}]}
///   }'
///
/// And this just copies the image:
///
///   --input_spec='{
///     "driver":"png",
///     "kvstore":"file:///home/data/myfile.png",
///     "domain": { "labels": ["y", "x", "c"] }
///   }'
///
ABSL_FLAG(tensorstore::JsonAbslFlag<tensorstore::Spec>, input_spec,
          DefaultInputSpec(), "tensorstore JSON input specification");

// Required. The output file. Must be a .jpeg, .webp, .avif, or .png.
ABSL_FLAG(std::string, output_file, "-",
          "Slice will be written to this image file; use - for STDOUT");

int main(int argc, char** argv) {
  
  absl::ParseCommandLine(argc, argv);  // InitTensorstore
  double total_time = 0.0;
  auto total_start = std::chrono::steady_clock::now();

  if (absl::GetFlag(FLAGS_output_file).empty()) {
    std::cerr << "Missing required flag: --output_file" << std::endl;
    return 2;
  }
  std::cerr << "--input_spec="
            << AbslUnparseFlag(absl::GetFlag(FLAGS_input_spec)) << std::endl;

  auto status = Run(absl::GetFlag(FLAGS_input_spec).value,
                    absl::GetFlag(FLAGS_output_file));

  if (!status.ok()) {
    std::cerr << status << std::endl;
  }
  auto total_end = std::chrono::steady_clock::now();
  total_time = std::chrono::duration<double>(total_end - total_start).count();
  std::cout << "Total time: " << total_time << std::endl;

  return status.ok() ? 0 : 1;
}
