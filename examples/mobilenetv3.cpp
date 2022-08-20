// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2017 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#include <stdio.h>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "layer/softmax.h"
#include "platform.h"
#include "net.h"
#if NCNN_VULKAN
#include "gpu.h"
#endif // NCNN_VULKAN

struct Object
{
    int label;
    float prob;
};

static int detect_mobilenet(const cv::Mat &bgr, std::vector<float> &cls_scores)
{
    ncnn::Net mobilenet;

#if NCNN_VULKAN
    mobilenet.opt.use_vulkan_compute = true;
#endif // NCNN_VULKAN

    mobilenet.load_param("mobilenet_v3.param");
    mobilenet.load_model("mobilenet_v3.bin");

    const int target_size = 224;

    int img_w = bgr.cols;
    int img_h = bgr.rows;

    ncnn::Mat in = ncnn::Mat::from_pixels_resize(bgr.data, ncnn::Mat::PIXEL_BGR2RGB, bgr.cols, bgr.rows, target_size, target_size);

    const float mean_vals[3] = {0, 0, 0};
    const float norm_vals[3] = {1.0 / 255.0, 1.0 / 255.0, 1.0 / 255.0};
    in.substract_mean_normalize(mean_vals, norm_vals);

    ncnn::Extractor ex = mobilenet.create_extractor();
    //     ex.set_num_threads(4);

    ex.input("input", in);
    ncnn::Mat out;
    ex.extract("output", out);
    printf("%d %d %d\n", out.w, out.h, out.c);

    cls_scores.resize(out.w);
    for (int j = 0; j < out.w; j++)
    {
        cls_scores[j] = out[j];
    }

    return 0;
}

static int print_topk(const std::vector<float> &cls_scores, int topk)
{
    // partial sort topk with index
    int size = cls_scores.size();
    std::vector<std::pair<float, int>> vec;
    vec.resize(size);
    for (int i = 0; i < size; i++)
    {
        vec[i] = std::make_pair(cls_scores[i], i);
    }

    std::partial_sort(vec.begin(), vec.begin() + topk, vec.end(),
                      std::greater<std::pair<float, int>>());

    // print topk and score
    for (int i = 0; i < topk; i++)
    {
        float score = vec[i].first;
        int index = vec[i].second;
        fprintf(stderr, "%d = %f\n", index, score);
    }

    return 0;
}

int main(int argc, char **argv)
{
    if (argc != 2)
    {
        fprintf(stderr, "Usage: %s [imagepath]\n", argv[0]);
        return -1;
    }

    const char *imagepath = argv[1];

    cv::Mat m = cv::imread(imagepath, cv::IMREAD_COLOR);

    if (m.empty())
    {
        fprintf(stderr, "cv::imread %s failed\n", imagepath);
        return -1;
    }

#if NCNN_VULKAN
    ncnn::create_gpu_instance();
#endif // NCNN_VULKAN

    std::vector<float> cls_scores;
    detect_mobilenet(m, cls_scores);
    print_topk(cls_scores, 3);

#if NCNN_VULKAN
    ncnn::destroy_gpu_instance();
#endif // NCNN_VULKAN

    return 0;
}
