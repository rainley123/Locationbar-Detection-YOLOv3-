#ifndef DETECTMULTIMARK_H
#define DETECTMULTIMARK_H

#include <iostream>
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/public/session.h"

using tensorflow::Tensor;
using tensorflow::Status;
using tensorflow::string;
using tensorflow::int32;

typedef long long int64;

// Given an image file name, read in the data, try to decode it as an image,
// resize it to the requested size, and then scale the values as desired.
Status preprocess(const Tensor input_image, const int input_height, const int input_width, std::vector<Tensor>* out_tensors) {
    auto root = tensorflow::Scope::NewRootScope();
    using namespace ::tensorflow::ops;

    std::vector<std::pair<string, Tensor>> inputs = {{"input", input_image}, };

    auto image = Placeholder(root.WithOpName("input"), tensorflow::DataType::DT_FLOAT);

    // The convention for image ops in TensorFlow is that all images are expected
    // to be in batches, so that they're four-dimensional arrays with indices of
    // [batch, height, width, channel]. Because we only have a single image, we
    // have to add a batch dimension of 1 to the start with ExpandDims().
    auto dims_expander = ExpandDims(root.WithOpName("ExpandDims"), image, 0);

    // Bilinearly resize the image to fit the required dimensions.
    auto resized = ResizeBilinear(root.WithOpName("Resize"), dims_expander, Const(root, {input_height, input_width}));

    // This runs the GraphDef network definition that we've just constructed, and
    // returns the results in the output tensor.
    tensorflow::GraphDef graph;
    TF_RETURN_IF_ERROR(root.ToGraphDef(&graph));

    std::unique_ptr<tensorflow::Session> session(tensorflow::NewSession(tensorflow::SessionOptions()));
    TF_RETURN_IF_ERROR(session->Create(graph));
    TF_RETURN_IF_ERROR(session->Run({inputs}, {"Resize"}, {}, out_tensors));
    return Status::OK();
}

Status multipreprocess(const Tensor input_image, const int input_height, const int input_width, std::vector<Tensor>* out_tensors) {
    auto root = tensorflow::Scope::NewRootScope();
    using namespace ::tensorflow::ops;

    std::vector<std::pair<string, Tensor>> inputs = {{"input", input_image}, };

    auto image = Placeholder(root.WithOpName("input"), tensorflow::DataType::DT_FLOAT);

    // Bilinearly resize the image to fit the required dimensions.
    auto resized = ResizeBilinear(root.WithOpName("Resize"), image, Const(root, {input_height, input_width}));

    // This runs the GraphDef network definition that we've just constructed, and
    // returns the results in the output tensor.
    tensorflow::GraphDef graph;
    TF_RETURN_IF_ERROR(root.ToGraphDef(&graph));

    std::unique_ptr<tensorflow::Session> session(tensorflow::NewSession(tensorflow::SessionOptions()));
    TF_RETURN_IF_ERROR(session->Create(graph));
    TF_RETURN_IF_ERROR(session->Run({inputs}, {"Resize"}, {}, out_tensors));
    return Status::OK();
}

Status loadGraph(const string& graph_file_name, std::unique_ptr<tensorflow::Session>* session) {
    tensorflow::GraphDef graph_def;
    Status load_graph_status = tensorflow::ReadBinaryProto(tensorflow::Env::Default(), graph_file_name, &graph_def);
    if (!load_graph_status.ok()) {
        return tensorflow::errors::NotFound("Failed to load compute graph at", graph_file_name, "'");
    }

    session->reset(tensorflow::NewSession(tensorflow::SessionOptions()));
    Status session_create_status = (*session)->Create(graph_def);
    if (!session_create_status.ok()) {
        return session_create_status;
    }

    return Status::OK();
}

void recoverOriginShape(tensorflow::Tensor &boxes, const int32 *input_shape, const int32 *origin_shape) {
    if (boxes.dims() == 2) {
        for (int i = 0; i < boxes.dim_size(0); ++i) {
            for (int j = 0; j < boxes.dim_size(1); ++j) {
                int index = (j + 1) % 2;
                boxes.matrix<float>()(i, j) = boxes.matrix<float>()(i, j) / input_shape[index] * origin_shape[index];
            }
        }
    }

    if (boxes.dims() == 3) {
        for (int num = 0; num < boxes.dim_size(0); ++num) {
            for (int i = 0; i < boxes.dim_size(1); ++i) {
                for (int j = 0; j < boxes.dim_size(2); ++j) {
                    int index = (j + 1) % 2;
                    boxes.tensor<float, 3>()(num, i, j) = boxes.tensor<float, 3>()(num, i, j) / input_shape[index] * origin_shape[index];
                }
            }
        }
    }
}

bool vectorToTensor(const std::vector<double> pixeldata, Tensor &onechannelimage, Tensor &threechannelimage, int input_size) {
    for (int channel = 0; channel < 3; ++channel) {
        for (auto iter = pixeldata.begin(); iter != pixeldata.end(); ++iter) {
            int x = int(iter - pixeldata.begin()) / input_size;
            int y = int(iter - pixeldata.begin()) % input_size;
            threechannelimage.tensor<float, 3>()(x, y, channel) = float(*iter);
        }
    }
    std::copy(pixeldata.begin(), pixeldata.end(), onechannelimage.flat<float>().data());

    return 1;
}

float calSum(const Tensor &core5x5, int x, int y) {
    float sum = 0;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            sum += core5x5.matrix<float>()(y - 1 + i, x - 1 + j);
        }
    }
    return sum;
}

bool correct(const Tensor &image, int *x_center, int *y_center) {
    int count = 0;
    int x_temp = *x_center;
    int y_temp = *y_center;

    while (true) {
        if (count == 5) {
            *x_center = x_temp;
            *y_center = y_temp;
            break;
        }

        Tensor core5x5(tensorflow::DT_FLOAT, tensorflow::TensorShape({5, 5}));
        for (int i = 0; i < 5; ++i) {
            for (int j = 0; j < 5; ++j) {
                core5x5.matrix<float>()(i, j) = image.matrix<float>()(*y_center - 2 + i, *x_center - 2 + j);
            }
        }

        std::vector<float> sumcore;
        for (int i = 1; i < 4; ++i) {
            for (int j = 1; j < 4; ++j) {
                sumcore.push_back(calSum(core5x5, j, i));
            }
        }
        auto min = std::min_element(sumcore.begin(), sumcore.end());
        int index = int(min - sumcore.begin());
        int xoffset = index % 3 - 1;
        int yoffset = index / 3 - 1;

        if (xoffset == 0 && yoffset ==0) break;

        *x_center += xoffset;
        *y_center += yoffset;
        count += 1;
    }
    return 1;
}

bool correctPoints(const Tensor &image, std::vector<int> &x_centers, std::vector<int> &y_centers, const std::vector<int> &labels) {
    int num = int(labels.size());
    for (int i = 0; i < num; ++i) {
        unsigned long index = static_cast<unsigned long>(i);
        if (labels[index] == 0) {
            if (!correct(image, &x_centers[index], &y_centers[index])) {
                std::cout<<"Correct the points failed!"<<std::endl;
                return 0;
            }
        }
        else {
            x_centers.erase(x_centers.begin() + i);
            y_centers.erase(y_centers.begin() + i);
            i --;
            num --;
        }
    }
    return 1;
}

bool removeExtra(const Tensor &image, std::vector<int> &x_centers, std::vector<int> &y_centers, int input_size) {
    // Remove the points which is out of range
    for (auto iter = x_centers.begin(); iter != x_centers.end();) {
        if (double(*iter) / image.dim_size(1) < 50.0 / double(input_size) || double(*iter) / image.dim_size(1) > 200.0 / double(input_size)) {
            y_centers.erase(iter - x_centers.begin() + y_centers.begin());
            iter = x_centers.erase(iter);
        }
        else {
            iter ++;
        }
    }

    // Remove the same points
    std::vector<int> x_temp, y_temp;
    for (unsigned long i = 0; i < x_centers.size(); ++i) {
        bool same = false;
        for (unsigned long j = 0; j < x_temp.size(); ++j) {
            if (x_centers[i] == x_temp[j] && y_centers[i] == y_temp[j]) {
                same = true;
                break;
            }
        }
        if (!same) {
            x_temp.push_back(x_centers[i]);
            y_temp.push_back(y_centers[i]);
        }
    }
    x_centers.assign(x_temp.begin(), x_temp.end());
    y_centers.assign(y_temp.begin(), y_temp.end());

    // Remove the extra points
    if (y_centers.size() == 7) {
        for (unsigned long i = 0; i < 7; ++i) {
            bool indivual = true;
            for (unsigned long j = 0; j < 7; ++j) {
                if (std::abs(y_centers[j] - y_centers[i]) < 5 && (i - j != 0)) {
                    indivual = false;
                    break;
                }
            }
            if (indivual) {
                y_centers.erase(y_centers.begin() + long(i));
                x_centers.erase(x_centers.begin() + long(i));
                break;
            }
        }
    }

    return 1;
}

bool predictMissPoints(const Tensor &image, std::vector<int> &x_centers, std::vector<int> &y_centers) {
    if (x_centers.size() == 9 || x_centers.size() == 6) {
        return 1;
    }

    std::vector<int> x_temp, y_temp;
    x_temp.assign(x_centers.begin(), x_centers.end());
    std::sort(x_temp.begin(), x_temp.end());
    y_temp.assign(y_centers.begin(), y_centers.end());
    std::sort(y_temp.begin(), y_temp.end());

    if (x_centers.size() == 8) {
        // Seek which box is not detected
        unsigned long pre_index_x = 2;
        for (unsigned long i = 0; i < 2; ++i) {
            if (x_temp[i * 3 + 2] - x_temp[i * 3] > 5) {
                pre_index_x = i;
                break;
            }
        }

        unsigned long pre_index_y = 2;
        for (unsigned long i = 0; i < 2; ++i) {
            if (y_temp[i * 3 + 2] - y_temp[i * 3] > 5) {
                pre_index_y = i;
                break;
            }
        }

        // Predict the missin point
        int pre_x = int((x_temp[3 * pre_index_x] + x_temp[3 * pre_index_x + 1]) / 2.0);
        int pre_y = int((y_temp[3 * pre_index_y] + y_temp[3 * pre_index_y + 1]) / 2.0);

        // Recorrect the point
        if (!correct(image, &pre_x, &pre_y)) {
            std::cout<<"Correct the missing point falied!"<<std::endl;
            return 0;
        }
        x_centers.push_back(pre_x);
        y_centers.push_back(pre_y);
    }

    if (x_centers.size() == 5) {
        // Seek which box is not detected
        unsigned long pre_index_x = 2;
        for (unsigned long i = 0; i < 2; ++i) {
            if (x_temp[i * 2 + 1] - x_temp[i * 2] > 5) {
                pre_index_x = i;
                break;
            }
        }
        unsigned long pre_index_y = 0;
        if (y_temp[2] - y_temp[0] > 5) pre_index_y = 0;
        else pre_index_y = 1;

        // Predict the missing point
        int pre_x = x_temp[2 * pre_index_x];
        int pre_y = int((y_temp[3 * pre_index_y] + y_temp[3 * pre_index_y + 1]) / 2.0);

        // Recorrect the point
        if (!correct(image, &pre_x, &pre_y)) {
            std::cout<<"Correct the missing point falied!"<<std::endl;
            return 0;
        }
        x_centers.push_back(pre_x);
        y_centers.push_back(pre_y);
    }
    return 1;
}

bool remainSixPoints(std::vector<int> &x_centers, std::vector<int> &y_centers) {
    std::vector<int> x_temp, y_temp;
    for (int i = 0; i < 3; ++i) {
        auto min =std::min_element(y_centers.begin(), y_centers.end());
        y_temp.push_back(*min);
        x_temp.push_back(*(min - y_centers.begin() + x_centers.begin()));

        y_centers.erase(min);
        x_centers.erase(min - y_centers.begin() + x_centers.begin());
    }

    for (int i = 0; i < 3; ++i) {
        auto max =std::max_element(y_centers.begin(), y_centers.end());
        y_temp.push_back(*max);
        x_temp.push_back(*(max - y_centers.begin() + x_centers.begin()));

        y_centers.erase(max);
        x_centers.erase(max - y_centers.begin() + x_centers.begin());
    }

    x_centers.assign(x_temp.begin(), x_temp.end());
    y_centers.assign(y_temp.begin(), y_temp.end());

    return 1;
}

int detectMark(const std::vector<double> input_vec, std::vector<int> &x_centers, std::vector<int> &y_centers, int input_size)
{
    string graph_path = "./single_image.pb";

    // H & W
    int32 input_shape[2] = {416, 416};
    int32 origin_shape[2] = {input_size, input_size};

    std::string predict_input = "Placeholder";
    std::string predict_output1 = "output_1/concat";
    std::string predict_output2 = "output_1/concat_1";
    std::string predict_output3 = "output_1/concat_2";
    string root_dir = "";

    // Make vector to 2D tensor and 3D tensor
    Tensor threechannelimage(tensorflow::DT_FLOAT, tensorflow::TensorShape({origin_shape[0], origin_shape[1], 3}));
    Tensor onechannelimage(tensorflow::DT_FLOAT, tensorflow::TensorShape({origin_shape[0], origin_shape[1]}));

    if (!vectorToTensor(input_vec, onechannelimage, threechannelimage, input_size)) {
        std::cout<<"Transform vector to tensor failed"<<std::endl;
    }

    // First we load and initialize the model.
    std::unique_ptr<tensorflow::Session> session;
    Status load_graph_status = loadGraph(graph_path, &session);
    if (!load_graph_status.ok()) {
        LOG(ERROR) << load_graph_status;
        return -1;
    }

    // Get the image from disk as a float array of numbers, resized and normalized
    // to the specifications the main graph expects.
    std::vector<Tensor> resized_tensors;
    Status preprocess_status = preprocess(threechannelimage, input_shape[0], input_shape[1], &resized_tensors);
    if (!preprocess_status.ok()) {
        LOG(ERROR) << preprocess_status;
        return -1;
    }
    const Tensor& resized_tensor = resized_tensors[0];

    // Actually run the image through the model.
    std::vector<Tensor> outputs;
    Status run_status = session->Run({{predict_input, resized_tensor}},
                                         {predict_output1, predict_output2, predict_output3}, {}, &outputs);
    if (!run_status.ok()) {
        LOG(ERROR) << "Running model failed: " << run_status;
        return -1;
    }

    // Get the result
    Tensor boxes = outputs[0];
    Tensor scores = outputs[1];
    Tensor labels = outputs[2];

    // Recover the coordinate in original image
    recoverOriginShape(boxes, input_shape, origin_shape);

    // Get the x_center and y_center
    for (int i = 0; i < boxes.dim_size(0); ++i) {
        x_centers.push_back(int((boxes.matrix<float>()(i, 0) + boxes.matrix<float>()(i, 2)) / 2));
        y_centers.push_back(int((boxes.matrix<float>()(i, 1) + boxes.matrix<float>()(i, 3)) / 2));
    }

    // Get the label vector
    std::vector<int> labelvec;
    for (int i = 0; i < labels.dim_size(0); ++i) {
        labelvec.push_back(labels.vec<int>()(i));
    }

    // Correct the boxes to have a better result
    if (!correctPoints(onechannelimage, x_centers, y_centers, labelvec)) {
        std::cout<<"Correct boxes failed!"<<std::endl;
    }

    // Remove the boxes
    if (!removeExtra(onechannelimage, x_centers, y_centers, input_size)) {
        std::cout<<"Remove boxes failed!"<<std::endl;
    }

    //Predict the missing boxes
    if (!predictMissPoints(onechannelimage, x_centers, y_centers)) {
        std::cout<<"Predict boxes failed!"<<std::endl;
    }
    return 0;
}

int detectMultiMark(const std::vector<std::vector<double>> input_vec, std::vector<std::vector<int>> &x_centers, std::vector<std::vector<int>> &y_centers, int input_size)
{
    string graph_path = "../detectmultimark/multi_images.pb";

    // H & W
    int32 input_shape[2] = {416, 416};
    int32 origin_shape[2] = {input_size, input_size};

    std::string predict_input = "Placeholder";
    std::string predict_output1 = "predict_output/unstack";
    std::string predict_output2 = "predict_output/Cast";
    std::string predict_output3 = "predict_output/strided_slice";
    string root_dir = "";

    // Make vector to 3D tensor and 4D tensor
    int image_num = int(input_vec.size());
    Tensor onechannelimages(tensorflow::DT_FLOAT, tensorflow::TensorShape({image_num, origin_shape[0], origin_shape[1]}));
    Tensor threechannelimages(tensorflow::DT_FLOAT, tensorflow::TensorShape({image_num, origin_shape[0], origin_shape[1], 3}));

    for (auto iter1 = input_vec.begin(); iter1 != input_vec.end(); iter1++) {
        int index = int(iter1 - input_vec.begin());
        for (int channel = 0; channel < 3; ++channel) {
            for (auto iter2 = (*iter1).begin(); iter2 != (*iter1).end(); ++iter2) {
                int x = int(iter2 - (*iter1).begin()) / input_size;
                int y = int(iter2 - (*iter1).begin()) % input_size;
                threechannelimages.tensor<float, 4>()(index, x, y, channel) = float(*iter2);

                if (channel == 0) {
                    onechannelimages.tensor<float, 3>()(index, x, y) = float(*iter2);
                }
            }
        }
    }

    // First load and initialize the model.
    std::unique_ptr<tensorflow::Session> session;
    Status load_graph_status = loadGraph(graph_path, &session);
    if (!load_graph_status.ok()) {
        LOG(ERROR) << load_graph_status;
        return -1;
    }

    // Preprocess to make images fit to the input
    std::vector<Tensor> resized_tensors;
    Status preprocess_status = multipreprocess(threechannelimages, input_shape[0], input_shape[1], &resized_tensors);
    if (!preprocess_status.ok()) {
        LOG(ERROR) << preprocess_status;
        return -1;
    }
    const Tensor& resized_tensor = resized_tensors[0];

    // Actually run the image through the model.
    std::vector<Tensor> outputs;
    Status run_status = session->Run({{predict_input, resized_tensor}},
                                         {predict_output1, predict_output2, predict_output3}, {}, &outputs);
    if (!run_status.ok()) {
        LOG(ERROR) << "Running model failed: " << run_status;
        return -1;
    }

    // Get the result
    Tensor scores = outputs[0];
    Tensor labels = outputs[1];
    Tensor boxes = outputs[2];

    // Recover the coordinate in original image
    recoverOriginShape(boxes, input_shape, origin_shape);

    // Get the x_center and y_center
    for (unsigned long num = 0; num < static_cast<unsigned long>(boxes.dim_size(0)); ++num) {
        std::vector<int> x_temp, y_temp;
        for (int i = 0; i < boxes.dim_size(1); ++i) {
            int x = int((boxes.tensor<float, 3>()(int(num), i, 0) + boxes.tensor<float, 3>()(int(num), i, 2)) / 2);
            int y = int((boxes.tensor<float, 3>()(int(num), i, 1) + boxes.tensor<float, 3>()(int(num), i, 3)) / 2);

            if (x != 0 && y != 0){
                x_temp.push_back(x);
                y_temp.push_back(y);
            }
        }
        x_centers.push_back(x_temp);
        y_centers.push_back(y_temp);
    }

    // Get the vector labels
    std::vector<std::vector<int>> labelsvec;
    for (unsigned long num = 0; num < static_cast<unsigned long>(labels.dim_size(0)); ++num) {
        std::vector<int> label_temp;
        for (int i = 0; i < int(x_centers[num].size()); ++i) {
            label_temp.push_back(labels.tensor<int, 2>()(int(num), i));
        }
        labelsvec.push_back(label_temp);
    }

    for (unsigned long num = 0; num < static_cast<unsigned long>(boxes.dim_size(0)); ++num) {
         // Correct the boxes to have a better result
        if (!correctPoints(onechannelimages.SubSlice(int(num)), x_centers[num], y_centers[num], labelsvec[num])) {
            std::cout<<"Correct boxes failed!"<<std::endl;
        }

        // Remove the boxes
        if (!removeExtra(onechannelimages.SubSlice(int(num)), x_centers[num], y_centers[num], input_size)) {
            std::cout<<"Remove boxes failed!"<<std::endl;
        }

        //Predict the missing boxes
        if (!predictMissPoints(onechannelimages.SubSlice(int(num)), x_centers[num], y_centers[num])) {
            std::cout<<"Predict boxes failed!"<<std::endl;
        }

        if (x_centers[num].size() == 9) {
            if (!remainSixPoints(x_centers[num], y_centers[num])) {
                std::cout<<"Get the max and min six points failed"<<std::endl;
            }
        }
    }

    return 0;
}


#endif // DETECTMULTIMARK_H
