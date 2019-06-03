#include <iostream>
#include <fstream>
#include <dirent.h>
#include <unistd.h>
#include "detectmultimark.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/public/session.h"

using tensorflow::Tensor;
using tensorflow::Status;
using tensorflow::string;

typedef long long int64;
typedef unsigned int uint8;

static Status readEntireFile(tensorflow::Env* env, const string& filename, Tensor *output) {
    tensorflow::uint64 file_size = 0;
    TF_RETURN_IF_ERROR(env->GetFileSize(filename, &file_size));

    string contents;
    contents.resize(file_size);

    std::unique_ptr<tensorflow::RandomAccessFile> file;
    TF_RETURN_IF_ERROR(env->NewRandomAccessFile(filename, &file));

    tensorflow::StringPiece data;
    TF_RETURN_IF_ERROR(file->Read(0, file_size, &data, &(contents)[0]));
    if (data.size() != file_size) {
        return tensorflow::errors::DataLoss("Truncated read of '", filename,
                                            "' expected ", file_size, " got ",
                                            data.size());
      }
    output->scalar<string>()() = string(data);
    return Status::OK();
}

Status readFile(const string& file_name, std::vector<Tensor>* out_tensors) {
    auto root = tensorflow::Scope::NewRootScope();
    using namespace ::tensorflow::ops;

    string input_name = "file_reader";
    string output_name = "Unstack";

    // read file_name into a tensor named input
    Tensor input(tensorflow::DT_STRING, tensorflow::TensorShape());
    TF_RETURN_IF_ERROR(readEntireFile(tensorflow::Env::Default(), file_name, &input));

    // use a placeholder to read input data
    auto file_reader = Placeholder(root.WithOpName("input"), tensorflow::DataType::DT_STRING);

    std::vector<std::pair<string, Tensor>> inputs = {
        {"input", input}
    };

    // Now try to figure out what kind of file it is and decode it.
    const int wanted_channels = 3;
    tensorflow::Output image_reader;
    if (tensorflow::str_util::EndsWith(file_name, ".png")) {
        image_reader = DecodePng(root.WithOpName("png_reader"), file_reader, DecodePng::Channels(wanted_channels));
    }

    // Now cast the image data to float so we can do normal math on it.
    auto float_caster = Cast(root.WithOpName("float_cast"), image_reader, tensorflow::DT_FLOAT);

    float mean = 0;
    float graylevel = 255;
    auto div = Div(root.WithOpName("Div"), Sub(root, float_caster, {mean}), {graylevel});

    auto onechannel = Unstack(root.WithOpName("Unstack"), div, 3, Unstack::Axis(2))[0];

    tensorflow::GraphDef graph;
    TF_RETURN_IF_ERROR(root.ToGraphDef(&graph));

    std::unique_ptr<tensorflow::Session> session(tensorflow::NewSession(tensorflow::SessionOptions()));
    TF_RETURN_IF_ERROR(session->Create(graph));
    TF_RETURN_IF_ERROR(session->Run({inputs}, {output_name}, {}, out_tensors));

    return Status::OK();
}

void SplitString(const string& s, std::vector<string>& v, const string& c)
{
    string::size_type pos1, pos2;
    pos2 = s.find(c);
    pos1 = 0;
    while(string::npos != pos2)
    {
        v.push_back(s.substr(pos1, pos2-pos1));

        pos1 = pos2 + c.size();
        pos2 = s.find(c, pos1);
    }
    if(pos1 != s.length())
        v.push_back(s.substr(pos1));
}

void getFiles(string path, std::vector<string>& files)
{
    DIR *dir;
    struct dirent *ptr;

    if ((dir=opendir(path.c_str())) == nullptr)
        {
            perror("Open dir error...");
            exit(1);
        }

    while ((ptr=readdir(dir)) != nullptr)
    {
        if (strcmp(ptr->d_name, ".") == 0 || strcmp(ptr->d_name, "..") == 0)
            continue;
        files.push_back(ptr->d_name);
    }
    closedir(dir);
}

int main(int argc, char* argv[])
{
    if (argc < 2) {
        std::cout<<"Please give the input dir!"<<std::endl;
    }
    std::vector<string> files;
    string image_dir = argv[1];
    getFiles(image_dir, files);

    std::vector<std::vector<int>> x_centers, y_centers;
    std::vector<std::vector<double>> pixeldatavec;

    for (std::size_t index = 0; index < files.size(); ++index) {
        string image_path = image_dir + files[index];

        std::vector<Tensor> origin_tensors;
        Status read_tensor_status = readFile(image_path, &origin_tensors);
        if (!read_tensor_status.ok()) {
            LOG(ERROR) << read_tensor_status;
            return -1;
        }

        Tensor origin_image = origin_tensors[0];

        std::vector<double> pixeldata;
        for (int i = 0; i < 256; ++i) {
            for (int j = 0; j < 256; ++j) {
                pixeldata.push_back(double(origin_image.matrix<float>()(i, j)));
            }
        }
        pixeldatavec.push_back(pixeldata);
    }

    if (detectMultiMark(pixeldatavec, x_centers, y_centers, 256) == 0) {
        std::cout<<"Detect the mark successfully!"<<std::endl;
    }
    else {
        std::cout<<"Detect the mark failed!"<<std::endl;
    }
    std::cout<<"Run main function successfully!"<<std::endl;

    for (auto iter1 = x_centers.begin(); iter1 != x_centers.end(); iter1++) {
        unsigned long num = static_cast<unsigned long>(iter1 - x_centers.begin());
        std::cout<<files[num]<<std::endl;
        string result_name = "./result_txt/" + files[num] + ".txt";
        std::ofstream out(result_name);

        for (auto iter2 = (*iter1).begin(); iter2 < (*iter1).end(); ++iter2) {
            unsigned long index = static_cast<unsigned long>(iter2 - (*iter1).begin());
            std::cout<<x_centers[num][index]<<" "<<y_centers[num][index]<<std::endl;
            out<<x_centers[num][index]<<" "<<y_centers[num][index]<<std::endl;
        }
        out.close();
        std::cout<<std::endl;
    }

    return 0;
}
