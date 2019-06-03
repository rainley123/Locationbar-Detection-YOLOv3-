头文件:detectmultimark.h
INCLUDE (换了路径记得修改):      ./include/tensorflow \
               			 ./include/tensorflow/bazel-genfiles \
               			 ./include/eigen3 \
               			 ./include/tensorflow/tensorflow/contrib/makefile/downloads/absl \

LIBS(换了路径记得修改):   	 -L./tf_lib/ -ltensorflow_cc \
          		  	 -L./tf_lib/ -ltensorflow_framework \

模型结构和参数文件: multi_images.pb
PS:该文件的位置改变之后,需要修改detectmultimark.h文件中427行的路径

调用的检测函数: int detectMultiMark(const std::vector<std::vector<double>> input_vec, std::vector<std::vector<int>> &x_centers, std::vector<std::vector<int>> &y_centers, int input_size)
参数说明:	input_vec 二维的vector, 存放图片的像素值,一张图片的像素值变成一维的vecotr
		x_centers 二维的vector,存放检测定位柱中心坐标的x值
		y_centers 二维的vector,存放检测定位柱中心坐标的y值
		input_size 输入图片的尺寸
