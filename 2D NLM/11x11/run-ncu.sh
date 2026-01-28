

#!/bin/bash

# 获取脚本所在目录
script_dir="$(dirname "$(realpath "$0")")"

# 修改nlm.cpp文件的main部分
updateMain() {
    sed -i 's,^\( *processImage(".*", \).*,\1'"$1"');,' "$script_dir/nlm.cpp"
}

# 打印参数信息
printParameters() {
    echo "Image Path: $1, Search Window Size: $2, Neighborhood Size: $3"
}

# 进入 build 目录
cd "$script_dir/build"

# 图片路径数组
imagePaths=("256.bmp" "512.bmp" "1024.bmp" "2048.bmp")

# 搜索窗口大小
searchWindowSizes=("21")

# 邻域窗口大小
neighborhoodSizes=("11")

# 循环遍历测试用例并执行
for imagePath in "${imagePaths[@]}"; do
    for searchSize in "${searchWindowSizes[@]}"; do
        for neighborSize in "${neighborhoodSizes[@]}"; do
            # 修改main参数
            updateMain "processImage(\"$script_dir/data/$imagePath\", $searchSize, $neighborSize)"

            # 打印参数信息
            
			echo -e "\n"
			echo -e "\n"
            printParameters "$imagePath" "$searchSize" "$neighborSize"

            # 编译项目
            # echo "Compiling project..."
            # cmake ..
            make

            # 运行 CUDA 程序
            sudo /usr/local/cuda-11.6/bin/ncu ./nl-means "$script_dir/data/$imagePath" "$searchSize" "$neighborSize"
        done
    done
done

# 返回上一级目录
cd "$script_dir"
