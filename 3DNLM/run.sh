#!/bin/bash

# 获取脚本所在目录
script_dir="$(dirname "$(realpath "$0")")"

# 打印参数信息
printParameters() {
    echo "Image Path: $1, Search Window Size: $2, Neighborhood Size: $3, Channels: $4"
    echo "SS: $5, NS: $6, IMAGE_SIZE_X: $7, IMAGE_SIZE_Y: $8, depth: $9"
}

# 修改config.h文件的参数
updateConfig() {
    sed -i 's/#define SS .*/#define SS '"$1"'/g' "$script_dir/config.h"
    sed -i 's/#define NS .*/#define NS '"$2"'/g' "$script_dir/config.h"
    sed -i 's/#define IMAGE_SIZE_X .*/#define IMAGE_SIZE_X '"$3"'/g' "$script_dir/config.h"
    sed -i 's/#define IMAGE_SIZE_Y .*/#define IMAGE_SIZE_Y '"$3"'/g' "$script_dir/config.h"
    sed -i 's/#define depth .*/#define depth '"$4"'/g' "$script_dir/config.h"
}

# 进入 build 目录
cd "$script_dir/build"

# 图片路径数组 "128.bmp" "256.bmp" "512.bmp" 
imagePaths=("512.bmp" )

# 搜索窗口大小
searchWindowSizes=("11" "13")

# 邻域窗口大小"5" "7" "9" "11"
neighborhoodSizes=("13" "15" "17")

# 通道数
channelsSizes=("16" "32" "64" "128")

# 循环遍历测试用例并执行
for imagePath in "${imagePaths[@]}"; do
    # 根据图片路径设置IMAGE_SIZE_X和IMAGE_SIZE_Y的大小
    case $imagePath in
        "128.bmp")
            imageSize=128
            ;;
        "256.bmp")
            imageSize=256
            ;;
        "512.bmp")
            imageSize=512
            ;;
        *)
            echo "Unknown image size for $imagePath"
            exit 1
            ;;
    esac

    for searchSize in "${searchWindowSizes[@]}"; do
        for neighborSize in "${neighborhoodSizes[@]}"; do
            for channels in "${channelsSizes[@]}"; do
                # 修改config.h参数
                updateConfig "$searchSize" "$neighborSize" "$imageSize" "$channels"

                # 获取config.h中的参数值
                SS=$(grep '#define SS ' "$script_dir/config.h" | awk '{print $3}')
                NS=$(grep '#define NS ' "$script_dir/config.h" | awk '{print $3}')
                IMAGE_SIZE_X=$(grep '#define IMAGE_SIZE_X ' "$script_dir/config.h" | awk '{print $3}')
                IMAGE_SIZE_Y=$(grep '#define IMAGE_SIZE_Y ' "$script_dir/config.h" | awk '{print $3}')
                depth=$(grep '#define depth ' "$script_dir/config.h" | awk '{print $3}')

                # 打印参数信息
                echo -e "\n"
                echo -e "\n"
                printParameters "$imagePath" "$searchSize" "$neighborSize" "$channels" "$SS" "$NS" "$IMAGE_SIZE_X" "$IMAGE_SIZE_Y" "$depth"

                # 编译项目
                make

                # 运行 CUDA 程序
                nsys nvprof ./nl-means "$script_dir/data/$imagePath"
            done
        done
    done
done

# 返回上一级目录
cd "$script_dir"
