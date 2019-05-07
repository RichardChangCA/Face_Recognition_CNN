# -*- codeing: utf-8 -*-
import sys
import os
import cv2
import dlib
import argparse

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True,
	help="path to input dir")
ap.add_argument("-o", "--output", required=True,
	help="path to output dir")
args = vars(ap.parse_args())

input_dir = args["input"]
output_dir = args["output"]
size = 64

if not os.path.exists(output_dir):
	os.makedirs(output_dir)

#使用dlib自带的frontal_face_detector作为我们的特征提取器
detector = dlib.get_frontal_face_detector()

dirnames = os.listdir(input_dir)
#print(dirnames)
# for (path, dirnames, filenames) in os.walk(input_dir):
#     print("path", path)
#     print("dirnames", dirnames)
for dir in dirnames:

	new_path = output_dir + "/" + dir
	isExists = os.path.exists(new_path)
    # 判断结果
	if not isExists:
        # 如果不存在则创建目录
        # 创建目录操作函数
		os.makedirs(new_path)

		print(new_path + ' 创建成功')
	else:
        # 如果目录存在则不创建，并提示目录已存在
		print(new_path + ' 目录已存在')
		continue

	next_dir = input_dir + "/" + dir
	for (next_path, next_dirnames, next_filenames) in os.walk(next_dir):
		print("next_filenames", next_filenames)
		for filename in next_filenames:
			if filename.endswith('.bmp') or filename.endswith('.BMP') or filename.endswith('.PNG') or filename.endswith('.png') or filename.endswith('.JPG') or filename.endswith('.jpg'):
				picture_name = os.path.basename(filename) #获取当前文件名
				print('Being processed picture %s' % picture_name)
				img_path = next_path+'/'+filename
                # 从文件读取图片
				img = cv2.imread(img_path)
                # 转为灰度图片
				gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                # 使用detector进行人脸检测 dets为返回的结果
				dets = detector(gray_img, 1)

                #使用enumerate 函数遍历序列中的元素以及它们的下标
                #下标i即为人脸序号
                #left：人脸左边距离图片左边界的距离 ；right：人脸右边距离图片左边界的距离
                #top：人脸上边距离图片上边界的距离 ；bottom：人脸下边距离图片上边界的距离
				for i, d in enumerate(dets):
					x1 = d.top() if d.top() > 0 else 0
					y1 = d.bottom() if d.bottom() > 0 else 0
					x2 = d.left() if d.left() > 0 else 0
					y2 = d.right() if d.right() > 0 else 0
                    # img[y:y+h,x:x+w]
					face = img[x1:y1, x2:y2]
                    # 调整图片的尺寸
					face = cv2.resize(face, (size, size))
					cv2.imshow('image', face)
                    # 保存图片
					cv2.imwrite(new_path+'/'+picture_name, face)
				# key = cv2.waitKey(30) & 0xff
				# if key == 27:
					# sys.exit(0)
print("[INFO]end of processing")
