#-*- coding: UTF-8 -*- 
 
from PIL import Image
import random
import os
import numpy as np
import argparse


#读取文件内的所有,迭代，将生成的图片与原图片放在同一文件夹下

def read_file_all(data_dir_path):
    for f in os.listdir(data_dir_path):
        print("data_dir_path:",data_dir_path)
        data_file_path = os.path.join(data_dir_path, f)
        if os.path.isfile(data_file_path):
            image_rotate(data_file_path, data_dir_path)
			
            #print(collected)
        else:
            read_file_all(data_file_path)
            #文件夹下套文件夹情况

def read_file_all_again(data_dir_path):
	for f in os.listdir(data_dir_path):
		print("data_dir_path:",data_dir_path)
		data_file_path = os.path.join(data_dir_path, f)
		if os.path.isfile(data_file_path):
			relight(data_file_path, data_dir_path, random.uniform(0.5, 1.5), random.randint(-50, 50))
		else:
			read_file_all_again(data_file_path)
            #文件夹下套文件夹情况
					
# 改变图片的亮度与对比度
def relight(image_path, save_dir, light=1, bias=0):
	img = Image.open(image_path)
	img = np.array(img)
	w = img.shape[1]
	h = img.shape[0]
    #image = []
	for i in range(w):
		for j in range(h):
			for c in range(3):  #三通道
				tmp = int(img[j, i, c]*light + bias)
				if tmp > 255:
					tmp = 255
				elif tmp < 0:
					tmp = 0
				img[j, i, c] = tmp
	img = Image.fromarray(img)
	save_path = save_dir + "/" + random_name() + '.bmp'
	img.save(save_path)
	print("save_path:",save_path)

def image_rotate(image_path,save_dir):
	#读取图像
	im = Image.open(image_path)
	im = im.transpose(Image.FLIP_LEFT_RIGHT)#左右互换
	save_path = save_dir + "/" + random_name() + '.bmp'
	im.save(save_path)
	print("save_path:",save_path)
	# img=np.array(Image.open(image_path)) 
    # #随机生成100个椒盐
    # rows,cols,dims=img.shape
    # for i in range(100):
        # x=np.random.randint(0,rows)
        # y=np.random.randint(0,cols)
        # img[x,y,:]=255
    # img.flags.writeable = True  # 将数组改为读写模式
    # dst=Image.fromarray(np.uint8(img))
    # save_path = save_dir + "/" + random_name() + '.bmp'
    # dst.save(save_path)
    # print("save_path:",save_path)
def random_name():
    #随机数，用来随机取名字
	a_list = ['0','1','2','3','4','5','6','7','8','9']
	name = random.sample(a_list,5)
	file_name = "".join(name)
	return file_name

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--path", required=True,
	help="path to input/output dir")
args = vars(ap.parse_args())
	
if __name__ == '__main__':
    #这里test_enlarge_faces文件夹下还有文件夹，每个文件夹下是同一个人的人脸
	data_dir_path = args["path"] #读取/保存的文件路径
	read_file_all(data_dir_path)
	read_file_all_again(data_dir_path)
