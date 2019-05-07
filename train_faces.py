import tensorflow as tf
import cv2
import numpy as np
import os
import random
import sys
from sklearn.model_selection import train_test_split

from datetime import datetime
TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())

# my_faces_path = './my_faces'
# # other_faces_path = './other_faces'
face_path = './faces'   #数据集路径
size = 64

imgs = []
labs = []

def readData(path):
    for file_dir_name in os.listdir(path):
        next_files = path + '/' + file_dir_name #左斜杠linux与windows都兼容
        for filename in os.listdir(next_files):
            if filename.endswith('.bmp') or filename.endswith('.BMP') or filename.endswith('.PNG') or filename.endswith('.png') or filename.endswith('.JPG') or filename.endswith('.jpg'):
			#三种图像格式,bmp/png/jpg
                filename = path + '/' + file_dir_name + '/' + filename

                img = cv2.imread(filename)

                imgs.append(img)
                labs.append(file_dir_name)

# readData(my_faces_path)
# readData(other_faces_path)
readData(face_path)
# 将图片数据与标签转换成数组
imgs = np.array(imgs)
#labs = np.array([[0, 1] if lab == my_faces_path else [1, 0] for lab in labs])
labs_mid = labs
labs_set = labs
image_numbers = len(labs)
faces_number = len(set(labs_set))
labs = np.zeros((image_numbers, faces_number), dtype=np.int)
labs_index = 0
for x in labs_mid:
    labs[labs_index][int(x)] = 1
    labs_index = labs_index + 1
# 随机划分测试集与训练集
train_x, test_x, train_y, test_y = train_test_split(imgs, labs, test_size=0.05, random_state=random.randint(0, 100))
# 参数：图片数据的总数，图片的高、宽、通道
train_x = train_x.reshape(train_x.shape[0], size, size, 3)
test_x = test_x.reshape(test_x.shape[0], size, size, 3)
# 将数据转换成小于1的数
train_x = train_x.astype('float32')/255.0
test_x = test_x.astype('float32')/255.0

print('train size:%s, test size:%s' % (len(train_x), len(test_x)))
# 图片块，每次取100张图片
batch_size = 100
num_batch = len(train_x) // batch_size

x = tf.placeholder(tf.float32, [None, size, size, 3])
y_ = tf.placeholder(tf.float32, [None, faces_number])
#这里也是根据识别人脸个数
keep_prob_5 = tf.placeholder(tf.float32)
keep_prob_75 = tf.placeholder(tf.float32)

v1 = tf.get_variable("v1", shape=[3], initializer=tf.zeros_initializer)  #测试参数传递
inc_v1 = v1.assign(v1+1)

def weightVariable(shape):
    init = tf.random_normal(shape, stddev=0.01)
    return tf.Variable(init)

def biasVariable(shape):
    init = tf.random_normal(shape)
    return tf.Variable(init)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

def maxPool(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

def dropout(x, keep):
    return tf.nn.dropout(x, keep)

def cnnLayer():
    # 第一层
    #W1 = weightVariable([3,3,3,32]) # 卷积核大小(3,3)， 输入通道(3)， 输出通道(32)
    W1 = tf.get_variable("W1", shape=[3,3,3,32], initializer=tf.random_normal_initializer(stddev=0.01))
    #b1 = biasVariable([32])
    b1 = tf.get_variable("b1", shape=[32], initializer=tf.random_normal_initializer)
    # 卷积
    conv1 = tf.nn.relu(conv2d(x, W1) + b1)
    # 池化
    pool1 = maxPool(conv1)
    # 减少过拟合，随机让某些权重不更新
    drop1 = dropout(pool1, keep_prob_5)

    # 第二层
    #W2 = weightVariable([3,3,32,64])
    W2 = tf.get_variable("W2", shape=[3, 3, 32, 64], initializer=tf.random_normal_initializer(stddev=0.01))
    #b2 = biasVariable([64])
    b2 = tf.get_variable("b2", shape=[64], initializer=tf.random_normal_initializer)
    conv2 = tf.nn.relu(conv2d(drop1, W2) + b2)
    pool2 = maxPool(conv2)
    drop2 = dropout(pool2, keep_prob_5)

    # 第三层
    #W3 = weightVariable([3,3,64,64])
    W3 = tf.get_variable("W3", shape=[3,3,64,64], initializer=tf.random_normal_initializer(stddev=0.01))
    #b3 = biasVariable([64])
    b3 = tf.get_variable("b3", shape=[64], initializer=tf.random_normal_initializer)
    conv3 = tf.nn.relu(conv2d(drop2, W3) + b3)
    pool3 = maxPool(conv3)
    drop3 = dropout(pool3, keep_prob_5)

    # 全连接层
    #Wf = weightVariable([8*8*64, 512])
    Wf = tf.get_variable("Wf", shape=[8*8*64, 512], initializer=tf.random_normal_initializer(stddev=0.01))
    #bf = biasVariable([512])
    bf = tf.get_variable("bf", shape=[512], initializer=tf.random_normal_initializer)
    drop3_flat = tf.reshape(drop3, [-1, 8*8*64])
    dense = tf.nn.relu(tf.matmul(drop3_flat, Wf) + bf)
    dropf = dropout(dense, keep_prob_75)

    # 输出层
    #Wout = weightVariable([512, 2])
    Wout = tf.get_variable("Wout", shape=[512, faces_number], initializer=tf.random_normal_initializer(stddev=0.01))
    #bout = biasVariable([2])
    bout = tf.get_variable("bout", shape=[faces_number], initializer=tf.random_normal_initializer)
    #out = tf.matmul(dropf, Wout) + bout
    out = tf.add(tf.matmul(dropf, Wout), bout)
    return out
#输出层个数根据标签决定
def cnnTrain():
    out = cnnLayer()

    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=out, labels=y_))
    #learning rate = 0.01
    train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)
    # 比较标签是否相等，再求的所有数的平均值，tf.cast(强制转换类型)
    #accuracy = tf.reduce_mean(tf.cast(tf.equal(out, y_), tf.float32))
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(out, 1), tf.argmax(y_, 1)), tf.float32))
    # 将loss与accuracy保存以供tensorboard使用
    tf.summary.scalar('loss', cross_entropy)
    tf.summary.scalar('accuracy', accuracy)
    merged_summary_op = tf.summary.merge_all()
    # 数据保存器的初始化
    saver = tf.train.Saver()

    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())

        inc_v1.op.run()  #测试参数传递

        train_writer = tf.summary.FileWriter('./tmp/train' + TIMESTAMP, graph=tf.get_default_graph())
        test_writer = tf.summary.FileWriter("./tmp/test" + TIMESTAMP, graph=tf.get_default_graph())

        for n in range(5000):
            for i in range(num_batch):
                batch_x = train_x[i*batch_size : (i+1)*batch_size]
                batch_y = train_y[i*batch_size : (i+1)*batch_size]
                # 开始训练数据，同时训练三个变量，返回三个数据
                _,loss,train_result = sess.run([train_step, cross_entropy, merged_summary_op],
                                           feed_dict={x:batch_x,y_:batch_y, keep_prob_5:0.5,keep_prob_75:0.75})
                train_writer.add_summary(train_result, n*num_batch+i)
                # 打印损失
                print(n*num_batch+i, "loss", loss)

                # if (n*num_batch+i) % 100 == 0:
                #     # 获取测试数据的准确率
                #     acc = accuracy.eval({x:test_x, y_:test_y, keep_prob_5:1.0, keep_prob_75:1.0})
                #     print(n*num_batch+i,"accuracy", acc)
                #     # 准确率大于0.98时保存并退出
                #     if acc > 0.98 and n > 2:
                #         saver.save(sess, './train_faces.model', global_step=n*num_batch+i)
                #         sys.exit(0)

                # n1 = n
                # num_batch1 = num_batch
                # i1 = i
                acc, test_result = sess.run([accuracy, merged_summary_op],
                                            feed_dict={x: test_x, y_: test_y, keep_prob_5: 1.0, keep_prob_75: 1.0})
                #acc = accuracy.eval({x: test_x, y_: test_y, keep_prob_5: 1.0, keep_prob_75: 1.0})
                test_writer.add_summary(test_result, n * num_batch + i)
                #saver.save(sess, './train_faces.model', global_step=n1 * num_batch1 + i1)
                print(n*num_batch+i, 'accuracy', acc)
                if acc > 0.94:
                    saver.save(sess, './tmp/model.ckpt')
                    sys.exit(0)

        saver.save(sess, './tmp/model.ckpt')


cnnTrain()
