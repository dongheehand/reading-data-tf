
# coding: utf-8

# In[1]:


import tensorflow as tf
from vgg19 import Vgg19
import argparse
import time
from PIL import Image
from data_loader import *


# In[ ]:


def str2bool(v):
    return v.lower() in ('true')

parser = argparse.ArgumentParser()

parser.add_argument('--data_path', type = str, default = './test_image/')
parser.add_argument('--vgg_path', type = str, default = './vgg19/vgg19.npy')
parser.add_argument('--in_memory', type = str2bool, default = False)
parser.add_argument('--pipe_lining', type = str2bool, default = False)
parser.add_argument('--aug_num', type = int, default = 1000)
parser.add_argument('--batch_size', type = int, default = 32)
parser.add_argument('--test_step', type = int, default = 50)
args = parser.parse_args()


# In[ ]:


vgg19_net = Vgg19(args.vgg_path)

result_time = []

if args.pipe_lining:
    
    loader = data_loader(args)
    loader.build_loader()
    
    if args.in_memory:
        img_arr = image_loader(args.data_path, args.aug_num)
        vgg19_net.build(loader.next_batch)
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        _ = sess.run(loader.init_op, feed_dict = {loader.img_data : img_arr})
        
        for i in range(args.test_step):
            s_time = time.time()
            sess.run(vgg19_net.prob)
            e_time = time.time()
            result_time.append((e_time - s_time))
        
        result = result_time[args.test_step // 2:]
        print('Elpased time with pipelining : %4f' % (sum(result) / len(result)))

    else:
        vgg19_net.build(loader.next_batch)
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        sess.run(loader.init_op)
        
        for i in range(args.test_step):
            s_time = time.time()
            sess.run(vgg19_net.prob)
            e_time = time.time()
            result_time.append((e_time - s_time))
            
        result = result_time[args.test_step // 2:]
        print('Elpased time with pipelining : %4f' % (sum(result) / len(result)))
        
else:
    if args.in_memory:
        img_arr = image_loader(args.data_path, args.aug_num)
        input_image = tf.placeholder(shape = [args.batch_size, 224, 224, 3], dtype = tf.float32)
        vgg19_net.build(input_image)
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        
        for i in range(args.test_step):
            s_time = time.time()
            sess.run(vgg19_net.prob, feed_dict = {input_image : img_arr[:args.batch_size]})
            e_time = time.time()
            result_time.append((e_time - s_time))
        
        result = result_time[args.test_step // 2:]
        print('Elpased time without pipelining : %4f' % (sum(result) / len(result)))

    else:
        input_image = tf.placeholder(shape = [args.batch_size, 224, 224, 3], dtype = tf.float32)
        vgg19_net.build(input_image)
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        
        for i in range(args.test_step):
            s_time = time.time()
            img_arr = batch_loader(args.data_path, args.batch_size)
            sess.run(vgg19_net.prob, feed_dict = {input_image : img_arr})
            e_time = time.time()
            result_time.append((e_time - s_time))
            
        result = result_time[args.test_step // 2:]
        print('Elpased time without pipelining : %4f' % (sum(result) / len(result)))

