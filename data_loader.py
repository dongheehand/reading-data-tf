
# coding: utf-8

# In[1]:


import tensorflow as tf
import numpy as np
import os
import glob
from PIL import Image


# In[ ]:


class data_loader():
    
    def __init__(self, args):
        
        self.in_memory = args.in_memory
        self.aug_num = args.aug_num
        self.batch_size = args.batch_size
        self.img_path = glob.glob(os.path.join(args.data_path, '*.png'))
    
    def build_loader(self):
        
        if self.in_memory:
            self.img_data = tf.placeholder(shape = [None, 224,224,3], dtype = tf.float32)
        else:
            self.img_data = np.array([self.img_path[0] for i in range(self.aug_num)])
        
        self.dataset = tf.data.Dataset.from_tensor_slices(self.img_data)
        
        if not self.in_memory:
            self.dataset = self.dataset.map(self._parse, num_parallel_calls = 4).prefetch(32)
            
        self.dataset = self.dataset.map(self._resize, num_parallel_calls = 4).prefetch(32)
        self.dataset = self.dataset.shuffle(32)
        self.dataset = self.dataset.repeat()
        self.dataset = self.dataset.batch(self.batch_size)        
        iterator = tf.data.Iterator.from_structure(self.dataset.output_types, self.dataset.output_shapes)
        self.next_batch = iterator.get_next()
        self.init_op = iterator.make_initializer(self.dataset)
        
    def _parse(self, image):
        
        image = tf.read_file(image)
        image = tf.image.decode_png(image, channels = 3)
        image = tf.cast(image, tf.float32)
        return image
    
    def _resize(self, image):
        
        image = tf.image.resize_images(image, (224, 224), tf.image.ResizeMethod.BICUBIC)
        
        return image
    


# In[ ]:


def image_loader(data_path, data_num):
    img_path = glob.glob(os.path.join(data_path, '*.png'))
    img = Image.open(img_path[0])
    img = np.array(img.resize((224,224)))
    img_list = [img for i in range(data_num)]
    return np.array(img_list)

def batch_loader(data_path, batch_size):
    img_list = []
    img_path = glob.glob(os.path.join(data_path, '*.png'))
    for i in range(batch_size):
        img = Image.open(img_path[0])
        img = np.array(img.resize((224,224)))
        img_list.append(img)
        
    return np.array(img_list)

