
# coding: utf-8

# In[5]:


#coding=utf-8
import tensorflow as tf
x = tf.placeholder(tf.float32)
sess = tf.Session()
sess.run(tf.global_variables_initializer())
y = sess.run(x, feed_dict={x:0.0})
print(y)
sess.close


# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')
#setting plt
# 确定横纵坐标范围
plt.xlim(0,8)
plt.ylim(0,8)
# 确定横纵轴标题
plt.xlabel('x1')
plt.ylabel('x2')
# 确定该图的标题
plt.title('Discriminant interface')
# 把想画的散点按照横纵坐标输入进来
x=[0,2,2,0]
y=[0,0,2,2]
# bs是点的形状
plt.plot(x,y,'bs',color='red',label='w1')

x=[4,6,6,4]
y=[4,4,6,6]
# g^是点的形状
plt.plot(x,y,'g^',color='green',label='w2')
# 再画一条分类的线，线的范围是0-8
x= np.linspace(0,8)
plt.plot(x,6-x)
# 把点的标注协商
plt.legend(loc='upper left', shadow=True, fontsize='x-large')
plt.grid(True)
plt.savefig('./test.jpg')
plt.show()


# In[3]:


import numpy as np
import matplotlib.pyplot as plt
a = np.loadtxt('./data/Artificial Data/gauss1.txt')
a_split1 = a[:,0:2]
a_split2 = a[:,2]
plt.rcParams['figure.figsize']=(5.0,4.0)
plt.plot(a_split1[a_split2 >= 2,0:1], a_split1[a_split2 >= 2,1:2], 'bo')
plt.plot(a_split1[a_split2 < 2,0:1], a_split1[a_split2 < 2,1:2], 'ro')
plt.show()

