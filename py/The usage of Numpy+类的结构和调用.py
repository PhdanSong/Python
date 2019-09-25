
# coding: utf-8

# In[2]:


import numpy as np


# In[10]:


#一维数组
my_array = np.array([1,2,3,4,5])
print(my_array) #[1 2 3 4 5]
print(my_array.shape)#(5,)一个包含5个元素的数组
print(my_array[0])# 1
my_new_array = np.zeros((5)) 
print(my_new_array)# [0. 0. 0. 0. 0.]一个包含5个0元素的数组
my_random_array = np.random.random((5))
print(my_random_array)#[0.58661046 0.12141738 0.94540112 0.84402769 0.46708468]一个包含5个随机数的数组，取值0-1


# In[18]:


#二维数组
my_2d_array = np.zeros((2, 3)) #创建2行3列全0数组
print(my_2d_array)
# [[0. 0. 0.]
#  [0. 0. 0.]]

my_2d_array_new = np.ones((2, 4)) #创建2行4列全1数组
print(my_2d_array_new)
# [[1. 1. 1. 1.]
#  [1. 1. 1. 1.]]

my_array = np.array([[4, 5], [6, 1]])
print(my_array[0][1]) #5
print(my_array.shape)#(2,2)

my_array_column_2 = my_array[:, 1] #取出所有行的第1列
print(my_array_column_2)#[5 1]
print(my_array_column_2.shape)#(2,)数组


# In[25]:


#数组操作
import numpy as np
a = np.array([[1.0,2.0],[3.0,4.0]])
b = np.array([[5.0,6.0],[7.0,8.0]])
sum = a + b
difference = a - b
product = a * b
quotient = a / b
print("sum = \n",sum)
# sum = 
#  [[ 6.  8.]
#  [10. 12.]]
print("difference = \n",difference)
# difference = 
#  [[-4. -4.]
#  [-4. -4.]]
print("Product = \n",product)
# Product = 
#  [[ 5. 12.]
#  [21. 32.]]
print("Quotient = \n",quotient)
# Quotient = 
#  [[0.2        0.33333333]
#  [0.42857143 0.5       ]]
matrix_product = a.dot(b) 
print("Matrix Product = ", matrix_product)
# Matrix Product =  [[19. 22.]
#  [43. 50.]]


# In[31]:


#创建数组
a = np.array([0, 1, 2, 3, 4])
b = np.array((0, 1, 2, 3, 4))
c = np.arange(5)
d = np.array([[11, 12, 13, 14, 15],
              [16, 17, 18, 19, 20],
              [21, 22, 23, 24, 25],
              [26, 27, 28 ,29, 30],
              [31, 32, 33, 34, 35]])
print(a)#[0 1 2 3 4]
print(a.shape)#(5,)
print(b)#[0 1 2 3 4]
print(b.shape)#(5,)
print(c)#[0 1 2 3 4]
print(c.shape)#(5,)
print(d)
# [[11 12 13 14 15]
#  [16 17 18 19 20]
#  [21 22 23 24 25]
#  [26 27 28 29 30]
#  [31 32 33 34 35]]
print(d.shape)#(5, 5)


# In[303]:


#多维数组切片
a = np.array([[11, 12, 13, 14, 15],
              [16, 17, 18, 19, 20],
              [21, 22, 23, 24, 25],
              [26, 27, 28 ,29, 30],
              [31, 32, 33, 34, 35]])
print(a[0, 1:4]) # [12 13 14]
print(a[1:4, 0]) # [16 21 26]
print(a[::2,::2]) #间隔为2
# [[11 13 15]
#  [21 23 25]
#  [31 33 35]]
print(a[:, 1]) # [12 17 22 27 32]
print(a[2:4,:])
# [[21 22 23 24 25]
#  [26 27 28 29 30]]
print(a[:,2:4])
# [[13 14]
#  [18 19]
#  [23 24]
#  [28 29]
#  [33 34]]
print(a[::-1,::-1])


# In[37]:


#数组属性
a = np.array([[11, 12, 13, 14, 15],
              [16, 17, 18, 19, 20],
              [21, 22, 23, 24, 25],
              [26, 27, 28 ,29, 30],
              [31, 32, 33, 34, 35]])

print(type(a)) # <class 'numpy.ndarray'>
print(a.dtype) # int64
print(a.size) # 25
print(a.shape) # (5, 5)
print(a.ndim) # 2


# In[47]:


#基本操作符
a = np.arange(9)
a = a.reshape((3,3))
b = np.array([10, 2, 1, 4, 2, 6, 7, 2, 1])
b = b.reshape((3,3))
print(a)
# [[0 1 2]
#  [3 4 5]
#  [6 7 8]]
print(b)#按行更改
# [[10  2  1]
#  [ 4  2  6]
#  [ 7  2  1]]
print(a + b)
# [[10  3  3]
#  [ 7  6 11]
#  [13  9  9]]
print(a - b)
# [[-10  -1   1]
#  [ -1   2  -1]
#  [ -1   5   7]]
print(a * b)
print(a / b)
print(a ** 2)
print(a.dot(b))#矩阵乘（点积）
# [[ 18   6   8]
#  [ 81  24  32]
#  [144  42  56]]


# In[49]:


#特殊运算符
a = np.arange(10)
print(a)
print(a.sum()) # >>>45
print(a.min()) # >>>0
print(a.max()) # >>>9
print(a.cumsum()) # >>>[ 0  1  3  6 10 15 21 28 36 45]第一个值为a[0],第二个值为a[0]+a[1],第三个值为a[0]+a[1]+a[2]……


# In[306]:


#索引
a = np.arange(0, 100, 10)
indices = [1, 5, -1]
print(type(indices))
b = a[indices]
print(a) # >>>[ 0 10 20 30 40 50 60 70 80 90]
print(b) # >>>[10 50 90]


# In[5]:


#布尔屏蔽
import matplotlib.pyplot as plt
import numpy as np
a = np.linspace(0, 2 * np.pi, 50)
b = np.sin(a)
plt.rcParams['figure.figsize']=(5.0,4.0)
# plt.rcParams['image.interpolation']='nearest'
# plt.rcParams['image.cmap']='gray'
plt.plot(a,b)
mask = (b >= 0)
print(mask)
# plt.plot(a[mask], b[mask], 'bo')
# mask = (b >= 0) & (a <= np.pi / 2)
# plt.plot(a[mask], b[mask], 'go')
plt.show()


# In[6]:


import numpy as np
import matplotlib.pyplot as plt

X_train = np.array([[1,2,3],[2,3,4],[2,2,1],[3,2,2]])
X_test = np.array([[1,1,1],[2,1,1]])

y_train = np.array([1,1,-1,1])
y_test = np.array([1,-1])

plt.rcParams['figure.figsize']=(5.0,4.0)
plt.plot(X_train[:,0],X_train[:,1],"ro")
plt.show()


# In[67]:


#缺省索引
a = np.arange(0, 100, 10)
b = a[:5]
c = a[a >= 50]
print(a) # [ 0 10 20 30 40 50 60 70 80 90]
print(b) # [ 0 10 20 30 40]
print(c) # [50 60 70 80 90]


# In[308]:


#Where 函数
a = np.arange(0, 100, 10)
a = np.array((10,50,70,20))
b = np.where(a < 50) #下标数组
c = np.where(a >= 50)[0] #下标
print(a) # [ 0 10 20 30 40 50 60 70 80 90]
print(b) # (array([0, 1, 2, 3, 4]),)
print(c) # [5 6 7 8 9]


# In[76]:


#数字类型
x = 3
print(type(x)) # Prints "<class 'int'>"
y = 2.5
print(type(y)) # Prints "<class 'float'>"
#Booleans(布尔类型)
t = True
f = False
print(type(t)) # Prints "<class 'bool'>"
print(t and f) # Logical AND; prints "False"
print(t or f)  # Logical OR; prints "True"
print(not t)   # Logical NOT; prints "False"
print(t != f)  # Logical XOR; prints "True"
# Strings(字符串类型)
hello = 'hello'  
world = "world"    
print(hello)       # "hello"
print(len(hello))  # "5"
hw = hello + ' ' + world 
print(hw)  # "hello world"
hw12 = '%s %s %d' % (hello, world, 12) 
print(hw12)  # "hello world 12"

s = "hello"
print(s.capitalize())  # Capitalize a string; prints "Hello"
print(s.upper())       # 大写 "HELLO"
print(s.rjust(7))      # 靠右，prints "  hello"
print(s.center(7))     # 中间，prints " hello "
print(s.replace('l', '(ell)'))  # 将l换成ell  "he(ell)(ell)o"
print('  world '.strip())  #删除空格，prints "world"


# In[78]:


#列表list
xs = [3, 1, 2]    # 列表
print(xs, xs[2])  # "[3, 1, 2] 2"
print(xs[-1])     #  "2"
xs[2] = 'foo'     # 列表中元素类型可以不同
print(xs)         # [3, 1, 'foo']
xs.append('bar')  # 末尾添加
print(xs)         # [3, 1, 'foo', 'bar']"
x = xs.pop()      # 末尾去除
print(x, xs)      # bar [3, 1, 'foo']"

#list切片
nums = list(range(5))     # list 
print(nums)               # [0, 1, 2, 3, 4]
print(nums[2:4])          # [2, 3]
print(nums[2:])           # [2, 3, 4]
print(nums[:2])           # [0, 1]
print(nums[:])            # [0, 1, 2, 3, 4]
print(nums[:-1])          # [0, 1, 2, 3]
nums[2:4] = [8, 9] 
print(nums)               #[0, 1, 8, 9, 4]

#循环
animals = ['cat', 'dog', 'monkey']
for animal in animals:
    print(animal)
# cat
# dog
# monkey

#加索引
animals = ['cat', 'dog', 'monkey']
for idx, animal in enumerate(animals):
    print('%d: %s' % (idx + 1, animal))
#1: cat
#2: dog
#3: monkey

#声明空列表a = [],声明空字典b = {}

#列表推导式
nums = [0, 1, 2, 3, 4]
squares = []
for x in nums:
    squares.append(x ** 2)
print(squares)   # Prints [0, 1, 4, 9, 16]

#等价于

nums = [0, 1, 2, 3, 4]
squares = [x ** 2 for x in nums]
print(squares)   # Prints [0, 1, 4, 9, 16]

#包含if条件
nums = [0, 1, 2, 3, 4]
even_squares = [x ** 2 for x in nums if x % 2 == 0]
print(even_squares)  # Prints "[0, 4, 16]"


# In[96]:


#字典
d = {'cat': 'cute', 'dog': 'furry'} 
print(d['cat'])       # "cute"
print('cat' in d)     # "True"
d['fish'] = 'wet'     # 添加
print(d['fish'])      # "wet"
print(d.get('monkey', 'N/A'))  # N/A"
print(d.get('fish', 'N/A'))    # "wet"
del d['fish']         # 删除
print(d.get('fish', 'N/A')) # "N/A"

#循环1
d = {'person': 2, 'cat': 4, 'spider': 8}
for animal in d:
    legs = d[animal] #得到键
    print('A %s has %d legs' % (animal, legs))
# A person has 2 legs
# A cat has 4 legs
# A spider has 8 legs  

#循环2
d = {'person': 2, 'cat': 4, 'spider': 8}
for animal, legs in d.items():#得到键值对
    print('A %s has %d legs' % (animal, legs))
# A person has 2 legs
# A cat has 4 legs
# A spider has 8 legs

#函数推导式
nums = [0, 1, 2, 3, 4]
even_num_to_square = {x: x ** 2 for x in nums if x % 2 == 0}
print(even_num_to_square)  
# Prints "{0: 0, 2: 4, 4: 16}"


# In[101]:


#集合
animals = {'cat', 'dog'}
print('cat' in animals)   # True
print('fish' in animals)  # False
animals.add('fish')       
print('fish' in animals)  # True
print(len(animals))       # 3
animals.add('cat')        # do nothing
print(len(animals))       # 3
animals.remove('cat')     # 删除
print(len(animals))       # 2

#循环
animals = {'cat', 'dog', 'fish'}
for idx, animal in enumerate(animals):
    print('%d: %s' % (idx + 1, animal))#无序
# 1: dog
# 2: cat
# 3: fish


# In[104]:


#元组
d = {(x, x + 1): x for x in range(10)}  # Create a dictionary with tuple keys
print(d)
#{(0, 1): 0, (1, 2): 1, (2, 3): 2, (3, 4): 3, (4, 5): 4, (5, 6): 5, (6, 7): 6, (7, 8): 7, (8, 9): 8, (9, 10): 9}
t = (5, 6)        # Create a tuple
print(type(t))    # Prints "<class 'tuple'>"
print(d[t])       # Prints "5"
print(d[(1, 2)])  # Prints "1"


# In[103]:


#类
class Greeter(object):

    # Constructor
    def __init__(self, name):
        self.name = name  # Create an instance variable

    # Instance method
    def greet(self, loud=False):
        if loud:
            print('HELLO, %s!' % self.name.upper())
        else:
            print('Hello, %s' % self.name)

g = Greeter('Fred')  # 生成对象
g.greet()            # "Hello, Fred"
g.greet(loud=True)   # "HELLO, FRED!"


# In[105]:


#创建数组
import numpy as np

a = np.zeros((2,2))  
print(a)             
# [[0. 0.]
#  [0. 0.]]                   
b = np.ones((1,2))   
print(b)              
#[[1. 1.]]
c = np.full((2,2), 7)  
print(c)               
# [[7 7]
#  [7 7]]                     
d = np.eye(2)         
print(d)            
# [[1. 0.]
#  [0. 1.]]                    
e = np.random.random((2,2))  
print(e)                     
# [[0.14483878 0.93726129]
#  [0.787366   0.46925537]]


# In[110]:


a = np.array([[1,2,3], [4,5,6], [7,8,9], [10, 11, 12]])

  # prints "array([[ 1,  2,  3],
          #                [ 4,  5,  6],
          #                [ 7,  8,  9],
          #                [10, 11, 12]])"

# Create an array of indices
b = np.array([0, 2, 0, 1])
print(np.arange(4),b) #[0 1 2 3] [0 2 0 1]
# Select one element from each row of a using the indices in b
print(a[np.arange(4), b])  # Prints "[ 1  6  7 11]"


# In[119]:


#在numpy数组中只打印小数点后三位？
# Input
rand_arr = np.random.random((5,3))
print(rand_arr)
# [[0.161 0.211 0.163]
#  [0.869 0.714 0.712]
#  [0.867 0.982 0.88 ]
#  [0.155 0.92  0.461]
#  [0.855 0.305 0.004]]

rand_arr = np.random.random([5,3])
print(rand_arr)
# [[0.992 0.329 0.845]
#  [0.843 0.929 0.286]
#  [0.976 0.134 0.961]
#  [0.658 0.565 0.322]
#  [0.331 0.052 0.827]]

# Limit to 3 decimal places
np.set_printoptions(precision=1)  
rand_arr[:5]  #前5行
# array([[0.161, 0.609, 0.748],
#        [0.411, 0.405, 0.271],
#        [0.49 , 0.096, 0.588],
#        [0.584, 0.018, 0.18 ]])


# In[302]:


#导入iris数据集
from sklearn.datasets import load_iris
iris = load_iris()
data = iris['data']
# print(data)
target = iris['target']
# print(target)

#求出鸢尾属植物萼片长度的平均值、中位数和标准差(第1列)
mu, med, sd = np.mean(data[:,0]), np.median(data[:,0]), np.std(data[:,0])
# print(mu, med, sd)

#创建一种标准化形式的鸢尾属植物间隔长度，其值正好介于0和1之间，这样最小值为0，最大值为1。
Smax, Smin = data[:,0].max(), data[:,0].min()
S = (data[:,0] - Smin)/(Smax - Smin)
# print(S)

#第4列中查找第一次出现的值大于1.0的位置。
np.argwhere(data[:,3].astype(float) > 1.0)[0]  #array([50])

vals, counts = np.unique(data[:, 2], return_counts=True)
# print(vals[np.argmax(counts)])#np.unique存放的只有下标，没有值

#根据data[0]列对所有数据进行排序。
# print(data[data[:,0].argsort()][:20])

#第二长的物种setosa的价值是多少
petal_len_setosa = data[51:100,2:3].astype('float')
# print(np.unique(np.sort(petal_len_setosa,axis = 0)))
# [3.  3.3 3.5 3.6 3.7 3.8 3.9 4.  4.1 4.2 4.3 4.4 4.5 4.6 4.7 4.8 4.9 5. 5.1]
# print("\n")
# print(np.unique(np.sort(petal_len_setosa,axis = 0))[-2])#取第二大元素
#声明 return_index=True时，保存的下标，默认时元素

#随机取样
a1 = np.random.choice(a=5, size=3, replace=False, p=None)
# print("a1",a1)#在0-5之间取3个数
np.random.choice((50), 20)

#在data中第0行查找缺失值的数量和位置（第1列）
# print("Number of missing values: \n", np.isnan(data[:, 0]).sum())
# print("Position of missing values: \n", np.where(np.isnan(data[:, 0])))

#petallength（第3列）> 1.5 和 sepallength（第1列）< 5.0 的iris_2d行
condition = (data[:, 2] > 1.5) & (data[:, 0] < 5.0)
data[condition]

#选择没有任何nan值的data行。
data[np.random.randint(150, size=20), np.random.randint(4, size=20)] = np.nan
#nu.sum(bool)时，true为1，false为0
data[np.sum(np.isnan(data), axis = 1) == 0][:5]
# array([[5.1, 3.5, 1.4, 0.2],
#        [4.9, 3. , 1.4, 0.2],
#        [4.7, 3.2, 1.3, 0.2],
#        [4.6, 3.1, 1.5, 0.2],
#        [5. , 3.6, 1.4, 0.2]])

#两列之间的相关性
data = iris['data']
print(np.corrcoef(data[:, 0], data[:, 2])[0, 1])

np.isnan(data).any()#any全为假则假，有真为真

#numpy数组中用0替换所有缺失值
data[np.isnan(data)] = 0


# In[141]:


#从数组a中，替换所有大于30到30和小于10到10的值。
# Input
np.set_printoptions(precision=2)
np.random.seed(100)
a = np.random.uniform(1,50, 20)#均匀分布随机采样，1-50中采20个数
print(a,"\n")

b = np.where(a > 30, 30, a)


print(np.where(a < 10, 10,b))# np.where(condition, x, y)
# > [ 27.63  14.64  21.8   30.    10.    10.    30.    30.    10.    29.18  30.
# >   11.25  10.08  10.    11.77  30.    30.    10.    30.    14.43]


# In[154]:


#获取给定数组a中前5个最大值的位置。
# Input
np.random.seed(100)
a = np.random.uniform(1,50, 20)
print("a",a)
# Solution:
print("a.argsort",a.argsort())#升序排序，输出类标
print(a.argsort()[-5:])

print("value",a[a.argsort()][-5:])


# In[155]:


#将array_of_arrays转换为扁平线性1d数组。
arr1 = np.arange(3)
arr2 = np.arange(3,7)
arr3 = np.arange(7,10)
print(arr1)
print(arr2)
print(arr3)
array_of_arrays = np.array([arr1, arr2, arr3])
arr_2d = np.concatenate(array_of_arrays)
print(arr_2d)


# In[159]:


#one-hot
np.random.seed(101) 
arr = np.random.randint(1,4, size=6)
print(arr)
# > array([2, 3, 2, 2, 2, 1])

def one_hot_encodings(arr):
    uniqs = np.unique(arr)
    print(uniqs) #类似于set
    out = np.zeros((arr.shape[0], uniqs.shape[0]))
    for i, k in enumerate(arr):
        out[i, k-1] = 1
    return out

print(one_hot_encodings(arr))


# In[162]:


#为给定的数字数组a创建排名。
np.random.seed(10)
a = np.random.randint(20, size=10)
print('Array: ', a)

# Solution
print(a.argsort())#对数组a的元素升序排序，输出下标
print(a.argsort().argsort())#对下标数组排序
print('Array: ', a)


# In[168]:


#创建与给定数字数组a相同形状的排名数组。
np.random.seed(10)
a = np.random.randint(20, size=[2,5])
print(a)
print(a.ravel())#降维
print(a.ravel().argsort().argsort().reshape(a.shape))


# In[171]:


#计算给定数组中每行的最大值和最小值。
# Input
np.random.seed(100)
a = np.random.randint(1,10, [5,3])
print(a)

print(np.amax(a, axis=1))
print(np.amin(a, axis=1))


# In[183]:


#在给定的numpy数组中找到重复的条目(第二次出现以后)，并将它们标记为True。第一次出现应该是False的。
# Input
np.random.seed(100)
a = np.random.randint(0, 5, 10)
print("a",a)

out = np.full(a.shape[0], True)
print("out",out)
unique_positions = np.unique(a, return_index=True)[1]
c,s = np.unique(a, return_index=True)
print("c",c)#c是不重复元素
print("s",s)#s是对应下标
print("unique_positions",unique_positions)
out[unique_positions] = False
print(out)


# In[185]:


#从一维numpy数组中删除所有NaN值
a = np.array([1,2,3,np.nan,5,6,7,np.nan])
a[~np.isnan(a)]
print(a[~np.isnan(a)])


# In[186]:


#计算两个数组a和数组b之间的欧氏距离。
a = np.array([1,2,3,4,5])
b = np.array([4,5,6,7,8])
dist = np.linalg.norm(a-b)
print(dist)


# In[197]:


#找到一个一维数字数组a中的所有峰值。峰顶是两边被较小数值包围的点。
a = np.array([1, 3, 7, 1, 2, 6, 0, 1])
doublediff = np.diff(np.sign(np.diff(a)))#np.diff是后一个元素-前一个元素
print(np.diff(a))
print(doublediff)
peak_locations = np.where(doublediff == -2)[0] + 1
print(np.where(doublediff == -2))#(array([1, 4]),)
peak_locations
# > array([2, 5])


# In[202]:


# 从二维数组中减去一维数组，其中一维数组的每一项从各自的行中减去？
a_2d = np.array([[3,3,3],[4,4,4],[5,5,5]])
b_1d = np.array([1,2,3]).reshape(3,1)
print(b_1d)
print(a_2d - b_1d)


# In[205]:


#找出x中数字1的第5次重复的索引。
x = np.array([1, 2, 1, 1, 3, 4, 3, 1, 1, 2, 1, 1, 2])
n = 5

# Solution 2: Numpy version
np.where(x == 1)[0][n-1]
print(np.where(x == 1))#(array([ 0,  2,  3,  7,  8, 10, 11]),)输出元素为1的下标，是一个tuple，得加[0]，构成数组
# > 8


# In[208]:


#将numpy的datetime64对象转换为datetime的datetime对象
dt64 = np.datetime64('2018-02-25 22:10:10')
dt64.tolist() #  datetime.datetime(2018, 2, 25, 22, 10, 10)


# In[213]:


#对于给定的一维数组，计算窗口大小为3的移动平均值。
def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)#累加第i个元素=a[0]+a[1]+……+a[i-1]
    print("ret",ret)
    print("ret[n:]",ret[n:])#下标从n开始，一直到结尾
    print("ret[:-n]",ret[:-n])
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

np.random.seed(100)
Z = np.random.randint(10, size=10)
print('array: ', Z)
moving_average(Z, n=3).round(2)


# In[214]:


#创建长度为10的numpy数组，从5开始，在连续的数字之间的步长为3。
length = 10
start = 5
step = 3

def seq(start, length, step):
    end = start + (step*length)
    return np.arange(start, end, step)

seq(start, length, step)


# In[215]:


#从给定的一维数组arr中，利用步进生成一个二维矩阵，窗口长度为4，步距为2，类似于 [[0,1,2,3], [2,3,4,5], [4,5,6,7]..]
def gen_strides(a, stride_len=5, window_len=5):
    n_strides = ((a.size-window_len)//stride_len) + 1
    # return np.array([a[s:(s+window_len)] for s in np.arange(0, a.size, stride_len)[:n_strides]])
    return np.array([a[s:(s+window_len)] for s in np.arange(0, n_strides*stride_len, stride_len)])

print(gen_strides(np.arange(15), stride_len=2, window_len=4))


# In[230]:


#给定一系列不连续的日期序列。填写缺失的日期，使其成为连续的日期序列。
dates = np.arange(np.datetime64('2018-02-01'), np.datetime64('2018-02-25'), 2)
print(dates)
# ['2018-02-01' '2018-02-03' '2018-02-05' '2018-02-07' '2018-02-09'
#  '2018-02-11' '2018-02-13' '2018-02-15' '2018-02-17' '2018-02-19'
#  '2018-02-21' '2018-02-23']
print(np.diff(dates))#dates元素12个，diff11个元素
# [2 2 2 2 2 2 2 2 2 2 2]
filled_in = np.array([np.arange(date, (date+d)) for date, d in zip(dates, np.diff(dates))]).reshape(-1) #reshape(-1)一维数组，reshape(-1,1)1列n行
print("filled_in_preReshape",np.array([np.arange(date, (date+d)) for date, d in zip(dates, np.diff(dates))]))
# filled_in [['2018-02-01' '2018-02-02']
#  ['2018-02-03' '2018-02-04']
#  ['2018-02-05' '2018-02-06']
#  ['2018-02-07' '2018-02-08']
#  ['2018-02-09' '2018-02-10']
#  ['2018-02-11' '2018-02-12']
#  ['2018-02-13' '2018-02-14']
#  ['2018-02-15' '2018-02-16']
#  ['2018-02-17' '2018-02-18']
#  ['2018-02-19' '2018-02-20']
#  ['2018-02-21' '2018-02-22']]
print("filled_in",filled_in)
#  ['2018-02-01' '2018-02-02' '2018-02-03' '2018-02-04' '2018-02-05'
#  '2018-02-06' '2018-02-07' '2018-02-08' '2018-02-09' '2018-02-10'
#  '2018-02-11' '2018-02-12' '2018-02-13' '2018-02-14' '2018-02-15'
#  '2018-02-16' '2018-02-17' '2018-02-18' '2018-02-19' '2018-02-20'
#  '2018-02-21' '2018-02-22']
output = np.hstack([filled_in, dates[-1]])#加上最后一个
print("output",output)

