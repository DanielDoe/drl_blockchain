import numpy as np
import pandas as pd
import time

'''
# 创建DataFrame
# df = pd.DataFrame(np.random.randn(10, 4), index=np.arange(10), columns=list('ABCD'))
df = pd.DataFrame(np.random.randn(10, 4), columns=list('ABCD'))

# 根据index删除行
del_index = df.index[[0, 1, 3]]  # 删除第0,1,3行
df = df.drop(del_index)
# df.drop(index=[0, 1])

# 根据条件筛选行
df = df[(df['A'] > 0) & (df['D'] > 0)]

# 选取行
df.loc[0, :]

# 添加行
df2 = pd.DataFrame(np.random.rand(1, 4), index=10, columns=list('ABCD'))
df = df.append(df2)

# 两个DataFrame连接
df1 = pd.DataFrame(np.random.randn(3,4),columns=['a','b','c','d'])
df2 = pd.DataFrame(np.random.randn(2,3),columns=['b','d','a'])  
pd.concat([df1, df2], ignore_index=True)  # 通过参数ignore_index=True 重建索引
Out[8]:   
          a         b         c         d  
0 -0.848557 -1.163877 -0.306148 -1.163944  
1  1.358759  1.159369 -0.532110  2.183934  
2  0.532117  0.788350  0.703752 -2.620643  
3 -0.316156 -0.707832       NaN -0.416589  
4  0.406830  1.345932       NaN -1.874817 

# 根据列或者行进行排序
df.sort_index()
df.sort_values(by='A', ascending=False)
'''

def fun():
    return 1, 2, 3

def test():
    df = pd.DataFrame(columns=list('ABCD'))
    t1 = time.time()
    for i in range(10000):
        df = df.append(pd.DataFrame(data=np.random.randn(1, 4), columns=list('ABCD')), ignore_index=True)
    t2 = time.time()
    x = np.zeros([1, 4])
    for i in range(10000):
        x = np.concatenate((x, np.zeros([1, 4])), axis=0)
    t3 = time.time()
    time1 = t2 - t1
    time2 = t3 - t2
    print(f"DataFrame 耗时：{time1} 秒")
    print(f"np.array 耗时：{time2} 秒")


# def fun2(arr):
#     arr2 = arr
#     arr2[0] = 1


if __name__ == '__main__':
    # df2 = pd.DataFrame(index=list(range(4)), columns=list('ABCD'))
    # for index, row in df2.iterrows():
    #     w = row[1:].tolist()
    #     w.append(10)
    #     print([w])
    # print(df2['A'])
    # for index, value in enumerate(df2['A']):
    #     print(index, value)
    # x, y, z = fun()
    # print(x, y)
    # test()
    # x = np.zeros([4, 4])
    # for ri in range(x.shape[0]):
    #     print(ri, ' ', x[ri])
    # df = pd.DataFrame(data=np.ones([4, 4]), columns=list('ABCD'))
    # df2 = df.loc[[0, 2]]
    # for index, row in df2.iterrows():
    #     df.loc[index] = [1, 2, 3, 4]
    # print(df)
    # arr = np.zeros(4)
    # fun2(arr)
    # print(arr)
    # print(np.linspace(1, 100, 20))

    x, y, z = fun()
    print(x, y)

    # x = np.zeros(3)
    # y = np.ones(3)
    # z = x - y
    # if (z < 0).all():
    #     print(True)
    # else:
    #     print(False)



