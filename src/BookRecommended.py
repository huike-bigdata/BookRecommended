from pandas import np

from DataLoader import DataLoader


def fit(save=True):
    dataLoader = DataLoader()
    # num: 获取的数据条数，决定了后边处理数据的时间，以及预测评分的时间
    ratings = dataLoader.getDataFrame("../data/BX-Book-Ratings.csv", ";", "utf-8", num=100)
    R = dataLoader.processDataFrametoArray(ratings)
    if save:
        np.save("../Model/BookRecommendedModel.npy", R)
    return R

# 核心算法
def LFM_grad_desc(R, K, max_iter, alpha=0.001, lamda=0.002):
    global cost
    M=len(R)
    N=len(R[0])

    #P、Q初始值，随机生成
    P=np.random.rand(M,K)
    Q=np.random.rand(N,K)
    Q=Q.T

    # 开始迭代
    for step in range(max_iter):
        # 对所有的用户u、物品i做遍历，对应的特征向量Pu、Qi梯度下降
        for u in range(M):
            for i in range(N):
                # 对于每一个大于0的评分，求出预测评分误差
                if R[u][i] > 0:
                    eui=np.dot(P[u,:],Q[:,i])-R[u][i]

                    # 带入公式，按照梯度下降算法更新当前的Pu和Qi
                    for k in range(K):
                        P[u][k]=P[u][k]-alpha*(2*eui*Q[k][i]+2*lamda*P[u][k])
                        Q[k][i]=Q[k][i]-alpha*(2*eui*P[u][k]+2*lamda*Q[k][i])

        # ui遍历完成，所有特征向量更新完成，可以得到P、Q，可以计算预测评分矩阵
        # 统计损失函数
        cost=0
        for u in range(M):
            for i in range(N):
                if R[u][i]>0:
                    cost+=(np.dot(P[u,:],Q[:,i])-R[u][i])**2
                    # 正则化项
                    for k in range(K):
                        cost+=lamda *(P[u][k]**2 + Q[k][i]**2)
        # 提前结束迭代
        if cost<0.0001:
            break

    return P,Q,cost

if __name__=="__main__":
    # 训练模型
    model=fit()
    print(model)

    # 使用模型

