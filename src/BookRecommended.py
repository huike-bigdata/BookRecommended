from pandas import np

from DataLoader import DataLoader


def fit(saveModel=True, saveuser_list=True, saveISBN_list=True):
    """
    加载并训练模型
    :param save:是否保存模型到本地
    :return:
    """
    dataLoader = DataLoader()
    # num: 获取的数据条数，决定了后边处理数据的时间，以及预测评分的时间
    ratings = dataLoader.getDataFrame("../data/BX-Book-Ratings.csv", ";", "utf-8", num=10)
    R, user_list, ISBN_list = dataLoader.processDataFrametoArray(ratings)
    if saveModel:
        np.save("../Model/BookRecommendedModel.npy", R)
    if saveuser_list:
        np.save("../Model/user_list.npy", R)
    return R, user_list, ISBN_list


# 往模型添加用户评价，预测其他未评价的评分
def addRatings(R, user_list, ISBN_list, ISBNS, ratings, user_id=88888888):
    """

    :param R:加载并训练好的模型
    :param user_list:已存在的用户列表
    :param ISBN_list:已存在的书籍列表
    :param ISBNS:新用户的评价的书籍 列表
    :param ratings:新用户给书籍的评价 列表
    :param user_id:分配给新用户的id，存在默认值
    :return:
    """
    zers = np.zeros(shape=(1, R.shape[1]))
    for i in range(len(ratings)):
        zers[0][ISBN_list.index(ISBNS[i])] = ratings[i]
    R = np.append(R, zers, axis=0)
    user_list.append(user_id)

    return R


# 核心算法,预测评分
def LFM_grad_desc(R, K, max_iter, alpha=0.001, lamda=0.002):
    global cost
    M = len(R)
    N = len(R[0])

    # P、Q初始值，随机生成
    P = np.random.rand(M, K)
    Q = np.random.rand(N, K)
    Q = Q.T

    # 开始迭代
    for step in range(max_iter):
        # 对所有的用户u、物品i做遍历，对应的特征向量Pu、Qi梯度下降
        for u in range(M):
            for i in range(N):
                # 对于每一个大于0的评分，求出预测评分误差
                if R[u][i] > 0:
                    eui = np.dot(P[u, :], Q[:, i]) - R[u][i]

                    # 带入公式，按照梯度下降算法更新当前的Pu和Qi
                    for k in range(K):
                        P[u][k] = P[u][k] - alpha * (2 * eui * Q[k][i] + 2 * lamda * P[u][k])
                        Q[k][i] = Q[k][i] - alpha * (2 * eui * P[u][k] + 2 * lamda * Q[k][i])

        # ui遍历完成，所有特征向量更新完成，可以得到P、Q，可以计算预测评分矩阵
        # 统计损失函数
        cost = 0
        for u in range(M):
            for i in range(N):
                if R[u][i] > 0:
                    cost += (np.dot(P[u, :], Q[:, i]) - R[u][i]) ** 2
                    # 正则化项
                    for k in range(K):
                        cost += lamda * (P[u][k] ** 2 + Q[k][i] ** 2)
        # 提前结束迭代
        if cost < 0.0001:
            break

    return P, Q, cost


def getTopRatings(predR, user_id=88888888, topnum=10, duplicateremoval=True, ISBNS=None):
    """
    获取用户评分最高的书籍（预测后）
    """
    print(predR[predR["User-ID"] == user_id])
    pass


if __name__ == "__main__":
    """
    M：用户数、N：物品数
    @输入参数：
    R：M*N的评分矩阵
    K：隐特征向量维度
    max_iter：最大迭代次数
    alpha：步长
    lamda：正则化系数
    @输出：
    分解之后的P，Q
    P：初始化用户特征矩阵M*K
    Q：初始化物品特征矩阵N*K
    """

    # 训练模型
    model, user_list, ISBN_list = fit()
    print(model)
    # print(type(model))
    # # user_list
    # print(user_list)
    # print(type(user_list))
    # print(ISBN_list)
    # print(type(ISBN_list))

    # 使用模型
    newbooklist = ["034545104X", '0155061224', '0446520802', '052165615X', '0521795028']
    newratinglist = [0, 2, 3, 0, 5]

    # 添加新用户
    R = addRatings(model, user_list, ISBN_list, newbooklist, newratinglist)
    print(R)

    # 预测
    P,Q,cost=LFM_grad_desc(R, K=5, max_iter=500, alpha=0.001, lamda=0.002)
    print(P)
    print(Q)
    print(cost)
    predR = P.dot(Q)
    print(R)
    print(predR)

