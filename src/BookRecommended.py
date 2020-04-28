import random
import time

from pandas import np

from DataLoader import DataLoader


class BookBookRecommendedSystem(object):
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

    def __init__(self, K):
        self.K = K
        self.R = None
        self.user_list = None
        self.ISBN_list = None
        self.predR = None

    def fit(self, saveModel=True, saveuser_list=True, saveISBN_list=True, rating_num=10):
        """
        加载并训练模型
        :param save:是否保存模型到本地
        :param saveuser_list:是否保存用户列表到本地
        :param saveISBN_list:是否保存书籍列表到本地
        :param rating_num:加载数据的条数
        :return:
        """
        dataLoader = DataLoader()
        # num: 获取的数据条数，决定了后边处理数据的时间，以及预测评分的时间
        ratings = dataLoader.getDataFrame("../data/BX-Book-Ratings.csv", ";", "utf-8", num=rating_num)
        self.R, self.user_list, self.ISBN_list = dataLoader.processDataFrametoArray(ratings)
        if saveModel:
            np.save("../Model/BookRecommendedModel.npy", self.R)
        if saveuser_list:
            with open("../Model/user_list", "w+", encoding="UTF-8") as f:
                f.write(str(self.user_list))
        if saveISBN_list:
            with open("../Model/ISBN_list", "w+", encoding="UTF-8") as f:
                f.write(str(self.ISBN_list))

    # 往模型添加新用户评价，用来预测其他未评价的评分
    def addRatings(self, ISBNS, ratings, user_id=88888888):
        """
        :param ISBNS:新用户的评价的书籍 列表
        :param ratings:新用户给书籍的评价 列表
        :param user_id:分配给新用户的id，存在默认值
        :return:
        """
        zers = np.zeros(shape=(1, self.R.shape[1]))
        for i in range(len(ratings)):
            print(type(ISBNS))
            zers[0][self.ISBN_list.index(ISBNS[i])] = ratings[i]
        self.R = np.append(self.R, zers, axis=0)
        self.user_list.append(user_id)

    # 核心算法,预测评分
    def LFM_grad_desc(self, max_iter, alpha=0.001, lamda=0.002):
        self.M = len(self.R)
        self.N = len(self.R[0])

        # self.P、Q初始值，随机生成
        self.P = np.random.rand(self.M, self.K)
        self.Q = np.random.rand(self.N, self.K)
        self.Q = self.Q.T
        # 开始迭代
        for step in range(max_iter):
            # 对所有的用户u、物品i做遍历，对应的特征向量Pu、Qi梯度下降
            for u in range(self.M):
                for i in range(self.N):
                    # 对于每一个大于0的评分，求出预测评分误差
                    if self.R[u][i] > 0:
                        eui = np.dot(self.P[u, :], self.Q[:, i]) - self.R[u][i]

                        # 带入公式，按照梯度下降算法更新当前的Pu和Qi
                        for k in range(self.K):
                            self.P[u][k] = self.P[u][k] - alpha * (2 * eui * self.Q[k][i] + 2 * lamda * self.P[u][k])
                            self.Q[k][i] = self.Q[k][i] - alpha * (2 * eui * self.P[u][k] + 2 * lamda * self.Q[k][i])

            # ui遍历完成，所有特征向量更新完成，可以得到P、self.Q，可以计算预测评分矩阵
            # 统计损失函数
            self.cost = 0
            for u in range(self.M):
                for i in range(self.N):
                    if self.R[u][i] > 0:
                        self.cost += (np.dot(self.P[u, :], self.Q[:, i]) - self.R[u][i]) ** 2
                        # 正则化项
                        for k in range(self.K):
                            self.cost += lamda * (self.P[u][k] ** 2 + self.Q[k][i] ** 2)
            # 提前结束迭代
            if self.cost < 0.0001:
                break
        self.predR = self.P.dot(self.Q)  # 预测结果（ndarry）

    def getTopRatings(self, user_id=88888888, topnum=3, duplicateremoval=True, ISBNS=None):
        """
        获取用户评分最高的书籍（预测后的）与对应的预测评分
        :param predR:
        :param user_list:所有用户的列表
        :param ISBN_list:所有书籍的ISBN列表
        :param user_id:要提取最高评分与对应的的用户的id
        :param topnum:提取数量
        :param duplicateremoval:是否去重
        :param ISBNS:新用户评价的书籍列表，用于去重
        :return:返回被找到的几个书ISBN号码列表，评分列表。两个列表长度相等，相同位置一一对应
        """

        # 根据用户id找到对应的行索引
        h_index = self.user_list.index(user_id)

        # 根据行索引找到行（评分）
        info = self.predR[h_index]

        # 评分排序之后从小到大返回原本的列索引组成一个列表
        sort_list = list(info.argsort())

        # 反转为评分从到大到小，取出了前几个的索引
        index = sort_list[::-1]

        # 保存被提取的最高分的几个书的ISBN编号
        ISBN_topN = []  # 存放TopN的书籍ISBN号
        index_a = []  # 存放TopN的书籍对应的索引
        for i in index:
            if len(ISBN_topN) < topnum:
                # 如果已保存的书籍仍小于想要提取的数量，则继续，否则退出循环
                if duplicateremoval:
                    if self.ISBN_list[i] in ISBNS:
                        # 如果开启了去重，且当前书籍在此用户已经评价的书籍的列表里面，则不予认定为操作者想要去除的
                        continue
                index_a.append(i)
                ISBN_topN.append(self.ISBN_list[i])
            else:
                break

        # ISBN列表与评分列表，相应位置一一对应
        return ISBN_topN, list(info[index_a])


if __name__ == "__main__":
    # 图书推荐系统
    BRS = BookBookRecommendedSystem(K=5)
    print("{} — 开始训练/读取已训练的模型".format(time.strftime('%Y.%m.%d %H:%M:%S', time.localtime(time.time()))))
    # 训练模型
    BRS.fit(rating_num=40)

    """
    这里可以手动从本地读取已训练好的模型与2个列表
    事例：
    new = np.load("../Model/BookRecommendedModel.npy")
    with open("../Model/user_list", "r") as f:
        user_list = eval(f.read())
    with open("../Model/ISBN_list", "r") as f:
        ISBN_list = eval(f.read())
    BRS.R = new
    BRS.user_list = user_list
    BRS.ISBN_list = ISBN_list
    """

    # new = np.load("../Model/BookRecommendedModel.npy")
    # with open("../Model/user_list", "r") as f:
    #     user_list = eval(f.read())
    # with open("../Model/ISBN_list", "r") as f:
    #     ISBN_list = eval(f.read())
    # BRS.R = new
    # BRS.user_list = user_list
    # BRS.ISBN_list = ISBN_list

    # print(BRS.R)        #输出模型

    # 模拟增加的新用户
    # newbooklist = ["034545104X", '0155061224', '0446520802', '052165615X', '0521795028']
    newbooklist = random.sample(BRS.ISBN_list, 5)
    newratinglist = [0, 5, 3, 7, 5]

    # 添加新用户
    BRS.addRatings(newbooklist, newratinglist)
    # print(BRS.R)

    print("{} — 开始预测".format(time.strftime('%Y.%m.%d %H:%M:%S', time.localtime(time.time()))))
    # 预测
    BRS.LFM_grad_desc(max_iter=1000, alpha=0.001, lamda=0.002)

    print(BRS.cost)  # 输出损失函数

    # print(BRS.predR)  # 输出预测值

    # 取出单用户预测的TopN评分
    ISBN_topN, Rating_topN = BRS.getTopRatings(topnum=5, ISBNS=newbooklist)
    print("{} — 预测完成".format(time.strftime('%Y.%m.%d %H:%M:%S', time.localtime(time.time()))))
    print("推荐您阅读：{}".format(ISBN_topN), Rating_topN)
