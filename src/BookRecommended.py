from DataLoader import DataLoader

dataLoader = DataLoader()
# num: 获取的数据条数，决定了后边处理数据的时间，以及计算时间
ratings = dataLoader.getDataFrame("../data/BX-Book-Ratings.csv", ";", "utf-8", num=100)
R = dataLoader.processDataFrametoArray(ratings)
print(R)