#学习目标1:使用CountVectorizer和TfidfVectorizer对非结构化的符号化数据(如一系列字符串)进行特征抽取和向量化
from sklearn.datasets import fetch_20newsgroups
#从互联网上即时下载新闻样本，subset = 'all'参数表示下载全部近2万条文本文件
# subset : 'train' or 'test', 'all', optional
# Select the dataset to load: 'train' for the training set, 'test'
# for the test set, 'all' for both, with shuffled ordering.
news = fetch_20newsgroups(subset='all')
#分割数据集
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(news.data, news.target, test_size=0.25, random_state=33)
#用CountVectorizer提取特征
from sklearn.feature_extraction.text import CountVectorizer
count_vec = CountVectorizer()
X_count_train = count_vec.fit_transform(X_train)
X_count_test = count_vec.transform(X_test)
#使用朴素贝叶斯分类器来训练模型并预测
from sklearn.naive_bayes import MultinomialNB
mnb_count = MultinomialNB()
mnb_count.fit(X_count_train, y_train)
y_count_predict = mnb_count.predict(X_count_test)
print('The Accuracy of mnb(CountVectorizer) is', mnb_count.score(X_count_test, y_test))
from sklearn.metrics import classification_report
print(classification_report(y_test,y_count_predict,target_names=news.target_names))
#由输出结果可知，使用CountVectorizer在不去掉停用词的条件下，使用默认配置的朴素贝叶斯分类器，可以得到83.977%的预测准确性

#对比使用TfidfVectorizer且不去掉停用词的条件下，对文本特征进行量化
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_vec = TfidfVectorizer()
X_tfidf_train = tfidf_vec.fit_transform(X_train)
X_tfidf_test = tfidf_vec.transform(X_test)
#使用朴素贝叶斯分类器来训练模型并预测
mnb_tfidf= MultinomialNB()
mnb_tfidf.fit(X_tfidf_train,y_train)
y_tfidf_predict = mnb_tfidf.predict(X_tfidf_test)
print('The Accuracy of mnb(CountVectorizer) is', mnb_tfidf.score(X_tfidf_test, y_test))
from sklearn.metrics import classification_report
print(classification_report(y_test, y_tfidf_predict, target_names=news.target_names))
#由输出结果可知，使用TfidfVectorizer在不去掉停用词的条件下，使用默认配置的朴素贝叶斯分类器，可以得到84.634%的预测准确性

#这说明，在训练文本较多的时候，利用TfidfVectorizer压制这些常用词汇对分类决策的干扰，往往可以起到提升模型性能的作用。

#***************************************************************************************************

#学习目标2:在去掉停用词的前提下，分别使用CountVectorizer和TfidfVectorizer对文本特征进行量化，再用朴素贝叶斯进行训练评估
count_filter_vec,tfidf_filter_vec = CountVectorizer(analyzer='word', stop_words='english'),\
                                    TfidfVectorizer(analyzer='word', stop_words='english')
#使用带有停用词过滤的CountVectorizer和TfidfVectorizer对训练文本和测试文本进行量化处理
X_count_filter_train = count_filter_vec.fit_transform(X_train)
X_count_filter_test = count_filter_vec.transform(X_test)
X_tfidf_filter_train = tfidf_filter_vec.fit_transform(X_train)
X_tfidf_filter_test = tfidf_filter_vec.transform(X_test)
#使用默认配置的朴素贝叶斯分类器进行训练和评估
#1.CountVectorizer with filtering stopwords:
mnb_count.fit(X_count_filter_train,y_train)
y_count_filter_predict = mnb_count.predict(X_count_filter_test)
print('The Accuracy of mnb(CountVectorizer) is', mnb_count.score(X_count_filter_test,y_test))
from sklearn.metrics import classification_report
print(classification_report(y_test,y_count_filter_predict, target_names=news.target_names))
#由输出结果可知，使用CountVectorizer在去掉停用词的条件下，使用默认配置的朴素贝叶斯分类器，可以得到86.375%的预测准确性
#2.TfidfVectorizer with filtering stopwords:
mnb_tfidf.fit(X_tfidf_filter_train,y_train)
y_tfidf_filter_predict = mnb_tfidf.predict(X_tfidf_filter_test)
print('The Accuracy of mnb(CountVectorizer) is', mnb_tfidf.score(X_tfidf_filter_test, y_test))
from sklearn.metrics import classification_report
print(classification_report(y_test, y_tfidf_filter_predict,target_names=news.target_names))
#由输出结果可知，使用TfidfVectorizer在去掉停用词的条件下，使用默认配置的朴素贝叶斯分类器，可以得到88.264%的预测准确性

#综上所述，总结如下：
#  在统一训练模型下，且文本数据量较大时，去掉停用词的TfidfVectorizer的模型性能 优于 去掉停用词的CountVectorizer的模型性能
#  优于 未去掉停用词的TfidfVectorizer的模型性能 优于 未去掉停用词的CountVectorizer的模型性能