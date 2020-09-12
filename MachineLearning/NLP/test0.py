import os
import jieba
import jieba.posseg as pseg
import numpy as np
import pandas as pd
from gensim.models.word2vec import Word2Vec
from scipy import spatial
from sklearn.metrics import confusion_matrix

root_path = ''
pd.set_option('display.width', 1000)
pd.set_option('display.max_columns', 1000)
pd.set_option('display.max_rows', 1000)

jieba.set_dictionary(os.path.join(root_path, 'Train/corpus/dict.txt.big'))
jieba.load_userdict(os.path.join(root_path, 'Train/corpus/medical_term.txt'))

med_model = Word2Vec.load(os.path.join(root_path, 'Train/model/med_word2vec.model'))
index2word_set = set(med_model.wv.index2word)

def loadfile(file_name):
    file_path = os.path.join(root_path, file_name)
    lines = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            lines.append(line.replace('\n', ''))
    return lines

def load_stop_word(file_name):
    file_path = os.path.join(root_path, file_name)
    stop_words = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            stop_words.append(line.strip())
    return stop_words

def get_corpus_words(corpus_path):
    word_list =[]
    with open(corpus_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            line = line.replace('\n', '')
            s = line.split('@@')
            word = s[0].strip()
            try:
                type = s[1].strip('')
            except:
                print(s)
            word_list.append(word)
    return word_list

def trans_spec_char(str_cn):
    str1 = u'！@#%&（）【】：；，。《》？1234567890'
    str2 = u'!@#%&()[]:;,.<>?1234567890'
    table = {ord(x) : ord(y) for x, y in zip(str1, str2)}
    str_en = str_cn.translate(table)
    return str_en

def del_digits(text):
    text = text.replace('.', '')
    for i in range(10):
        text = text.replace(str(i), '')
    return text

import Levenshtein
def word_similarity(corpus, word):
    max = 0.0
    matched_word = '未找到'
    for name in corpus:
        matching = Levenshtein.jaro_winkler(name, word)
        if matching > max:
            max = matching
            matched_word = name
    return matched_word, max

def del_stop_word(text, stop_words):
    for word in stop_words:
        text.replace(word, '')
    return text

def extract_entity(text, flag='n'):
    entities = []
    segments = pseg.cut(text.strip())
    for w, f in segments:
        if f == flag:
            entities.append(w)
    if len(entities) == 0:
        entities.append(None)
    return entities

def avg_feature_vector(sentence):
    words = sentence.split()
    feature_vec = np.zeros((200,), dtype='float32')
    n_words = 0
    for word in words:
        n_words += 1
        feature_vec = np.add(feature_vec, med_model[word])
    if n_words > 0:
        feature_vec = np.divide(feature_vec, n_words)
    return feature_vec

def rule_eval(filepath, df_data, column, expression):
    rules = pd.read_csv(filepath)
    data_array = np.array(df_data.loc[:, ['ICD4', column]])
    count = 0
    for data in data_array:
        key = list(data)[0]
        val = list(data)[1]
        rule = rules[rules['ICD4'] == list(data)[0]]
        if len(rule) > 0:
            rule_min = float(rule['min'].tolist()[0])
            rule_max = float(rule['max'].tolist()[0])
            rule_25 = float(rule['25%'].tolist()[0])
            rule_50 = float(rule['50%'].tolist()[0])
            rule_75 = float(rule['75%'].tolist()[0])
            rule_mean = float(rule['mean'].tolist()[0])
            rule_std = float(rule['std'].tolist()[0])
            rule_count = float(rule['count'].tolist()[0])
            if eval(expression):
                df_data.loc[count, 'reason'] += '[' + str(column) + ']'
                df_data.loc[count, 'predict'] = 1.0
            else:
                df_data.loc[count, 'predict'] = 0.0
        count += 1
    return df_data

stop_words = load_stop_word(os.path.join(root_path, 'Train/corpus/stopwords.txt'))

texts = ['徐汇神经', '复旦中山卫生院', "国中人民医院"]
corpus = get_corpus_words(os.path.join(root_path, 'Train/corpus/medical_term.txt'))
for text in texts:
    print('---------------------------')
    print('需识别文本：', text)
    word = trans_spec_char(text.replace(' ', ''))
    word = del_digits(word)
    word = del_stop_word(word, stop_words)
    print(word)
    result = word_similarity(corpus, word)
    if result[1] < 0.618:
        result = extract_entity(text, flag='@@drug')
        if len(result) == 0:
            result = extract_entity(text, flag='@@diag')
    print('识别实体', result[0])

med_item0 = '尼莫地平 长春西汀 氯化钾 奥扎格雷钠 还原型谷胱甘肽 缬沙坦 地西泮 布洛芬 头孢西丁 艾司唑仑 吲哚美辛'
med_item1 = '全天麻胶囊(片) 参麦注射液 养血清脑丸(颗粒) 山莨菪碱 倍他司汀 泮托拉唑'
s1 = avg_feature_vector(med_item0)
s2 = avg_feature_vector(med_item1)
item_simility = 1.0 - spatial.distance.cosine(s1, s2)
print('诊疗记录1：', med_item0)
print('诊疗记录2：', med_item1)
print('诊疗记录相似度：%.2f%%' % (100 * item_simility))


sdata = pd.read_csv('Train/data/训练测试案例(ICD4).csv', engine="python", encoding="utf-8")
sdata['reason'] = ''
print(sdata.head())
filepath = 'Train/data/疾病住院费用统计(ICD4).csv'
sdata = rule_eval(filepath, sdata, "BILL_SUM", 'val > 2 * rule_75')
print(sdata.head())
filepath = 'Train/data/疾病住院年龄统计(ICD4).csv'
sdata = rule_eval(filepath, sdata, "AGE", '(val > 3 * rule_50) | (val < 0.2 * rule_50)')
print(sdata.head())
filepath = 'Train/data/疾病住院天数统计(ICD4).csv'
sdata = rule_eval(filepath, sdata, 'DAYS_OF_STAY', 'val>2.5*rule_75')
print(sdata.head())
predict = sdata['predict']
label = sdata['label']
cnf_matrix = confusion_matrix(label, predict)
#TN FP
#FN TP
percision = (cnf_matrix[0, 0] / (cnf_matrix[0, 0] + cnf_matrix[0, 1]))
recall = cnf_matrix[0, 0] / (cnf_matrix[0, 0] + cnf_matrix[1, 0])
print('准确率为: %.2f%%' % (100*percision))
print('召回率为: %.2f%%' % (100*recall))