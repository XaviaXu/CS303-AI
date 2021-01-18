import argparse
import json
import os
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.svm import NuSVC
from sklearn.model_selection import GridSearchCV

import numpy as np
from sklearn.datasets import fetch_20newsgroups

trainStr = []
trainValue = []

def readData(file):
    f = open(file,'r',encoding='utf-8')
    data = json.load(f)
    for i in data:
        trainStr.append(i['data'])
        trainValue.append(i['label'])
    f.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--training_data', type=str, default='train.json')
    args = parser.parse_args()
    trainingData = args.training_data

    readData(trainingData)


    #
    # twenty_train = fetch_20newsgroups(subset='train', shuffle=True)
    # twenty_train.target_names  # prints all the categories
    # print("\n".join(twenty_train.data[0].split("\n")[:3]))

    # text_clf = Pipeline([('vect',CountVectorizer),
    #                      ('tfidf',TfidfTransformer()),
    #                      ('clf-svm',SGDClassifier(loss='hinge'))
    #                      ])
    text_clf= Pipeline([('vect', CountVectorizer()),
                        ('tfidf', TfidfTransformer()),
                        ('clf-svm', NuSVC()),
    ])
    # param1 = {
    #     'vect__ngram_range':[(1,1),(1,2)],
    #     'tfidf__use_idf': (True, False),
    #     'clf-svm__nu':[0.5,0.6,0.4],
    #     'clf-svm__kernel':['poly'],
    #     'clf-svm__degree':[3,3.5,2.5],
    #     'clf-svm__coef0':[0,0.2,-0.2],
    #     #'clf-svm__class_weight':[None,'balanced'],
    #     'clf-svm__decision_function_shape':['ovo','ovr'],
    # }
    param1 = [{
            'vect__ngram_range': [(1, 1), (1, 2)],
            # 'tfidf__use_idf': (True, False),
            #'clf-svm__nu': [0.5, 0.6, 0.4],
            'clf-svm__kernel': ['poly'],
            #'clf-svm__degree': [3, 3.5, 2.5],
            # 'clf-svm__coef0':[0,0.2,-0.2],
            # 'clf-svm__class_weight':[None,'balanced'],
            'clf-svm__decision_function_shape': ['ovo', 'ovr'],
        },{
            #'vect__ngram_range': [(1, 1), (1, 2)],
            # 'tfidf__use_idf': (True, False),
            #'clf-svm__nu': [0.5, 0.6, 0.4],
            'clf-svm__kernel': ['poly'],
            #'clf-svm__degree': [3, 3.5, 2.5],
             'clf-svm__coef0':[0,0.2,-0.2],
             'clf-svm__class_weight':[None,'balanced'],
            #'clf-svm__decision_function_shape': ['ovo', 'ovr'],
        },{
        #'vect__ngram_range':[(1,1),(1,2)],
        #'tfidf__use_idf': (True, False),
        'clf-svm__nu':[0.5,0.55,0.45],
        'clf-svm__kernel':['poly'],
        'clf-svm__degree':[3,3.5,2.5],
        #'clf-svm__coef0':[0,0.2,-0.2],
        #'clf-svm__class_weight':[None,'balanced'],
        #'clf-svm__decision_function_shape':['ovo','ovr'],
        },



    ]
    param2 = {
        #'vect__ngram_range':[(1,1),(1,2)],
        #'tfidf__use_idf': (True, False),
        'clf-svm__nu':[0.5,0.55,0.45],
        'clf-svm__kernel':['rbf'],
        #'clf-svm__class_weight':[None,'balanced'],
        #'clf-svm__decision_function_shape':['ovo','ovr'],
    }
    param3 = {
        #'vect__ngram_range':[(1,1),(1,2)],
        #'tfidf__use_idf': (True, False),
        'clf-svm__nu':[0.5,0.55,0.45],
        'clf-svm__kernel':['sigmoid'],
        'clf-svm__coef0':[0,0.05,-0.05],
        #'clf-svm__class_weight':[None,'balanced'],
        #'clf-svm__decision_function_shape':['ovo','ovr'],
    }
    # params = [{
    #     'vect__ngram_range':[(1,1),(1,2)],
    #     'tfidf__use_idf': (True, False),
    #     'clf-svm__nu':[0.5,0.6,0.4,0.7,0.3],
    #     'clf-svm__kernel':['poly'],
    #     'clf-svm__degree':[3,4,2,3.5,2.5],
    #     'clf-svm__coef0':[0,0.2,0.4,-0.2,-0.4],
    #     'clf-svm__class_weight':[None,'balanced'],
    #     'clf-svm__decision_function_shape':['ovo','ovr'],
    # },{
    #     'vect__ngram_range':[(1,1),(1,2)],
    #     'tfidf__use_idf': (True, False),
    #     'clf-svm__nu':[0.5,0.6,0.4,0.7,0.3],
    #     'clf-svm__kernel':['rbf'],
    #     'clf-svm__class_weight':[None,'balanced'],
    #     'clf-svm__decision_function_shape':['ovo','ovr'],
    # },{
    #     'vect__ngram_range':[(1,1),(1,2)],
    #     'tfidf__use_idf': (True, False),
    #     'clf-svm__nu':[0.5,0.6,0.4,0.7,0.3],
    #     'clf-svm__kernel':['sigmoid'],
    #     'clf-svm__coef0':[0,0.2,0.4,-0.2,-0.4],
    #     'clf-svm__class_weight':[None,'balanced'],
    #     'clf-svm__decision_function_shape':['ovo','ovr'],
    # }]

    #text_clf.fit(training,training_val)


    with open("C://users//58376//training.txt","a+",encoding='utf-8')as f:
        f.write("===============Test Params 2==================\n")
        print("===============Test Params 2==================")

        gs_clf_svm = GridSearchCV(text_clf, param2, n_jobs=-1)
        gs_clf_svm = gs_clf_svm.fit(trainStr, trainValue)
        f.write("params2 results:\n")
        print("params2 results:")
        f.write("best scores:\n")
        f.write(gs_clf_svm.best_score_)
        f.write("best params:\n")
        f.write(gs_clf_svm.best_params_)
        print(gs_clf_svm.best_score_)
        print(gs_clf_svm.best_params_)

        f.write("===============Test Params 3==================")
        print("===============Test Params 3==================")

        gs_clf_svm = GridSearchCV(text_clf, param3, n_jobs=-1)
        gs_clf_svm = gs_clf_svm.fit(trainStr, trainValue)
        f.write("params3 results:")
        print("params3 results:")
        f.write("best scores:")
        f.write(gs_clf_svm.best_score_)
        f.write("best params:")
        f.write(gs_clf_svm.best_params_)
        print(gs_clf_svm.best_score_)
        print(gs_clf_svm.best_params_)




        f.write("===============Test Params 1==================")
        print("===============Test Params 1==================")
        for i in range(3):
            f.write("params1-"+str(i))
            print("params1-"+str(i))
            gs_clf_svm = GridSearchCV(text_clf, param1[i], n_jobs=-1)
            gs_clf_svm = gs_clf_svm.fit(trainStr, trainValue)
            f.write("params1-" + str(i)+"results:")
            print("params1-" + str(i)+"results:")
            f.write("best scores:")
            f.write(gs_clf_svm.best_score_)
            f.write("best params:")
            f.write(gs_clf_svm.best_params_)
            print(gs_clf_svm.best_score_)
            print(gs_clf_svm.best_params_)



    # print("===============Test Params 2==================")
    # gs_clf_svm = GridSearchCV(text_clf, param2, n_jobs=15)
    # gs_clf_svm = gs_clf_svm.fit(trainStr, trainValue)
    #
    # print(gs_clf_svm.best_score_)
    # print(gs_clf_svm.best_params_)
    #
    # print("===============Test Params 3==================")
    # gs_clf_svm = GridSearchCV(text_clf, param3, n_jobs=15)
    # gs_clf_svm = gs_clf_svm.fit(trainStr, trainValue)
    #
    # print(gs_clf_svm.best_score_)
    # print(gs_clf_svm.best_params_)

    # vectorizer = CountVectorizer()
    # trainCounts = vectorizer.fit_transform(trainStr)
    # trainCounts.shape
    # print(vectorizer)
    #
    # tf_idf_transformer = TfidfTransformer()
    # # 将文本转为词频矩阵并计算tf-idf
    # train_tfidf = tf_idf_transformer.fit_transform(trainCounts)
    # print(train_tfidf.shape)
    # # 将tf-idf矩阵抽取出来，元素a[i][j]表示j词在i类文本中的tf-idf权重


    # 对测试集进行tf-idf权重计算


    print('输出x_train文本向量：')

