#!/usr/bin/env python
# coding: utf-8

# In[26]:


import pandas as pd
import pickle
import requests
import os
import sys

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score


# In[2]:


def send_slack(msg, channel="#practice-using-slack", username="jaja_bot"):
    webhook_URL = "https://hooks.slack.com/services/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"
    payload = {
        "channel": channel,
        "username": username,
        "icon_emoji": ":provision:",
        "text": msg,
    }
    response = requests.post(
        webhook_URL,
        data = json.dumps(payload),
    )
    return response


# In[23]:


def find_accuracy(alpha):
    
    # 기사 데이터 프레임을 로드
    article_df = pd.read_csv("{}/article.csv".format(os.path.dirname(os.path.realpath("__file__"))))
    
    # 테스트 데이터와 트레인 데이터 분리
    X_train, X_test, y_train, y_test = train_test_split(article_df.content, article_df.category, test_size=0.1, random_state=1)
    
    # vectorizer와 classification algorithm 설정
    clf = Pipeline([
        ('vect', TfidfVectorizer()),
        ('clf', MultinomialNB(alpha=float(alpha)))
    ])
    
    # 모델 생성
    model = clf.fit(X_train, y_train)
    
    # 테스트 데이터 예측 결과 출력
    y_pred = model.predict(X_test)
    
    # 정확도 확인
    result = accuracy_score(y_test, y_pred)
    send_slack("alpha:{}, accuracy:{}".format(alpha, result))
    return result                                    


# In[ ]:


result = find_accuracy(sys.argv[1])
print(result)

