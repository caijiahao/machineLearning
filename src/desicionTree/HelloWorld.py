# -*- coding: gbk -*-  
import numpy as np  
import scipy as sp  
from sklearn import tree  
from sklearn.metrics import precision_recall_curve  
from sklearn.metrics import classification_report  
from sklearn.cross_validation import train_test_split  
  
  
''''' ���ݶ��� '''  
data   = []  
labels = []  
with open("1.txt") as ifile:  
        for line in ifile:  
            tokens = line.strip().split(' ')  
            data.append([float(tk) for tk in tokens[:-1]])  
            labels.append(tokens[-1])  
x = np.array(data)  
labels = np.array(labels)  
y = np.zeros(labels.shape)  
  
  
''''' ��ǩת��Ϊ0/1 '''  
y[labels=='fat']=1  
  
''''' ���ѵ��������������� '''  
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)  
  
''''' ʹ����Ϣ����Ϊ���ֱ�׼���Ծ���������ѵ�� '''  
clf = tree.DecisionTreeClassifier(criterion='entropy')  
print(clf)  
clf.fit(x_train, y_train)  
  
''''' �Ѿ������ṹд���ļ� '''  
with open("tree.dot", 'w') as f:  
    f = tree.export_graphviz(clf, out_file=f)  
      
''''' ϵ����ӳÿ��������Ӱ������Խ���ʾ�������ڷ������𵽵�����Խ�� '''  
print(clf.feature_importances_)  
  
'''''���Խ���Ĵ�ӡ'''  
answer = clf.predict(x_train)  
print(x_train)  
print(answer)  
print(y_train)  
print(np.mean( answer == y_train))  
  
'''''׼ȷ�����ٻ���'''  
precision, recall, thresholds = precision_recall_curve(y_train, clf.predict(x_train))  
answer = clf.predict_proba(x)[:,1]  
print(classification_report(y, answer, target_names = ['thin', 'fat']))  