#!/usr/bin/env python
# encoding: utf-8

#http://blog.sina.com.cn/s/blog_7103b28a0102w70h.html

from sklearn.cross_validation import StratifiedKFold
import pandas as pd

haichong='haichong-test.xlsx'
outputfile = 'result.xls'

data = pd.read_excel(haichong,header=None)

skf = StratifiedKFold(data[13], 4)
train_set = []
test_set = []
for train, test in skf:
    train_set.append(train)
    test_set.append(test)

print  train_set
print  test_set

train_set = []
test_set = []

from sklearn.cross_validation import LeaveOneOut
loo = LeaveOneOut(80)
for train,test in loo:
    print "%s %s" % (train,test)