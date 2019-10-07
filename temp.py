# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 21:04:29 2019

@author: user
"""

from apiclient.discovery import build
from oauth2client.service_account import ServiceAccountCredentials
import pandas as pd

SCOPES = ['https://www.googleapis.com/auth/analytics.readonly']
KEY_FILE_LOCATION = 'Insert your key.json'
VIEW_ID = 'input your view ID'

reportRequests = [{'viewId': VIEW_ID,
                   'dateRanges': [{'startDate': '2019-09-01', 'endDate': '2019-09-30'}],
                   'metrics': [{'expression': 'ga:uniquePageviews'}],
                   'dimensions': [{'name': 'ga:dateHourMinute'}, {'name': 'ga:pageTitle'}, {'name': 'ga:pagePath'}, {'name': 'ga:dimension1'}],
                   'pageSize': 100000}]

def initialize_analyticsreporting():
    credentials = ServiceAccountCredentials.from_json_keyfile_name(KEY_FILE_LOCATION, SCOPES)
    analytics = build('analyticsreporting', 'v4', credentials = credentials)
    return analytics

def get_report(analytics, reportRequests):
    return analytics.reports().batchGet(body = {'reportRequests': [reportRequests]}).execute()

def print_response(response):
    list = []
    for report in response.get('reports', []):
        columnHeader = report.get('columnHeader', {})
        dimensionHeaders = columnHeader.get('dimensions', [])
        metricHeaders = columnHeader.get('metricHeader', {}).get('metricHeaderEntries', [])
        rows = report.get('data', {}).get('rows', [])
        for row in rows:
            dict = {}
            dimensions = row.get('dimensions', [])
            dateRangeValues = row.get('metrics', [])
            for header, dimension in zip(dimensionHeaders, dimensions):
                dict[header] = dimension
            for i, values in enumerate(dateRangeValues):
                for metric, value in zip(metricHeaders, values.get('values')):
                    if ',' in value or '.' in value:
                        dict[metric.get('name')] = float(value)
                    else:
                        dict[metric.get('name')] = int(value)
            list.append(dict)
        df = pd.DataFrame(list)
        return df

join = print_response(get_report(initialize_analyticsreporting(), reportRequests[0]))

uid = join['ga:dimension1'].unique().tolist()
path = join['ga:pagePath'].unique().tolist()
title = join['ga:pageTitle'].unique().tolist()
day_name = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
hour = list(range(24))
use_info = ['個股資訊', '我的庫存', '群益快訊', '我的自選股', '帳務查詢']

join['ga:dateHourMinute'] = pd.to_datetime(join['ga:dateHourMinute'])
join['day_name'] = pd.to_datetime(join['ga:dateHourMinute']).dt.day_name()
join['hour'] = pd.to_datetime(join['ga:dateHourMinute']).dt.hour
user = [join[['ga:dateHourMinute', 'day_name', 'hour', 'ga:pagePath', 'ga:pageTitle', 'ga:uniquePageviews']][join['ga:dimension1'] == uid[i]] for i in range(len(uid))]

view_count = [user[i]['ga:uniquePageviews'].sum() for i in range(len(user))]
view_day_name = [[sum(user[i]['day_name'] == day_name[j]) for j in range(len(day_name))] for i in range(len(user))]
view_hour = [[sum(user[i]['hour'] == hour[j]) for j in range(len(hour))] for i in range(len(user))]
use_count = [[sum(user[i]['ga:pageTitle'] == use_info[j]) for j in range(len(use_info))] for i in range(len(user))]

raw = pd.concat([pd.DataFrame(view_count), pd.DataFrame(use_count), pd.DataFrame(view_day_name), pd.DataFrame(view_hour)], axis = 1)
raw.columns = ['每月瀏覽數'] + use_info + day_name + hour

del path, title, day_name, hour, use_info, view_count, view_day_name, view_hour, use_count, join, user

from sklearn.decomposition import PCA

pca_new = PCA(n_components = 3)
pca_new.fit(raw)
dim_red = pca_new.fit_transform(raw)

def gmm(n_clusters, input_data):
    from sklearn.mixture import GaussianMixture
    from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score
    model = GaussianMixture(n_components = n_clusters).fit(input_data)
    labels = model.predict_proba(input_data).argmax(axis = 1)
    score1 = calinski_harabasz_score(raw, labels)
    score2 = davies_bouldin_score(raw, labels)
    return [n_clusters, score1, score2, labels, input_data]

def classification(x, y):
    from skopt.space import Real, Integer, Categorical
    from skopt.utils import use_named_args
    from skopt import gp_minimize
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.model_selection import cross_val_score   
    model = GradientBoostingClassifier()    
    space = [Real(10**-2, 0.5, name = 'learning_rate'),
         Integer(10, 500, name = 'n_estimators'),
         Categorical(['sqrt', 'log2'], name = 'max_features'),
         Real(0.5, 1.0, name = 'subsample'),
         Real(0.1, 0.4, name = 'validation_fraction')]    
    @use_named_args(space)
    def objective(**params):
        import numpy as np
        model.set_params(**params)
        return -np.mean(cross_val_score(model, x, y, cv = 10, n_jobs = -1, scoring = 'f1_macro'))
    u0 = gp_minimize(objective, space, n_calls = 10, random_state = 0, n_jobs = -1)
    fun = u0.fun
    values = u0.x
    return [-fun, values]

results = [gmm(i, dim_red) for i in range(2, 8)]
score = pd.DataFrame([results[i][1:3] for i in range(len(results))])
model1, model2 = score.iloc[:, 0].argmin(), score.iloc[:, 1].argmin()
test1, test2 = classification(x = results[model1][-1], y = results[model1][-2]),  classification(x = results[model2][-1], y = results[model2][-2])