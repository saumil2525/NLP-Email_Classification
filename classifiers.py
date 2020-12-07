from sklearn.tree import ExtraTreeClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.preprocessing import StandardScaler, MaxAbsScaler, MinMaxScaler, RobustScaler
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, NuSVC, LinearSVC
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier,
                              BaggingClassifier, GradientBoostingClassifier)
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.neighbors import (KNeighborsClassifier)
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline, make_union, Pipeline, FeatureUnion
from sklearn.metrics import (confusion_matrix, classification_report, accuracy_score, auc,
                             roc_auc_score, f1_score, precision_recall_curve, precision_score,
                             recall_score, roc_curve, r2_score, log_loss)
from sklearn.model_selection import (train_test_split, cross_val_score, GridSearchCV, KFold,
                                     StratifiedKFold, ShuffleSplit, StratifiedShuffleSplit,
                                     RandomizedSearchCV, validation_curve, learning_curve)
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from wordcloud import WordCloud
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import seaborn as sns
import re
import os
import sys
import operator

print(sys.executable)
print(sys.path)


def get_cleaned_text(review, words_len=3, remove_stopwords=False,
                     is_stemming=False, is_lemma=False, split=False):

    review_text = re.sub("[^a-zA-Z]", " ", review)
    words = review_text.split()
    words = [w.lower() for w in words if len(w) >= words_len]
    if remove_stopwords:
        stop_words = set(stopwords.words("english"))
        # stop_words = stop_words.union(set('movi', 'film'))
        words = [w for w in words if not w in stop_words]
    if is_stemming:
        ps = PorterStemmer()
        words = [ps.stem(j) for j in words]
    if is_lemma:
        word_lemm = WordNetLemmatizer()
        pos = ['v', 'n', 'a']
        for p in pos:
            words = [word_lemm.lemmatize(k, pos=p) for k in words]
    if split:
        return words
    else:
        return (' '.join(words))


def clf_model(X_train, y_train, X_validation, y_validation, model=None,
              path4plot=None):

    clfs_dict = {
        'model_knn': KNeighborsClassifier(),
        'model_gb': GaussianNB(),
        'model_mnb': MultinomialNB(),
        'model_dt': DecisionTreeClassifier(),
        'model_rf': RandomForestClassifier(),
        'model_edt': ExtraTreesClassifier(),
        'model_svc': SVC(),
        'model_lr': LogisticRegression(),
        'model_xgb': XGBClassifier(),
        'model_abc': AdaBoostClassifier(),
        'model_nusvc': NuSVC(),
        'model_linsvc': LinearSVC(),
        'model_sgd': SGDClassifier(),
        'model_gpc': GaussianProcessClassifier(),
        # 'model_bgc': BaggingClassifier(KNeighborsClassifier(), ),
        'model_gbc': GradientBoostingClassifier(),
        'model_mlp': MLPClassifier(hidden_layer_sizes=(100, 100, 100), alpha=1e-5, solver='lbfgs', max_iter=500, ),
        'model_lgbm': LGBMClassifier()
    }
    if model is not None:
        clf = clfs_dict[model]
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_validation)
        train_acc = clf.score(X_train, y_train)
        test_acc = clf.score(X_validation, y_validation)

        print('Training Accuracy: ', train_acc)
        print('Test Accuracy: ', test_acc)
        cm = confusion_matrix(y_validation, y_pred)
        return None
    else:
        results = []
        cm = []
        for idx, model in enumerate(clfs_dict):
            try:
                print('\t {}. --> {}'.format((idx+1), model))
                clf = clfs_dict[model]
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_validation)
                train_acc = clf.score(X_train, y_train)
                test_acc = clf.score(X_validation, y_validation)
                f1_score_wighted = f1_score(
                    y_validation, y_pred, average='weighted')
                f1_score_micro = f1_score(
                    y_validation, y_pred, average='micro')
                f1_score_macro = f1_score(
                    y_validation, y_pred, average='macro')
                results.append(
                    [model, train_acc, test_acc, f1_score_wighted, f1_score_micro, f1_score_macro])
                cm.append([model, confusion_matrix(y_validation, y_pred)])
            except Exception as e:
                print('\t\t** error in ''{}'': {}'.format(model, str(e)))
                pass

        df_r = pd.DataFrame(
            results, columns=['classifier', 'train_acc', 'test_acc', 'f1_wighted', 'f1_macro', 'f1_macro'])
        # df_r.append([model, train_acc, test_acc])
        return df_r, cm


print('\nFinished.................')
