# , NeighborhoodComponentsAnalysis)
from sklearn.neighbors import (KNeighborsClassifier, NearestCentroid)
from sklearn.metrics import (confusion_matrix, classification_report, accuracy_score, auc,
                             roc_auc_score, f1_score, precision_recall_curve, precision_score,
                             recall_score, roc_curve, r2_score, log_loss)
from sklearn.model_selection import (train_test_split, cross_val_score, GridSearchCV, KFold,
                                     StratifiedKFold, ShuffleSplit, StratifiedShuffleSplit,
                                     RandomizedSearchCV, validation_curve, learning_curve)
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.preprocessing import StandardScaler, MaxAbsScaler, MinMaxScaler, RobustScaler
from sklearn.linear_model import LogisticRegression, SGDClassifier, RidgeClassifier, PassiveAggressiveClassifier
from sklearn.svm import SVC, NuSVC, LinearSVC
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier,
                              BaggingClassifier, GradientBoostingClassifier)
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline, make_union, Pipeline, FeatureUnion
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from wordcloud import WordCloud
from time import time
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import seaborn as sns
import re
import os
import operator
import collections
import warnings
warnings.simplefilter('ignore')

# npl related

# Evaluation
# scaling

# pipelines

# Dimenssion Reduction

# Cluster model

# Classifiers


def get_data(train_data_file, test_data_file, path, chunksize=100):
    """ Parse train and test data from csv file 

    Arguments:
        train_data_file {str} -- train dataset file 
        test_data_file {str} -- test dataset file
        path {str} -- Path where test and train files are located

    Keyword Arguments:
        chunksize {int} -- specification of chunk size for data parsing (default: {100})

    Returns:
        df_train [DataFrame] -- returns train data in pandas Dataframe object
        df_test [DataFrame] -- returns test data in pandas Dataframe object
    """
    dflist_train = []
    dflist_test = []
    print("preparing train dataframe.....")
    chunks = 0
    for df in pd.read_csv(path + train_data_file, chunksize=chunksize):
        chunks += chunksize
        # if len(dflist_test)%300 == 0:
        #     print 'adding chunk : {}'.format(str(chunks))
        dflist_train.append(df)
    print('train df list prepared....')
    df_train = pd.concat(dflist_train, axis=0)
    # print '++++ train df ++++'
    # print df_train.head()
    print('class count in train: ', df_train['lebel'].value_counts())
    print('number of documents in train dataset: ', df_train.shape[0])

    if test_data_file is not None:
        print("\npreparing test dataframe.....")
        chunks = 0
        for df in pd.read_csv(path + test_data_file, chunksize=chunksize):
            chunks += chunksize
            # if len(dflist_train)%300 == 0:
            #     print 'adding chunk : {}'.format(str(chunks))
            dflist_test.append(df)
        print('train df list prepared....')
        df_test = pd.concat(dflist_test, axis=0)
        # print '++++ test df ++++'
        # print df_test.head()
        print('class count in test: ', df_test['lebel'].value_counts())
        print('number of documents in train dataset: ', df_test.shape[0])
        return df_train, df_test
    else:
        return df_train


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


# vectorizer = 'tfidf' or 'tf'
def word_to_vectorize(train_clean_text, train_target, test_clean_text, vectorizer='tfifd',
                      filename='None', path4plot=None):

    if vectorizer == 'tfidf':
        vect = TfidfVectorizer(max_df=0.95, min_df=10,
                               stop_words='english')
    else:  # 'tf'
        vect = CountVectorizer(max_df=0.95, min_df=10,
                               stop_words='english')
    X = vect.fit_transform(train_clean_text)
    X_test = vect.fit_transform(test_clean_text)

    params = vect.get_feature_names()
    freq_param = X.toarray().sum(axis=0)
    feat_importances = dict(zip(params, freq_param))
    sorted_features_neg_cv = sorted(feat_importances.items(),
                                    key=operator.itemgetter(1), reverse=True)
    df_imp = pd.DataFrame(sorted_features_neg_cv)
    df_imp = df_imp.rename(columns={0: "word", 1: "Freq"}
                           ).sort_values(by="Freq", ascending=False)
    return X.toarray(), train_target, X_test.toarray(), df_imp


def plot_labeldist(imp, ylabel=None, title=None, filename='None', figsize=(12, 6), path4plot=None):
    labeldist = True
    if labeldist:
        plt.figure(figsize=figsize)
        sns_plot = sns.distplot(imp['Freq'].values, bins=50)
        sns_plot.set(ylabel=ylabel)
        sns_plot.set_title(title)
        sns_plot.get_figure().savefig(path4plot + filename)
    return None


def plot_labelcount(df, kind='count', vartocount=None, ylabel=None, title=None, figsize=(12, 6), path4plot=None):
    labelcount = True
    if labelcount:
        plt.figure(figsize=figsize)
        sns_plot = sns.catplot(x=vartocount, data=df,
                               kind=kind, palette='RdBu')
        # sns_plot.set_title(title)
        sns_plot.savefig(path4plot + 'class_count.png')
    return None


def plot_feat_importance(df_imp, xlabel=None, ylabel=None, title=None, figsize=(12, 6), path4plot=None):
    feat_imp = True
    if feat_imp:
        plt.figure(figsize=figsize)
        sns_plot = sns.barplot(np.arange(df_imp.shape[0]), df_imp.iloc[:, 1])
        sns_plot.set(ylabel=ylabel, xlabel=xlabel)
        sns_plot.set_title(title)
        sns_plot.set_xticklabels(df_imp.word.values.tolist(), rotation=90)
        sns_plot.get_figure().savefig(path4plot + 'feature_importance.png')
    return None


def tsne_visualization(X, figsize=(12, 6), path4plot=''):
    t = TSNE(learning_rate=50)
    tsne = t.fit_transform(X)

    knn = KMeans(n_clusters=2)
    knn.fit(X)
    pred = knn.predict(X)

    plt.figure(figsize=figsize)
    sns_plot = sns.scatterplot(x=tsne[:, 0], y=tsne[:, 1], hue=pred)
    sns_plot.set(xlabel="Feature 1", ylabel="Feature 2")
    sns_plot.set_title("Feature 1 Vs Feature 2")
    sns_plot.get_figure().savefig(path4plot + "tsne_plot.png")


def principal_component_analysis(X, numComponents=50, filename=None, path4plot=''):
    pca = PCA(n_components=numComponents)
    X_transform = pca.fit_transform(X)
    plt.figure(figsize=(12, 6))
    sns_plot = sns.barplot(range(pca.n_components_),
                           pca.explained_variance_ratio_.cumsum())
    sns_plot.set(xlabel="Principal Components",
                 ylabel="Cummulative Explained Variance")
    sns_plot.set_title(
        "Principal Components Vs Cummulative Explained Variance")
    sns_plot.get_figure().savefig(path4plot + filename)
    return X_transform


def dimension_red(X_train, y_train, X_test, y_test,  n_components=2, n_neighbors=3, path4plot=None):

    pca = make_pipeline(StandardScaler(), PCA(n_components=n_components))
    lda = make_pipeline(StandardScaler(), LinearDiscriminantAnalysis(
        n_components=n_components))
    # nca = make_pipeline(StandardScaler(), NeighborhoodComponentsAnalysis(n_components=n_components,
    #                                                                    random_state=random_state))
    dim_red = [('PCA', pca), ('LDA', lda)]  # , ('NCA', nca)]
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    names = []
    X_embedded_list = []
    for i, (name, model) in enumerate(dim_red):
        model.fit_transform(X_train)
        # sns.barplot(range(model.n_components), model.explained_variance_.cumsum())
        names.append(name)
        # knn.fit(model.transform(X_train), y_train)
        # acc_knn = knn.score(model.transform(X_test), y_test)
        # print 'Nearest neighbor acuracy: ', acc_knn
        X_embedded = model.transform(X_test)
        print(X_embedded.shape)
        X_embedded_list.append(X_embedded)
    return X_embedded_list, i+1


def clf_model(X_train, y_train, X_validation, y_validation, model=None, path4plot=None):
    # import ipdb; ipdb.set_trace()
    # def clf_model(model=None, path4plot=None):
    clfs = ['model_passive_aggressive', 'model_ridge', 'model_nc', 'model_knn', 'model_gb',
            'model_mnb', 'model_dt', 'model_rf', 'model_edt', 'model_svc', 'model_lr',
            'model_xgb', 'model_abc', 'model_nusvc', 'model_linsvc', 'model_sgd',
            'model_gpc', 'model_bgc', 'model_gbc', 'model_mlp', 'model_lgbm']

    clfs_dict = {
        'model_passiv_aggressive': PassiveAggressiveClassifier(),
        'model_ridge': RidgeClassifier(),
        'model_nc': NearestCentroid(),
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
        'model_bgc': BaggingClassifier(KNeighborsClassifier()),
        'model_gbc': GradientBoostingClassifier(),
        'model_mlp': MLPClassifier(hidden_layer_sizes=(100, 100, 100), alpha=1e-5, solver='lbfgs', max_iter=500)
        # 'model_lgbm': LGBMClassifier()
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
        return None, None
    else:
        results = []
        cm = []
        modelerror = []
        cvec = CountVectorizer()
        for clf in clfs_dict:
            t0 = time()
            name = clf
            print(clf)
            clf = clfs_dict[clf]
            try:
                pipeline = Pipeline([
                    ('vectorizer', cvec),
                    ('classifier', clf)
                ])
                pipeline.fit(X_train, y_train)
                y_pred = pipeline.predict(X_validation)
                train_acc = pipeline.score(X_train, y_train)
                test_acc = pipeline.score(X_validation, y_validation)
                weighted_f1 = f1_score(
                    y_validation, y_pred, average='weighted')
                macro_f1 = f1_score(y_validation, y_pred, average='macro')
                micro_f1 = f1_score(y_validation, y_pred, average='micro')
                elapsetime = time() - t0

                cm.append([name, confusion_matrix(y_validation, y_pred)])
                results.append([name, train_acc, test_acc,
                                weighted_f1, macro_f1, micro_f1, elapsetime])
            except:
                print('error with ' + name)
                modelerror.append(name)

        df_r = pd.DataFrame(results,  columns=[
                            'classifier', 'train_acc', 'test_acc', 'weighted_f1', 'macro_f1', 'micro_f1', 'elapsed_time'])
        df_r = df_r.sort_values(by='test_acc', ascending=False)
        df_cm = pd.DataFrame(cm, columns=['classifier', 'confusion_matrix'])
        return df_r, df_cm, modelerror
