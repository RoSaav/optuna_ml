#Paqueterias
#u
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import random
import warnings
warnings.filterwarnings('ignore')
#ml
import optuna
from sklearn.pipeline import Pipeline
from feature_engine.imputation import CategoricalImputer
from feature_engine.encoding import RareLabelEncoder, OrdinalEncoder 
from feature_engine.wrappers import SklearnTransformerWrapper
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
#val
from sklearn.metrics import roc_auc_score, classification_report, plot_confusion_matrix
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

from utils import optuna_optimizer_cat, optuna_optimizer_lgb, optuna_optimizer_xgb, plot_roc, format_plot

colors = ['#1A3252', '#EB5434', '#377A7B']


def train_optuna(
        random_state:int=123,
        test_size:float = 0.2,
        dataset_path:str='data/breast_cancer.csv',
        metric_name:str='target',
        classifier: str='lgb',
        n_trials: int=4,
        timeout: str=3600,
        n_splits: int=4,
):

    dataframe = pd.read_csv(dataset_path)

    #noise
    lst_cat = ['A', 'B', 'C', 'D', 'E', 'F' 'G', 'H', 'I', np.nan]
    lst_bool = [True, False]


    random.seed(random_state)
    dataframe['RANDOM_CAT'] = random.choices(lst_cat, k = dataframe.shape[0])
    dataframe['RANDOM_BOOL'] = random.choices(lst_bool, k = dataframe.shape[0])

    # Spliting
    print('Splitting...')
    train, test = train_test_split(dataframe, test_size=test_size, random_state=random_state)

    y_train = train.pop(metric_name)
    X_train = train

    y_test = test.pop(metric_name)
    X_test = test

    print(f'''
    Total size: {len(X_train) + len(X_test)}
    Train size: {len(X_train)}
    Test size: {len(X_test)}

    Columns: {X_train.shape[1]}
    ''')

    datadict = dict(zip(train.columns.tolist(),30*["numeric"]+['category', 'boolean']))

    ignore_features = []
    selected_features = {feature: type_ for feature, type_ in datadict.items() if feature not in ignore_features}

    numerical_features = [feature for feature, type_ in selected_features.items() if type_ == 'numeric']
    boolean_features = [feature for feature, type_ in selected_features.items() if type_ == 'boolean']
    categorical_features = [feature for feature, type_ in selected_features.items() if type_ == 'category']

    feature_names = [feature for feature in selected_features.keys()]

    # Pipeline
    steps = [
        ('categorical_imputer', CategoricalImputer(variables=categorical_features, ignore_format=True)),
        ('rare_label_encoder', RareLabelEncoder(variables=categorical_features, tol=0.01, ignore_format=True)),
        ('ordinal_encoder', OrdinalEncoder(variables=categorical_features, ignore_format=True, encoding_method='ordered')),
        ('standard_scaler', SklearnTransformerWrapper(StandardScaler(), variables=(numerical_features))),
        ]

    preprocess_pipeline = Pipeline(steps, verbose=True)
    #fit pipeline
    preprocess_pipeline.fit(X_train, y_train)
    #transform data
    X_train_preprocessed = preprocess_pipeline.transform(X_train)
    X_test_preprocessed = preprocess_pipeline.transform(X_test)

    study = optuna.create_study(pruner=optuna.pruners.HyperbandPruner(), direction='maximize', sampler=optuna.samplers.TPESampler(seed=random_state))
    if classifier == 'xgb':
        print('Trainig XGBClassifier with Optuna...')
        obj_function = lambda trial: optuna_optimizer_xgb(trial, X=X_train_preprocessed, y=y_train, n_splits=n_splits, random_state=random_state)
        study.optimize(obj_function, timeout=timeout, n_trials=n_trials)
        # Retrieving best params dict
        params = study.best_params
        params['n_estimators'] = params.pop('num_parallel_tree')
         # Instantiating with best params
        model = XGBClassifier(objective="binary:logistic", eval_metric="logloss", use_label_encoder=False, **params)
    elif classifier == 'cat':
        print('Trainig CatBoostClassifier with Optuna...')
        # Setting labels to int so cat can process it
        y_train = y_train.astype(int)
        y_test = y_test.astype(int)

        obj_function = lambda trial: optuna_optimizer_cat(trial, X=X_train_preprocessed, y=y_train, n_splits=n_splits, random_state=random_state)
        study.optimize(obj_function, timeout=timeout, n_trials=n_trials)

        # Retrieving best params dict
        params = study.best_params
        # Instantiating with best params
        model = CatBoostClassifier(**params)
    elif classifier == 'lgb':
        print('Trainig LGBMClassifier with Optuna...')
        obj_function = lambda trial: optuna_optimizer_lgb(trial, X=X_train_preprocessed, y=y_train, n_splits=n_splits, random_state=random_state)
        study.optimize(obj_function, timeout=timeout, n_trials=n_trials)

        # Retrieving best params dict
        params = study.best_params
        # Instantiating with best params
        model = LGBMClassifier(**params)

    model = model.fit(X_train_preprocessed, y_train, verbose= False)

    # path_img1 = f'img/plot_contour_{classifier}.png'
    # fig1 = optuna.visualization.plot_contour(study)
    # fig1.write_image(path_img1,  format='png', engine = 'kaleido')

    path_img2 = f'img/plot_edf_{classifier}.png'
    fig2 = optuna.visualization.plot_edf(study)
    fig2.write_image(path_img2,  format='png')

    path_img3 = f'img/plot_optimization_history_{classifier}.png'
    fig3 = optuna.visualization.plot_optimization_history(study)
    fig3.write_image(path_img3,  format='png')

    path_img4 = f'img/plot_slice_{classifier}.png'
    fig4 =optuna.visualization.plot_slice(study)
    fig4.write_image(path_img4,  format='png')

    path_img5 = f'img/plot_param_importances_{classifier}.png'
    fig5 =optuna.visualization.plot_param_importances(study)
    fig5.write_image(path_img5,  format='png')

    # Printing best model results and save best model and params
    print('Best score:', model.score(X_train_preprocessed, y_train))
    print('Best hyperparameters:', params, end='\n')

    # Defining final preprocess pipeline
    steps = [
        ('preprocessor', preprocess_pipeline),
        ('model', model)
    ]

    y_pred = model.predict(X_test_preprocessed)
    y_scores_train = model.predict_proba(X_train_preprocessed)[:, 1]
    y_scores_test = model.predict_proba(X_test_preprocessed)[:, 1]
    auc_train = roc_auc_score(y_train, y_scores_train)
    auc_test = roc_auc_score(y_test, y_scores_test)
    print(f'Roc Auc\n{20*"-"}\nTrain\t={auc_train:.2%}\nTest\t={auc_test:.2%}\n{20*"-"}')

    print(classification_report(y_test, y_pred))

    roc_path = f'img/roc_curve_{classifier}.png'
    # Printing ROC curves
    plt.figure(figsize=(6, 6))
    plot_roc(y_test, y_scores_test, 'Test', colors[1])
    plot_roc(y_train, y_scores_train, 'Train', colors[2])
    format_plot('ROC ' + roc_path[14:-4].upper()+f' {n_trials}TRIALS', 'False Positive Rate', 'True Positive Rate (Recall)')
    plt.savefig(roc_path, dpi=500, bbox_inches='tight'); plt.close()

    model_pipeline_path = f'train/models/model_pipeline_{classifier}_{n_trials}trials.joblib.dat'
    # Declaring and saving preprocess pipeline
    model_pipeline = Pipeline(steps, verbose=True)
    print('Exporting model pipeline .dat ...\n')
    joblib.dump(model_pipeline, model_pipeline_path)

    return model_pipeline

    
if __name__ == "__main__":
    train_optuna(
        random_state = 1337,
        test_size = 0.2,
        dataset_path = 'data/breast_cancer.csv',
        metric_name = 'target',
        classifier = 'xgb', # LightGBM -> lgb, XGBoost -> xgb, CatBoost -> cat
        n_trials = 50 ) 