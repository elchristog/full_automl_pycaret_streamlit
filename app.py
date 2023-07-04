from operator import index
import streamlit as st
import plotly.express as px
import pandas_profiling
import pandas as pd
import os 

from pycaret.classification import *
from sklearn.metrics import accuracy_score, roc_auc_score, recall_score, precision_score, f1_score, cohen_kappa_score, matthews_corrcoef
from streamlit_pandas_profiling import st_profile_report
from pyspark.sql import SparkSession
# from pycaret.parallel import FugueBackend

spark = SparkSession.builder.getOrCreate()

st.set_page_config(page_title="AutoClassification MLOps", page_icon="📈", layout="wide", initial_sidebar_state="expanded")

if os.path.exists('./dataset.csv'): 
    df = pd.read_csv('dataset.csv', index_col=None)
    val_df = pd.read_csv('validation.csv', index_col=None)

with st.sidebar: 
    st.image("https://www.onepointltd.com/wp-content/uploads/2019/12/ONE-POINT-01-1.png")
    st.title("Auto classification ML App to MLOps")
    choice = st.radio("Navigation", ["Upload","Profiling","Modelling", "Download"])
    st.info("This project was developed by Christian Gonzalez: cdgonzalezr@unal.edu.co c.gonzalezr@uniandes.edu.co")

if choice == "Upload":
    st.title("Upload Your Dataset")
    file = st.file_uploader("Upload Your Dataset")
    if file: 
        df = pd.read_csv(file, index_col=None)
        df.to_csv('dataset.csv', index=None)
        st.dataframe(df)
    file_val = st.file_uploader("Upload Your Validation Dataset")
    if file_val: 
        df_val = pd.read_csv(file_val, index_col=None)
        df_val.to_csv('validation.csv', index=None)
        st.dataframe(df_val)

if choice == "Profiling": 
    if st.button('Run Profiling'):         
        st.title("Exploratory Data Analysis")
        profile_df = df.profile_report()
        st_profile_report(profile_df)
        profile_df.to_file("output.html")

if choice == "Modelling": 
    st.title("Modelling")
    st.subheader("Antes de iniciar, abre una terminal y abre el diseño de experimentos de mflow usando el comando 'mlflow ui'")
    st.dataframe(df.head())
    train = df.sample(frac=0.8, random_state=786).reset_index(drop=True)
    test = df.drop(train.index).reset_index(drop=True)

    col1, col2, col3 = st.columns(3)
    with col1: 
        chosen_target = st.selectbox('Choose the Target Column', df.columns)
        chosen_numerical_features = st.multiselect('Choose the Numeric Feature Columns', df.columns)
        chosen_categorical_features = st.multiselect('Choose the Categorical Feature Columns', df.columns)
        chosen_exclude_features = st.multiselect('Choose the Exclude Feature Columns', df.columns)
        high_cardinality_features = st.multiselect('Choose the High Cardinality Feature Columns', df.columns)
    with col2: 
        models = st.multiselect('Choose the Models', ['Logistic Regression', 'Linear Discriminant Analysis', 'Naive Bayes', 'Random Forest Classifier', 'Quadratic Discriminant Analysis', 'Extra Trees Classifier', 'Gradient Boosting Classifier', 'Light Gradient Boosting Machine', 'Ada Boost Classifier', 'K Neighbors Classifier', 'Decision Tree Classifier', 'Dummy Classifier', 'SVM - Linear Kernel', 'Ridge Classifier'])
        mapped_model = {'Logistic Regression': 'lr', 'Linear Discriminant Analysis': 'lda', 'Naive Bayes': 'nb', 'Random Forest Classifier': 'rf', 'Quadratic Discriminant Analysis': 'qda', 'Extra Trees Classifier': 'et', 'Gradient Boosting Classifier': 'gbc', 'Light Gradient Boosting Machine': 'lightgbm', 'Ada Boost Classifier': 'ada', 'K Neighbors Classifier': 'knn', 'Decision Tree Classifier': 'dt', 'Dummy Classifier': 'dummy', 'SVM - Linear Kernel': 'svm', 'Ridge Classifier': 'ridge'}
        select_models = [mapped_model[i] for i in models]
    with col3: 
        metric = st.selectbox('Choose the Metric', ['Accuracy', 'AUC', 'Recall', 'Precision', 'F1', 'Kappa', 'MCC'])
    
    st.markdown("<hr/>", unsafe_allow_html=True)
    
    if st.button('Run Modelling'): 
        setup(train, target=chosen_target, silent=True, numeric_features=chosen_numerical_features, categorical_features=chosen_categorical_features, ignore_features=chosen_exclude_features,  fold=10, log_experiment=False, experiment_name='AutoChrisML', remove_outliers=True, normalize=True, transformation=True, normalize_method='robust', transformation_method='yeo-johnson', imputation_type='iterative', feature_selection=True, feature_selection_threshold=0.8, feature_interaction=True, feature_ratio=True, polynomial_features=True, polynomial_degree=2, combine_rare_levels=True, rare_level_threshold=0.1, remove_multicollinearity=True, multicollinearity_threshold=0.9, create_clusters=True, cluster_iter=20, ignore_low_variance=True, remove_perfect_collinearity=True, use_gpu=True, session_id=123, high_cardinality_features=high_cardinality_features, fix_imbalance= True, log_plots=True, log_profile=True, log_data=True)
        setup_df = pull()
        st.dataframe(setup_df)
        top_models = compare_models(sort= metric , n_select=3, include=select_models)#, parallel = FugueBackend(spark))
        compare_df = pull()
        st.dataframe(compare_df)

        st.text("🔹Tuning the top models")
        try:
            tune_top = [tune_model(i) for i in top_models]
            st.dataframe(tune_top)
        except:
            st.text(" Alguno de los modelos no se puede optimizar")
            tune_top = top_models

        st.text("🔹Bagging the top model")
        try:
            bagged_top = ensemble_model(tune_top[0], method='Bagging', choose_better=True)
        except:
            st.text("El modelo no se puede ensamblar")
            bagged_top = tune_top[0]
        st.text("🔹Boosting the top model")
        try:
            boost_top = ensemble_model(tune_top[0], method='Boosting', choose_better=True)
        except:
            st.text("El modelo no se puede ensamblar")
            boost_top = tune_top[0]

        st.text("🔹Blending the top models")
        try:
            blend_top = blend_models(estimator_list=top_models, fold=10, round=4, choose_better=True)  
        except:
            st.text("Los modelos no se pueden combinar")
            blend_top = tune_top[0]

        st.text("🔹Stacking the top models")
        try:
            stack_top = stack_models(estimator_list=top_models, fold = 10)
            best_model = automl(optimize=metric)
        except:
            st.text("Los modelos no se pueden apilar")
            stack_top = tune_top[0]
            best_model = tune_top[0]

        st.text("🔹Predicting on the test set")
        test['predictions_best_model'] = predict_model(best_model, data=test)['Label']
        test['predictions_blend_top'] = predict_model(blend_top, data=test)['Label']
        test['predictions_stack_top'] = predict_model(stack_top, data=test)['Label']
        test['predictions_bagged_top'] = predict_model(bagged_top, data=test)['Label']
        test['predictions_boost_top'] = predict_model(boost_top, data=test)['Label']
        st.dataframe(test)

        st.text("🔹Comparing the metrics")
        try:
            accuracy_best_model = accuracy_score(test[chosen_target], test['predictions_best_model'])
        except:
            accuracy_best_model = None
        try:
            accuracy_blend_top = accuracy_score(test[chosen_target], test['predictions_blend_top'])
        except:
            accuracy_blend_top = None
        try:
            accuracy_stack_top = accuracy_score(test[chosen_target], test['predictions_stack_top'])
        except:
            accuracy_stack_top = None
        try:
            accuracy_bagged_top = accuracy_score(test[chosen_target], test['predictions_bagged_top'])
        except:
            accuracy_bagged_top = None
        try:
            accuracy_boost_top = accuracy_score(test[chosen_target], test['predictions_boost_top'])
        except:
            accuracy_boost_top = None

        try:
            auc_best_model = roc_auc_score(test[chosen_target], test['predictions_best_model'])
        except:
            auc_best_model = None
        try:
            auc_blend_top = roc_auc_score(test[chosen_target], test['predictions_blend_top'])
        except:
            auc_blend_top = None
        try:
            auc_stack_top = roc_auc_score(test[chosen_target], test['predictions_stack_top'])
        except:
            auc_stack_top = None
        try:
            auc_bagged_top = roc_auc_score(test[chosen_target], test['predictions_bagged_top'])
        except:
            auc_bagged_top = None
        try:
            auc_boost_top = roc_auc_score(test[chosen_target], test['predictions_boost_top'])
        except:
            auc_boost_top = None

        try:
            recall_best_model = recall_score(test[chosen_target], test['predictions_best_model'])
        except:
            recall_best_model = None
        try:
            recall_blend_top = recall_score(test[chosen_target], test['predictions_blend_top'])
        except:
            recall_blend_top = None
        try:
            recall_stack_top = recall_score(test[chosen_target], test['predictions_stack_top'])
        except:
            recall_stack_top = None
        try:
            recall_bagged_top = recall_score(test[chosen_target], test['predictions_bagged_top'])
        except:
            recall_bagged_top = None
        try:
            recall_boost_top = recall_score(test[chosen_target], test['predictions_boost_top'])
        except:
            recall_boost_top = None

        try:
            precision_best_model = precision_score(test[chosen_target], test['predictions_best_model'])
        except:
            precision_best_model = None
        try:
            precision_blend_top = precision_score(test[chosen_target], test['predictions_blend_top'])
        except:
            precision_blend_top = None
        try:
            precision_stack_top = precision_score(test[chosen_target], test['predictions_stack_top'])
        except:
            precision_stack_top = None
        try:
            precision_bagged_top = precision_score(test[chosen_target], test['predictions_bagged_top'])
        except:
            precision_bagged_top = None
        try:
            precision_boost_top = precision_score(test[chosen_target], test['predictions_boost_top'])
        except:
            precision_boost_top = None

        try:
            f1_best_model = f1_score(test[chosen_target], test['predictions_best_model'])
        except:
            f1_best_model = None
        try:
            f1_blend_top = f1_score(test[chosen_target], test['predictions_blend_top'])
        except:
            f1_blend_top = None
        try:
            f1_stack_top = f1_score(test[chosen_target], test['predictions_stack_top'])
        except:
            f1_stack_top = None
        try:
            f1_bagged_top = f1_score(test[chosen_target], test['predictions_bagged_top'])
        except:
            f1_bagged_top = None
        try:
            f1_boost_top = f1_score(test[chosen_target], test['predictions_boost_top'])
        except:
            f1_boost_top = None

        try:
            kappa_best_model = cohen_kappa_score(test[chosen_target], test['predictions_best_model'])
        except:
            kappa_best_model = None
        try:
            kappa_blend_top = cohen_kappa_score(test[chosen_target], test['predictions_blend_top'])
        except:
            kappa_blend_top = None
        try:
            kappa_stack_top = cohen_kappa_score(test[chosen_target], test['predictions_stack_top'])
        except:
            kappa_stack_top = None
        try:
            kappa_bagged_top = cohen_kappa_score(test[chosen_target], test['predictions_bagged_top'])
        except:
            kappa_bagged_top = None
        try:
            kappa_boost_top = cohen_kappa_score(test[chosen_target], test['predictions_boost_top'])
        except:
            kappa_boost_top = None

        try:
            mcc_best_model = matthews_corrcoef(test[chosen_target], test['predictions_best_model'])
        except:
            mcc_best_model = None
        try:
            mcc_blend_top = matthews_corrcoef(test[chosen_target], test['predictions_blend_top'])
        except:
            mcc_blend_top = None
        try:
            mcc_stack_top = matthews_corrcoef(test[chosen_target], test['predictions_stack_top'])
        except:
            mcc_stack_top = None
        try:
            mcc_bagged_top = matthews_corrcoef(test[chosen_target], test['predictions_bagged_top'])
        except:
            mcc_bagged_top = None
        try:
            mcc_boost_top = matthews_corrcoef(test[chosen_target], test['predictions_boost_top'])
        except:
            mcc_boost_top = None
        
        st.dataframe(pd.DataFrame({
            'Accuracy': [accuracy_best_model, accuracy_blend_top, accuracy_stack_top, accuracy_bagged_top, accuracy_boost_top],
            'AUC': [auc_best_model, auc_blend_top, auc_stack_top, auc_bagged_top, auc_boost_top],
            'Recall': [recall_best_model, recall_blend_top, recall_stack_top, recall_bagged_top, recall_boost_top],
            'Precision': [precision_best_model, precision_blend_top, precision_stack_top, precision_bagged_top, precision_boost_top],
            'F1': [f1_best_model, f1_blend_top, f1_stack_top, f1_bagged_top, f1_boost_top],
            'Kappa': [kappa_best_model, kappa_blend_top, kappa_stack_top, kappa_bagged_top, kappa_boost_top],
            'MCC': [mcc_best_model, mcc_blend_top, mcc_stack_top, mcc_bagged_top, mcc_boost_top]
        }, index=['Best Model', 'Blend Top', 'Stack Top', 'Bagged Top', 'Boost Top']).sort_values(by=metric, ascending=True))

        st.text("🔹 Selecting the best model")
        if metric == 'Accuracy':
            if accuracy_best_model == max(accuracy_best_model, accuracy_blend_top, accuracy_stack_top, accuracy_bagged_top, accuracy_boost_top):
                selected_model = best_model
            elif accuracy_blend_top == max(accuracy_best_model, accuracy_blend_top, accuracy_stack_top, accuracy_bagged_top, accuracy_boost_top):
                selected_model = blend_top
            elif accuracy_stack_top == max(accuracy_best_model, accuracy_blend_top, accuracy_stack_top, accuracy_bagged_top, accuracy_boost_top):
                selected_model = stack_top
            elif accuracy_bagged_top == max(accuracy_best_model, accuracy_blend_top, accuracy_stack_top, accuracy_bagged_top, accuracy_boost_top):
                selected_model = bagged_top
            elif accuracy_boost_top == max(accuracy_best_model, accuracy_blend_top, accuracy_stack_top, accuracy_bagged_top, accuracy_boost_top):
                selected_model = boost_top
        elif metric == 'AUC':
            if auc_best_model == max(auc_best_model, auc_blend_top, auc_stack_top, auc_bagged_top, auc_boost_top):
                selected_model = best_model
            elif auc_blend_top == max(auc_best_model, auc_blend_top, auc_stack_top, auc_bagged_top, auc_boost_top):
                selected_model = blend_top
            elif auc_stack_top == max(auc_best_model, auc_blend_top, auc_stack_top, auc_bagged_top, auc_boost_top):
                selected_model = stack_top
            elif auc_bagged_top == max(auc_best_model, auc_blend_top, auc_stack_top, auc_bagged_top, auc_boost_top):
                selected_model = bagged_top
            elif auc_boost_top == max(auc_best_model, auc_blend_top, auc_stack_top, auc_bagged_top, auc_boost_top):
                selected_model = boost_top
        elif metric == 'Recall':
            if recall_best_model == max(recall_best_model, recall_blend_top, recall_stack_top, recall_bagged_top, recall_boost_top):
                selected_model = best_model
            elif recall_blend_top == max(recall_best_model, recall_blend_top, recall_stack_top, recall_bagged_top, recall_boost_top):
                selected_model = blend_top
            elif recall_stack_top == max(recall_best_model, recall_blend_top, recall_stack_top, recall_bagged_top, recall_boost_top):
                selected_model = stack_top
            elif recall_bagged_top == max(recall_best_model, recall_blend_top, recall_stack_top, recall_bagged_top, recall_boost_top):
                selected_model = bagged_top
            elif recall_boost_top == max(recall_best_model, recall_blend_top, recall_stack_top, recall_bagged_top, recall_boost_top):
                selected_model = boost_top
        elif metric == 'Precision':
            if precision_best_model == max(precision_best_model, precision_blend_top, precision_stack_top, precision_bagged_top, precision_boost_top):
                selected_model = best_model
            elif precision_blend_top == max(precision_best_model, precision_blend_top, precision_stack_top, precision_bagged_top, precision_boost_top):
                selected_model = blend_top
            elif precision_stack_top == max(precision_best_model, precision_blend_top, precision_stack_top, precision_bagged_top, precision_boost_top):
                selected_model = stack_top
            elif precision_bagged_top == max(precision_best_model, precision_blend_top, precision_stack_top, precision_bagged_top, precision_boost_top):
                selected_model = bagged_top
            elif precision_boost_top == max(precision_best_model, precision_blend_top, precision_stack_top, precision_bagged_top, precision_boost_top):
                selected_model = boost_top
        elif metric == 'F1':
            if f1_best_model == max(f1_best_model, f1_blend_top, f1_stack_top, f1_bagged_top, f1_boost_top):
                selected_model = best_model
            elif f1_blend_top == max(f1_best_model, f1_blend_top, f1_stack_top, f1_bagged_top, f1_boost_top):
                selected_model = blend_top
            elif f1_stack_top == max(f1_best_model, f1_blend_top, f1_stack_top, f1_bagged_top, f1_boost_top):
                selected_model = stack_top
            elif f1_bagged_top == max(f1_best_model, f1_blend_top, f1_stack_top, f1_bagged_top, f1_boost_top):
                selected_model = bagged_top
            elif f1_boost_top == max(f1_best_model, f1_blend_top, f1_stack_top, f1_bagged_top, f1_boost_top):
                selected_model = boost_top
        elif metric == 'Kappa':
            if kappa_best_model == max(kappa_best_model, kappa_blend_top, kappa_stack_top, kappa_bagged_top, kappa_boost_top):
                selected_model = best_model
            elif kappa_blend_top == max(kappa_best_model, kappa_blend_top, kappa_stack_top, kappa_bagged_top, kappa_boost_top):
                selected_model = blend_top
            elif kappa_stack_top == max(kappa_best_model, kappa_blend_top, kappa_stack_top, kappa_bagged_top, kappa_boost_top):
                selected_model = stack_top
            elif kappa_bagged_top == max(kappa_best_model, kappa_blend_top, kappa_stack_top, kappa_bagged_top, kappa_boost_top):
                selected_model = bagged_top
            elif kappa_boost_top == max(kappa_best_model, kappa_blend_top, kappa_stack_top, kappa_bagged_top, kappa_boost_top):
                selected_model = boost_top
        elif metric == 'MCC':
            if mcc_best_model == max(mcc_best_model, mcc_blend_top, mcc_stack_top, mcc_bagged_top, mcc_boost_top):
                selected_model = best_model
            elif mcc_blend_top == max(mcc_best_model, mcc_blend_top, mcc_stack_top, mcc_bagged_top, mcc_boost_top):
                selected_model = blend_top
            elif mcc_stack_top == max(mcc_best_model, mcc_blend_top, mcc_stack_top, mcc_bagged_top, mcc_boost_top):
                selected_model = stack_top
            elif mcc_bagged_top == max(mcc_best_model, mcc_blend_top, mcc_stack_top, mcc_bagged_top, mcc_boost_top):
                selected_model = bagged_top
            elif mcc_boost_top == max(mcc_best_model, mcc_blend_top, mcc_stack_top, mcc_bagged_top, mcc_boost_top):
                selected_model = boost_top

        tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10, tab11, tab12, tab13, tab14, tab15, tab16, tab17, tab18, tab19 = st.tabs(["Mainfold", "Interpret", "AUC", "Confusion Matrix", "Threshold", "Precision Recall", "Error", "Class Report", "RFE", "Learning", "Validation Curve", "Feature Importance", "Calibration", "Dimension", "Boundaries", "Lift", "Gain", "KS", "Parameter"])
        with tab1:
            st.write('🔹Plot of the best model Manifold')
            try:
                plot_model(selected_model, plot = 'manifold', save = True)
                st.image('Manifold Learning.png')
            except:
                try:
                    plot_model(best_model, plot = 'manifold', save = True)
                    st.image('Manifold Learning.png')
                except:
                    st.write('No Manifold Learning plot for this model')
        with tab2:
            st.write('🔹Plot of the best model Interpretation')
            try:
                interpret_model(selected_model, save = True)
                st.image('SHAP summary.png')
            except:
                try:
                    interpret_model(best_model, save = True)
                    st.image('SHAP summary.png')
                except:
                    st.write('No Interpretation plot for this model')
        with tab3:
            st.write('🔹Plot of the best model AUC')
            try:
                plot_model(selected_model, plot = 'auc', save = True)
                st.image('AUC.png')
            except:
                try:
                    plot_model(best_model, plot = 'auc', save = True)
                    st.image('AUC.png')
                except:
                    st.write('No AUC plot for this model')
        with tab4:
            st.write('🔹Plot of the best model Confusion Matrix')
            try:
                plot_model(selected_model, plot = 'confusion_matrix', save = True)
                st.image('Confusion Matrix.png')
            except:
                try:
                    plot_model(best_model, plot = 'confusion_matrix', save = True)
                    st.image('Confusion Matrix.png')
                except:
                    st.write('No Confusion Matrix plot for this model')
        with tab5:
            st.write('🔹Plot of the best model Threshold')
            # try:
            #     plot_model(selected_model, plot = 'threshold', save = True)
            #     st.image('Threshold.png')
            # except:
            #     try:
            #         plot_model(best_model, plot = 'threshold', save = True)
            #         st.image('Threshold.png')
            #     except:
            #         st.write('No Threshold plot for this model')
        with tab6:
            st.write('🔹Plot of the best model Precision Recall')
            try:
                plot_model(selected_model, plot = 'pr', save = True)
                st.image('Precision Recall.png')
            except:
                try:
                    plot_model(best_model, plot = 'pr', save = True)
                    st.image('Precision Recall.png')
                except:
                    st.write('No Precision Recall plot for this model')
        with tab7:
            st.write('🔹Plot of the best model Error')
            try:
                plot_model(selected_model, plot = 'error', save = True)
                st.image('Error.png')
            except:
                try:
                    plot_model(best_model, plot = 'error', save = True)
                    st.image('Error.png')
                except:
                    st.write('No Error plot for this model')
        with tab8:
            st.write('🔹Plot of the best model Class Report')
            try:
                plot_model(selected_model, plot = 'class_report', save = True)
                st.image('Class Report.png')
            except:
                try:
                    plot_model(best_model, plot = 'class_report', save = True)
                    st.image('Class Report.png')
                except:
                    st.write('No Class Report plot for this model')
        with tab9:
            st.write('🔹Plot of the best model RFE')
            try:
                plot_model(selected_model, plot = 'rfe', save = True)
                st.image('RFE.png')
            except:
                try:
                    plot_model(best_model, plot = 'rfe', save = True)
                    st.image('RFE.png')
                except:
                    st.write('No RFE plot for this model')
        with tab10:
            st.write('🔹Plot of the best model Learning')
            try:
                plot_model(selected_model, plot = 'learning', save = True)
                st.image('Learning Curve.png')
            except:
                try:
                    plot_model(best_model, plot = 'learning', save = True)
                    st.image('Learning Curve.png')
                except:
                    st.write('No Learning Curve plot for this model')
        with tab11:
            st.write('🔹Plot of the best model Validation Curve')
            # try:
            #     plot_model(selected_model, plot = 'vc', save = True)
            #     st.image('Validation Curve.png')
            # except:
            #     try:
            #         plot_model(best_model, plot = 'vc', save = True)
            #         st.image('Validation Curve.png')
            #     except:
            #         st.write('No Validation Curve plot for this model')
        with tab12:
            st.write('🔹Plot of the best model Feature Importance')
            try:
                plot_model(selected_model, plot = 'feature', save = True)
                st.image('Feature Importance.png')
            except:
                try:
                    plot_model(best_model, plot = 'feature', save = True)
                    st.image('Feature Importance.png')
                except:
                    st.write('No Feature Importance plot for this model')
        with tab13:
            st.write('🔹Plot of the best model Calibration')
            try:
                plot_model(selected_model, plot = 'calibration', save = True)
                st.image('Calibration.png')
            except:
                try:
                    plot_model(best_model, plot = 'calibration', save = True)
                    st.image('Calibration.png')
                except:
                    st.write('No Calibration plot for this model')
        with tab14:
            st.write('🔹Plot of the best model Dimension')
            try:
                plot_model(selected_model, plot = 'dimension', save = True)
                st.image('Dimension Learning.png')
            except:
                try:
                    plot_model(best_model, plot = 'dimension', save = True)
                    st.image('Dimension Learning.png')
                except:
                    st.write('No Dimension Learning plot for this model')
        with tab15:
            st.write('🔹Plot of the best model Boundaries')
            try:
                plot_model(selected_model, plot = 'boundary', save = True)
                st.image('Decision Boundary.png')
            except:
                try:
                    plot_model(best_model, plot = 'boundary', save = True)
                    st.image('Decision Boundary.png')
                except:
                    st.write('No Decision Boundary plot for this model')
        with tab16:
            st.write('🔹Plot of the best model Lift')
            try:
                plot_model(selected_model, plot = 'lift', save = True)
                st.image('Lift Chart.png')
            except:
                try:
                    plot_model(best_model, plot = 'lift', save = True)
                    st.image('Lift Chart.png')
                except:
                    st.write('No Lift Chart plot for this model')
        with tab17:
            st.write('🔹Plot of the best model Gain')
            try:
                plot_model(selected_model, plot = 'gain', save = True)
                st.image('Gain Chart.png')
            except:
                try:
                    plot_model(best_model, plot = 'gain', save = True)
                    st.image('Gain Chart.png')
                except:
                    st.write('No Gain Chart plot for this model')
        with tab18:
            st.write('🔹Plot of the best model KS')
            try:
                plot_model(selected_model, plot = 'ks', save = True)
                st.image('KS Chart.png')
            except:
                try:
                    plot_model(best_model, plot = 'ks', save = True)
                    st.image('KS Chart.png')
                except:
                    st.write('No KS Chart plot for this model')
        with tab19:
            st.write('🔹Plot of the best model Parameter')
            try:
                plot_model(selected_model, plot = 'parameter', save = True)
                st.image('Parameter.png')
            except:
                try:
                    plot_model(best_model, plot = 'parameter', save = True)
                    st.image('Parameter.png')
                except:
                    st.write('No Parameter plot for this model')
        
        st.text("🔹Predicting on the validation data")
        val_df['Predictions'] = predict_model(selected_model, data = val_df)['Label']
        val_df.to_csv('predictions.csv', index = False)

        # st.text("🔹Deep check of the model")
        # deep_check = deep_check(selected_model)
        # st.write(deep_check) 

        st.text("🔹Checking the model for fairness")
        try:
            fair = check_fairness(selected_model, sensitive_features = chosen_categorical_features)
            st.write(fair)
        except:
            st.write('No fairness check for this model')
        
        st.text("🔹Saving the one with the best metric")
        save_model(selected_model, 'best_model')

        st.text("🔹Registering the model to aws")
        deploy_model(selected_model, 'deployment', platform = 'aws', authentication = { 'bucket'  : 'pycaret-test' })
        # st.text("Loading the model")
        # saved_model = load_model('deployment')
        # st.text("Predicting on the test data")
        # predictions = predict_model(saved_model, data = test_data)
        

if choice == "Download": 
    with open('best_model.pkl', 'rb') as f: 
        st.download_button('Download Model', f, file_name="best_model.pkl")
    with open('predictions.csv', 'rb') as f:
        st.download_button('Download Predictions', f, file_name="predictions.csv")