# ml_handler.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import shap
import numpy as np

class MLHandler:
    def __init__(self, dataframe: pd.DataFrame):
        self.df = dataframe
        # Assume the last column is the target variable
        self.target_column = self.df.columns[-1]
        self.X = self.df.drop(self.target_column, axis=1)
        self.y = self.df[self.target_column]
        self.best_model = None
        self.preprocessor = self._create_preprocessor()
        self.feature_names = None
        self.explainer = None

    def _create_preprocessor(self):
        numeric_features = self.X.select_dtypes(include=np.number).columns.tolist()
        categorical_features = self.X.select_dtypes(exclude=np.number).columns.tolist()

        numeric_transformer = StandardScaler()
        categorical_transformer = OneHotEncoder(handle_unknown='ignore')

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ],
            remainder='passthrough'
        )
        return preprocessor

    def get_feature_names(self):
        return self.feature_names

    def run_analysis(self):
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42, stratify=self.y)

        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
            "Random Forest": RandomForestClassifier(random_state=42),
            "XGBoost Classifier": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
        }

        results = []
        best_accuracy = 0.0

        for name, model in models.items():
            pipeline = Pipeline(steps=[('preprocessor', self.preprocessor), ('classifier', model)])
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)
            
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

            results.append({
                "name": name, "accuracy": round(accuracy, 3),
                "precision": round(precision, 3), "f1_score": round(f1, 3)
            })

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                self.best_model = pipeline
                self.best_model_name = name

        try:
            ohe_feature_names = self.best_model.named_steps['preprocessor'].named_transformers_['cat'].get_feature_names_out()
            num_feature_names = self.X.select_dtypes(include=np.number).columns.tolist()
            self.feature_names = num_feature_names + list(ohe_feature_names)
        except Exception:
            self.feature_names = self.X.columns.tolist()

        X_train_transformed = self.best_model.named_steps['preprocessor'].transform(X_train)
        self.explainer = shap.TreeExplainer(self.best_model.named_steps['classifier'], X_train_transformed)
        
        results.sort(key=lambda x: x['accuracy'], reverse=True)

        return {
            "bestModel": self.best_model_name,
            "models": results,
            "features": self.X.columns.tolist()
        }

    def predict_and_explain(self, input_df: pd.DataFrame):
        input_df = input_df[self.X.columns]

        prediction_proba = self.best_model.predict_proba(input_df)[0]
        prediction_class = self.best_model.predict(input_df)[0]
        
        class_index = list(self.best_model.classes_).index(prediction_class)
        confidence = prediction_proba[class_index]

        input_transformed = self.best_model.named_steps['preprocessor'].transform(input_df)
        shap_values = self.explainer.shap_values(input_transformed)
        
        shap_values_for_output = shap_values[class_index] if isinstance(shap_values, list) else shap_values

        importance = pd.DataFrame(
            list(zip(self.feature_names, np.abs(shap_values_for_output[0]))),
            columns=['feature', 'value']
        )
        importance['value'] = importance['value'] / importance['value'].sum()
        importance = importance.sort_values('value', ascending=False).head(5)
        
        return {
            "outcome": str(prediction_class),
            "confidence": round(float(confidence), 3)
        }, importance.to_dict('records')
