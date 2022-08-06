import logging
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, FunctionTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.base import BaseEstimator, TransformerMixin


def delta_date_feature(dates):
    """
    Given a 2d array containing dates (in any format recognized by pd.to_datetime), it returns the delta in days
    between each date and the most recent date in its column
    """
    date_sanitized = pd.DataFrame(dates).apply(pd.to_datetime)
    return date_sanitized.apply(lambda d: (d.max() - d).dt.days, axis=0).to_numpy()


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


class Debugger(BaseEstimator, TransformerMixin):
    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame):
        print(np.argwhere(np.isnan(X)))
        return X


class PandasDebugger(BaseEstimator, TransformerMixin):
    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame):
        print(X.isnull().any())
        print(X.dtypes)
        return X


def get_inference_pipeline(rf_config, max_tfidf_features):
    # Let's handle the categorical features first
    # Ordinal categorical are categorical values for which the order is meaningful, for example
    # for room type: 'Entire home/apt' > 'Private room' > 'Shared room'
    ordinal_categorical = ["room_type"]
    non_ordinal_categorical = ["neighbourhood_group"]
    # NOTE: we do not need to impute room_type because the type of the room
    # is mandatory on the websites, so missing values are not possible in production
    # (nor during training). That is not true for neighbourhood_group
    ordinal_categorical_preproc = OrdinalEncoder()

    ######################################
    # Build a pipeline with two steps:
    # 1 - A SimpleImputer(strategy="most_frequent") to impute missing values
    # 2 - A OneHotEncoder() step to encode the variable
    non_ordinal_categorical_preproc = make_pipeline(
        SimpleImputer(strategy="most_frequent"),
        OneHotEncoder()
    )
    ######################################

    # Let's impute the numerical columns to make sure we can handle missing values
    # (note that we do not scale because the RF algorithm does not need that)
    float_imputed = [
        "reviews_per_month",
    ]
    float_imputer = make_pipeline(
        SimpleImputer(strategy="constant", fill_value=0.0),
    )

    int_imputed = [
        "minimum_nights",
        "number_of_reviews",
        "calculated_host_listings_count",
        "availability_365"
    ]
    int_imputer = make_pipeline(
        SimpleImputer(strategy="constant", fill_value=0)
    )

    # A MINIMAL FEATURE ENGINEERING step:
    # we create a feature that represents the number of days passed since the last review
    # First we impute the missing review date with an old date (because there hasn't been
    # a review for a long time), and then we create a new feature from it,
    date_imputer = make_pipeline(
        SimpleImputer(missing_values=None, strategy='constant', fill_value='2010-01-01'),
        FunctionTransformer(delta_date_feature, check_inverse=False, validate=False)
    )

    # Some minimal NLP for the "name" column
    reshape_to_1d = FunctionTransformer(np.reshape, kw_args={"newshape": -1})
    name_tfidf = make_pipeline(
        SimpleImputer(missing_values=None, strategy="constant", fill_value="imputed name"),
        reshape_to_1d,
        TfidfVectorizer(
            binary=False,
            max_features=max_tfidf_features,
            stop_words='english'
        )
    )

    # Let's put everything together
    preprocessor = ColumnTransformer(
        transformers=[
            ("ordinal_cat", ordinal_categorical_preproc, ordinal_categorical),
            ("non_ordinal_cat", non_ordinal_categorical_preproc, non_ordinal_categorical),
            ("impute_float", float_imputer, float_imputed),
            ("impute_int", int_imputer, int_imputed),
            ("transform_date", date_imputer, ["last_review"]),
            ("transform_name", name_tfidf, ["name"])
        ],
        remainder="drop",  # This drops the columns that we do not transform
    )

    processed_features = ordinal_categorical + \
                         non_ordinal_categorical + \
                         float_imputed + \
                         int_imputed + \
                         ["last_review", "name"]

    # Create random forest
    random_forest = RandomForestRegressor(**rf_config)

    ######################################
    # Create the inference pipeline. The pipeline must have 2 steps: a step called "preprocessor" applying the
    # ColumnTransformer instance that we saved in the `preprocessor` variable, and a step called "random_forest"
    # with the random forest instance that we just saved in the `random_forest` variable.
    # HINT: Use the explicit Pipeline instead of make_pipeline constructor to leverage named step
    sk_pipe = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("random_forest", random_forest)
        ]
    )

    return sk_pipe, processed_features
