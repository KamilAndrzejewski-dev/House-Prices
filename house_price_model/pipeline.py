from sklearn.preprocessing import MinMaxScaler
from house_price_model.Config.core import _config
from sklearn.pipeline import Pipeline
import house_price_model.Preprocessing.methods as F
from feature_engine.selection import DropFeatures
import numpy as np
from sklearn.linear_model import Lasso


# Pipeline that process data in sequential order
pipeline = Pipeline(
    [
        (
            'missing_imputation', F.CustomSimpleImpute(
                imputation='constant',
                variables=_config.config_model.categorical_vars_imputing_with_missing,
                fill_values='Missing'
            ),
        ),
        (
            'most_frequent_imputation', F.CustomSimpleImpute(
                imputation='most_frequent',
                variables=_config.config_model.categorical_vars_with_na_frequent,
            ),
        ),
        (
            'time_elapsing', F.TemporalVariableTransformer(
                variables=_config.config_model.temporal_variables,
                reference_str=_config.config_model.reference_var,
            ),
        ),
        (
            'drop_features', DropFeatures(
                features_to_drop=[_config.config_model.reference_var]
            ),
        ),
        (
            'log_transform', F.MathFunctionTransformer(
                variables=_config.config_model.log_transformation_variables,
                func='log'
            ),
        ),
        (
            'binazer', F.CustomBinarizer(
                variables=_config.config_model.binarize_transformation_variables,
                threshold=0
            ),
        ),
        (
            'quality_mapper', F.Mapper(
                variables=_config.config_model.quality_variables,
                mapping=_config.config_model.quality_mapping
            ),
        ),
        (
            'exposure_mapping', F.Mapper(
                variables=_config.config_model.exposure_variables,
                mapping=_config.config_model.exposure_mapping
            ),
        ),
        (
            'garage_mapping', F.Mapper(
                variables=_config.config_model.garagefinish_variables,
                mapping=_config.config_model.garagefinish_mapping
            ),
        ),
        (
            'rare_labels_encoders', F.RareLabelsEncoder(
                variables=_config.config_model.categorical_variables_to_encode,
                threshold=0.01
            ),
        ),
        (
            'monotonic_encoder', F.MonotonicOrdinalEncoder(
                variables=_config.config_model.categorical_variables_to_encode,
                method='mean'
            ),
        ),
        (
            'min_man_scaler', MinMaxScaler()
        ),
        (
            'lasso', Lasso(
                alpha=_config.config_model.alpha,
                random_state=_config.config_model.random_state)
        ),
    ]
)