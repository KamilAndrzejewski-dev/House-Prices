import pytest
import numpy as np
import pandas as pd
from click.parser import split_opt

from house_price_model.Preprocessing.data_manager import load_datasets
from house_price_model.PreprocessingPipeline import preprocessing_pipeline
from house_price_model.Config.core import _config
class TestPreprocessingPipeline:

    @pytest.fixture
    def dataframe(self):
        return pd.DataFrame({
            'FirstFlrSF': [856, 1262, 920, 756, 1145],
            'BldgType': ['1Fam', '2FmCon', '1Fam', 'Duplex', '1Fam'],
            'BsmtCond': ['TA', np.nan, 'TA', 'Fa', 'Ex'],
            'BsmtExposure': ['No', np.nan, 'Av', 'No', 'Gd'],
            'BsmtFinSF1': [0, 0, 400, 300, 700],
            'BsmtQual': ['Gd', np.nan, 'Gd', 'Fa', 'Ex'],
            'CentralAir': ['Y', 'Y', 'Y', 'N', 'Y'],
            'Condition1': ['Norm', 'Feedr', 'Norm', 'Artery', 'Norm'],
            'ExterQual': ['Gd', 'TA', 'Gd', 'Fa', 'Ex'],
            'ExterCond': ['TA', 'Gd', 'TA', 'Fa', 'Ex'],
            'FireplaceQu': [np.nan, 'TA', 'TA', 'Fa', 'Ex'],
            'Foundation': ['PConc', 'CBlock', 'PConc', 'BrkTil', 'PConc'],
            'Functional': ['Typ', 'Typ', 'Min1', 'Min2', 'Typ'],
            'GarageCond': ['TA', np.nan, 'TA', 'Fa', 'Ex'],
            'GarageFinish': ['Fin', np.nan, 'Fin', 'Unf', 'Fin'],
            'GrLivArea': [1650, 1800, 1700, 1400, 2100],
            'HeatingQC': ['Ex', 'Gd', 'TA', 'Fa', 'Ex'],
            'KitchenQual': ['Gd', 'TA', 'Gd', 'Gd', 'Ex'],
            'LandSlope': ['Gtl', 'Mod', 'Gtl', 'Gtl', 'Mod'],
            'LotArea': [8450, 9600, 11250, 9550, 14260],
            'LotConfig': ['Inside', 'Corner', 'Inside', 'CulDSac', 'Inside'],
            'LotShape': ['Reg', 'IR1', 'Reg', 'IR2', 'Reg'],
            'MSSubClass': [60, 70, 60, 50, 80],
            'MSZoning': ['RL', 'RM', 'RL', 'RM', 'RL'],
            'Neighborhood': ['CollgCr', 'Veenker', 'CollgCr', 'Crawfor', 'NoRidge'],
            'OpenPorchSF': [200, 0, 0, 100, 250],
            'PavedDrive': ['Y', 'Y', 'Y', 'N', 'Y'],
            'RoofStyle': ['Gable', 'Hip', 'Gable', 'Gable', 'Hip'],
            'SaleCondition': ['Normal', 'Normal', 'Normal', 'Abnorml', 'Normal'],
            'ScreenPorch': [0, 30, 0, 20, 60],
            'WoodDeckSF': [0, 200, 250, 150, 0],
            'YearRemodAdd': [2003, 1976, 2001, 1915, 2000],
            'YrSold': [2010, 2011, 2012, 2013, 2014],
            'SalePrice': [199320, 221002, 168832, 320002, 213000]
})

    @pytest.fixture
    def split_data(self, dataframe):
        X_train = dataframe.drop('SalePrice', axis=1)
        y_train = dataframe['SalePrice']
        return X_train, y_train

    pd.set_option('display.max_columns', None)




    def test_if_preprocessing_pipeline_removes_all_missing_values(self, split_data):
        X_train, y_train = split_data
        transformed_data = preprocessing_pipeline.fit_transform(X_train, y_train)

        assert transformed_data.isnull().sum().sum() == 0


    def test_if_preprocessing_pipeline_features_values_ranges_between_zero_and_one(self, dataframe, split_data):
        X_train, y_train = split_data
        transformed_data = preprocessing_pipeline.fit_transform(X_train, y_train)
        print(transformed_data)

        for col in transformed_data.columns:
            assert (0 <= transformed_data[col]).all() and (transformed_data[col] <= 1.01).all()



    @pytest.fixture
    def test_pipeline_steps(self, split_data):
        def _test_pipeline_steps(method):
            X_train, y_train = split_data
            transformed_data = preprocessing_pipeline.named_steps[method].fit_transform(X_train, y_train)
            return transformed_data
        return _test_pipeline_steps

    def test_if_preprocessing_pipeline_categorical_variables_contains_missing_after_transformation(self, test_pipeline_steps):
        transformed_data = test_pipeline_steps(method='missing_imputation')
        cols_with_missing = ['BsmtQual', 'BsmtCond', 'BsmtExposure', 'GarageFinish', 'GarageCond']
        assert all("Missing" in transformed_data[col].values for col in cols_with_missing)

    def test_if_preprocessing_pipeline_transforms_inputs_most_frequent_int_cat_var_with_nan_frequent(self, split_data, test_pipeline_steps):
        transformed_data = test_pipeline_steps(method='most_frequent_imputation')
        assert transformed_data['FireplaceQu'].mode()[0] == 'TA'

    def test_if_preprocessing_pipeline_transforms_correctly_temporal_variables(self, test_pipeline_steps, ):
        original_data = test_pipeline_steps(method='missing_imputation')
        transformed_data = test_pipeline_steps(method='log_transform')
        cols_after_log = ['LotArea', 'GrLivArea']
        assert not all(original_data[col].equals(transformed_data[col]) for col in cols_after_log)

    def test_if_preprocessing_pipeline_transforms_correctly_binarize_variables(self, test_pipeline_steps):
        transformed_data = test_pipeline_steps(method='binarizer')
        cols_after_binarize_transformation = ['BsmtFinSF1', 'WoodDeckSF', 'OpenPorchSF', 'ScreenPorch']
        assert all(transformed_data[col].unique() in (0, 1)  for col in cols_after_binarize_transformation)

    def test_if_preprocessing_pipeline_maps_correctly(self, test_pipeline_steps):
        transformed_data_quality = test_pipeline_steps(method='quality_mapper')
        transformed_data_exposure = test_pipeline_steps(method='exposure_mapping')
        transformed_data_garage = test_pipeline_steps(method='garage_mapping')
        transformed_data_monotonic = test_pipeline_steps(method='monotonic_encoder')

        cols_quality_mapping = [
            'ExterQual',
            'ExterCond',
            'BsmtQual',
            'BsmtCond',
            'HeatingQC',
            'KitchenQual',
            'FireplaceQu',
            'GarageCond'
        ]

        cols_monotonic_mapping = [
            'MSZoning',
            'LotShape',
            'LandSlope',
            'LotConfig',
            'Neighborhood',
            'Condition1',
            'BldgType',
            'RoofStyle',
            'Foundation',
            'CentralAir',
            'Functional',
            'PavedDrive',
            'SaleCondition',
            'MSSubClass'
        ]


        assert all(pd.api.types.is_numeric_dtype(transformed_data_quality[col]) for col in cols_quality_mapping)
        assert pd.api.types.is_numeric_dtype(transformed_data_exposure['BsmtExposure'])
        assert pd.api.types.is_numeric_dtype(transformed_data_garage['GarageFinish'])
        assert all(pd.api.types.is_numeric_dtype(transformed_data_monotonic[col]) for col in cols_monotonic_mapping)


    def test_if_preprocessing_pipeline_encodes_correctly_rare_labels(self, test_pipeline_steps):
        data = load_datasets(mode='train')
        X_train, y_train = data.drop('SalePrice', axis=1), data['SalePrice']
        transformed_data = preprocessing_pipeline.named_steps['rare_labels_encoders'].fit_transform(X_train, y_train)

        categorical_features = [
            'MSZoning',
            'LotShape',
            'LandSlope',
            'LotConfig',
            'Neighborhood',
            'Condition1',
            'BldgType',
            'RoofStyle',
            'Foundation',
            'CentralAir',
            'Functional',
            'PavedDrive',
            'SaleCondition',
            'MSSubClass'
        ]

        assert any("Rare" in transformed_data[col].values for col in categorical_features)



    def test_if_preprocessing_pipeline_scales_correctly_variables(self, test_pipeline_steps, split_data):
        X_train, y_train = split_data
        transformed_data = preprocessing_pipeline.fit_transform(X_train, y_train)

        assert all(0 < transformed_data[col].mean() < 1 for col in transformed_data)
        assert all(0 < transformed_data[col].std() < 1 for col in transformed_data)


    def test_if_preprocessing_pipeline_drop_reference_var(self, test_pipeline_steps):

        transformed_data = test_pipeline_steps(method='drop_features')

        assert _config.config_model.reference_var not in transformed_data.columns
