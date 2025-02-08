import pandas as pd
import pytest
from house_price_model.predict import predict

class TestPredictions:

    @pytest.fixture
    def dataframe(self):
        return pd.DataFrame(
            {
                'FirstFlrSF': [856, 1262, 920, 756, 1145],
                'BldgType': ['1Fam', '1Fam', '1Fam', '2fmCon', '1Fam'],
                'BsmtCond': ['TA', 'Gd', 'TA', 'Fa', 'Ex'],
                'BsmtExposure': ['No', 'Gd', 'Mn', 'No', 'Av'],
                'BsmtFinSF1': [500, 600, 400, 300, 700],
                'BsmtQual': ['Gd', 'TA', 'Gd', 'Fa', 'Ex'],
                'CentralAir': ['Y', 'Y', 'Y', 'N', 'Y'],
                'Condition1': ['Norm', 'Feedr', 'Norm', 'Artery', 'Norm'],
                'ExterQual': ['Gd', 'TA', 'Gd', 'Fa', 'Ex'],
                'ExterCond': ['TA', 'Gd', 'TA', 'Fa', 'Ex'],
                'FireplaceQu': ['Gd', 'TA', 'None', 'Fa', 'Ex'],
                'Foundation': ['PConc', 'CBlock', 'PConc', 'BrkTil', 'PConc'],
                'Functional': ['Typ', 'Typ', 'Min1', 'Min2', 'Typ'],
                'GarageCond': ['TA', 'Gd', 'TA', 'Fa', 'Ex'],
                'GarageFinish': ['Fin', 'Unf', 'Fin', 'Unf', 'Fin'],
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
                'OpenPorchSF': [200, 150, 180, 100, 250],
                'PavedDrive': ['Y', 'Y', 'Y', 'N', 'Y'],
                'RoofStyle': ['Gable', 'Hip', 'Gable', 'Gable', 'Hip'],
                'SaleCondition': ['Normal', 'Normal', 'Normal', 'Abnorml', 'Normal'],
                'ScreenPorch': [50, 30, 40, 20, 60],
                'WoodDeckSF': [300, 200, 250, 150, 400],
                'YearRemodAdd': [2003, 1976, 2001, 1915, 2000],
                'YrSold': [2010, 2011, 2012, 2013, 2014]
            }
        )

    @pytest.mark.parametrize("input_data", (dict, list, 312, "string", bool))
    def test_if_predict_raises_error_if_input_is_not_a_dataframe(self, input_data):
        with pytest.raises(TypeError):
            predict(input_data)

    def test_if_predict_returns_dictionary(self, dataframe):
        result = predict(input_data=dataframe)
        assert isinstance(result, dict)

    def test_if_predict_returns_result_column_types_correctly(self, dataframe):
        result = predict(input_data=dataframe)

        assert isinstance(result['prediction'], list)
        assert isinstance(result['errors'], list)
        assert isinstance(result['version'], str)

    def test_if_predict_predicts_correctly(self, dataframe):
        result = predict(input_data=dataframe)

        assert len(result['prediction']) == 5
        assert result['prediction'][0] == pytest.approx(219678, abs=1)
