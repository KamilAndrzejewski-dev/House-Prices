import pandas as pd
import pytest
from unittest.mock import patch
from sklearn.model_selection import train_test_split
from house_price_model.PipelineModel import model_pipeline
from house_price_model.PreprocessingPipeline import preprocessing_pipeline
from house_price_model.train_pipeline import train_pipeline
from house_price_model.Config.core import _config
class TestTrainPipeline:

    @patch('house_price_model.train_pipeline.save_pipeline')
    @patch.object(model_pipeline, 'fit')
    @patch.object(preprocessing_pipeline, 'transform')
    @patch.object(preprocessing_pipeline, 'fit')
    @patch('numpy.log')
    @patch('house_price_model.train_pipeline.train_test_split')
    @patch('house_price_model.train_pipeline._config.config_model.target')
    @patch('house_price_model.train_pipeline._config.config_model.features')
    @patch.object(pd.DataFrame, 'rename')
    @patch('house_price_model.train_pipeline.load_datasets')
    def test_train_pipeline(self,
                            mock_load,
                            mock_rename,
                            mock_config,
                            mock_target,
                            mock_split,
                            mock_log,
                            mock_fit_preprocessing,
                            mock_transform_preprocessing,
                            mock_fit_model_pipe,
                            mock_save_pipeline):

        mock_load.return_value = pd.DataFrame({
            'feature1': [1, 2], 'feature2': [3, 4], 'target': [5, 6]
        })

        mock_config.return_values = ['feature1', 'feature2', 'target']
        mock_target.return_values = 'target'

        mock_split.return_value = (
            pd.DataFrame({'feature1': [1], 'feature2': [3]}),
            pd.DataFrame({'feature1': [2], 'feature2': [4]}),
            pd.Series([5]),
            pd.Series([6])
        )

        train_pipeline()


        mock_load.assert_called_once()
        mock_rename.assert_called_once()
        mock_split.assert_called_once()
        mock_log.assert_called_once()
        mock_fit_preprocessing.assert_called_once()
        mock_transform_preprocessing.assert_called_once()
        mock_fit_model_pipe.assert_called_once()
        mock_save_pipeline.assert_called_once()


