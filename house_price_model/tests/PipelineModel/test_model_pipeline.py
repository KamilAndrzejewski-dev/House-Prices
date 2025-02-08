import pytest
from sklearn.linear_model import Lasso
from house_price_model.PipelineModel import model_pipeline
from sklearn.pipeline import Pipeline
from house_price_model.Config.core import _config

class TestModelPipeline:

    def test_if_model_pipeline_is_pipeline(self):
        assert isinstance(model_pipeline, Pipeline)

    def test_if_model_pipeline_has_lasso_step(self):
        step = dict(model_pipeline.steps)
        assert 'lasso' in step

    def test_if_model_pipeline_has_lasso_model(self):
        step = dict(model_pipeline.steps)
        assert isinstance(step['lasso'], Lasso)

    def test_if_model_pipeline_lasso_parameters_are_equal_as_in_config_setup(self):
        step = dict(model_pipeline.steps)

        assert step['lasso'].alpha == _config.config_model.alpha
        assert step['lasso'].random_state == _config.config_model.random_state