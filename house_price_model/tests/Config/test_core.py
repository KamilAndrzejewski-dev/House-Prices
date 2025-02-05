from unittest.mock import patch, MagicMock

import pytest
from pathlib import Path
from strictyaml import YAML
from house_price_model.Config.core import load_yaml_file, load_and_validate_config, CONFIG_FILE_PATH, AppConfig, ModelConfig
from pydantic import ValidationError

class TestCore:

    @pytest.fixture
    def valid_config(self, tmp_path):
        config = """
        model: test_model
        dataset: data.csv
        preprocessing_pipeline_save_file: preprocess.pkl
        model_pipeline_save_file: model.pkl
        categorical_vars_with_na_frequent:
            - var1
        categorical_vars_imputing_with_missing:
            - var2
        variables_to_rename: 
            1stFlrSF: FirstFlrSF
        temporal_variables:
            - YearBuilt
        reference_var: YearRemodAdd
        log_transformation_variables:
            - GrLivArea
        binarize_transformation_variables:
            - CentralAir
        quality_variables:
            - ExterQual
        exposure_variables:
            - BsmtExposure
        garagefinish_variables:
            - GarageFinish
        categorical_variables_to_encode:
            - Neighborhood
        quality_mapping: 
            Ex: 4
            Gd: 5
        exposure_mapping: 
            Av: 1
            Mn: 2
        
        garagefinish_mapping: 
            Fin: 3 
            RFn: 2
        features: 
            - FeatureOne
            - FeatureTwo
        alpha: 0.5
        test_size: 0.2
        random_state: 42
        target: SalePrice
        """
        config_path = tmp_path / 'valid.yaml'
        config_path.write_text(config)

        return config_path

    @pytest.fixture
    def invalid_config(self, tmp_path):
        invalid_config = """
        model: test_model
        dataset: data.csv
        preprocessing_pipeline_save_file: 4324
        model_pipeline_save_file: model.pkl
        categorical_vars_with_na_frequent: 312
        categorical_vars_imputing_with_missing: var2
        variables_to_rename: 
            old_name: new_name
        temporal_variables: YearBuilt
        reference_var: YearRemodAdd
        log_transformation_variables: GrLivArea
        binarize_transformation_variables: CentralAir
        quality_variables: ExterQual
        exposure_variables: BsmtExposure
        garagefinish_variables: GarageFinish
        categorical_variables_to_encode: Neighborhood
        alpha: 0.5
        test_size: 0.2
        random_state: 42
        target: SalePrice
        """

        config_path = tmp_path / 'invalid.yaml'
        config_path.write_text(invalid_config)

        return config_path


    def test_load_yaml_file_valid_config(self, valid_config):
        yaml = load_yaml_file(valid_config)

        assert isinstance(yaml, YAML)
        assert yaml.data['dataset'] == 'data.csv'
        assert yaml.data['test_size'] == '0.2'

    def test_if_load_yaml_file_raises_error_if_file_was_not_found(self):
        not_existent_yaml_file = Path('not_found.yaml')
        result = load_yaml_file(not_existent_yaml_file)
        assert result is None


    def test_load_and_validate_valid_config(self, valid_config):
        config = load_and_validate_config(config_yaml=valid_config)

        assert isinstance(config.config_app, AppConfig)
        assert isinstance(config.config_model, ModelConfig)
        assert config.config_app.model == 'test_model'
        assert config.config_model.alpha == 0.5

    def test_load_and_validate_invalid_config(self, invalid_config):
        with pytest.raises(ValidationError):
            load_and_validate_config(config_yaml=invalid_config)




