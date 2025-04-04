import numpy as np
from sklearn.model_selection import train_test_split
from house_price_model.Config.core import _config
from house_price_model.Preprocessing.data_manager import load_datasets, save_pipeline
from house_price_model.PreprocessingPipeline import  preprocessing_pipeline
from house_price_model.PipelineModel import model_pipeline


def train_pipeline() -> None:
    """Train Pipeline

    This function loads training dataset, renames variables that names start with number, splits data,
    perform logarithmic transformation on target, train and saves a pipeline in specified folder.
    """
    dataframe = load_datasets(mode='train')
    dataframe.rename(columns=_config.config_model.variables_to_rename, inplace=True)

    X_train, X_test, y_train, y_test = train_test_split(
        dataframe[_config.config_model.features],
        dataframe[_config.config_model.target],
        test_size=_config.config_model.test_size,
        random_state=_config.config_model.random_state
    )

    y_train = np.log(y_train)

    preprocessing_pipeline.fit(X_train, y_train)

    transformed_data = preprocessing_pipeline.transform(X_train)

    model_pipeline.fit(transformed_data, y_train)

    save_pipeline(model_pipeline=model_pipeline, transformer_pipeline=preprocessing_pipeline)


if __name__ == "__main__":
    train_pipeline()
