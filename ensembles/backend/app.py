from pathlib import Path

from ensembles.random_forest import RandomForestMSE
from ensembles.boosting import GradientBoostingMSE

from fastapi import FastAPI, UploadFile, Form, File, HTTPException

from .schemas import ExistingExperimentsResponse, ExperimentConfig
import json
from fastapi.responses import JSONResponse
from io import BytesIO

import pandas as pd


from sklearn.model_selection import train_test_split
app = FastAPI()


def get_runs_dir() -> Path:
    """
    Get the path to the directory where experiments are stored.

    Returns:
        Path: The path to the 'runs' directory.
    """
    return Path.cwd() / "runs"


@app.get("/existing_experiments/")
async def existing_experiments() -> ExistingExperimentsResponse:
    """
    Get information about existing experiments.

    This endpoint scans the directory where experiments are stored and returns a list of
    existing experiments along with their absolute paths. Each experiment is stored as
    a directory in the host filesystem.

    Returns:
        ExistingExperimentsResponse: A response containing the location of the experiments
        directory, absolute paths of the experiment directories, and the names of the experiments.
    """
    path = get_runs_dir()
    response = ExistingExperimentsResponse(location=path)
    if not path.exists():
        return response
    response.abs_paths = [obj for obj in path.iterdir() if obj.is_dir()]
    response.experiment_names = [filepath.stem for filepath in response.abs_paths]
    return response


@app.post("/registered_experiment/")
async def registered_experiment(train_file: UploadFile = File(...), experiment_config: str = Form(...)) -> None:
    """
    Register a new experiment by uploading a training file and experiment configuration.

    Args:
        train_file (UploadFile): The training file to be uploaded (CSV format).
        experiment_config (str): The experiment configuration in JSON format.

    Raises:
        HTTPException: If the uploaded file is empty.
    """
    experiment_config = ExperimentConfig.model_validate_json(experiment_config)

    exp_path = get_runs_dir() / experiment_config.name
    exp_path.mkdir(parents=True, exist_ok=True)
    with (exp_path / 'config.json').open('w') as f:
        json.dump(experiment_config.dict(), f)

    content = await train_file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Загруженный файл пустой")
    with (exp_path / 'train.csv').open('wb') as f:
        f.write(content)


@app.get("/experiment_config/{experiment_name}")
async def experiment_config(experiment_name: str):
    """
    Get the configuration of a specific experiment.

    Args:
        experiment_name (str): The name of the experiment.

    Returns:
        dict: The experiment configuration as a dictionary.
    """
    exp_path = get_runs_dir() / experiment_name
    config_path = exp_path / "config.json"
    with config_path.open('r') as f:
        config = json.load(f)
    return config


@app.get("/needs_training/")
async def needs_training(experiment_name: str) -> JSONResponse:
    """
    Check if a model for the specified experiment needs training.

    Args:
        experiment_name (str): The name of the experiment.

    Returns:
        JSONResponse: A JSON response indicating whether the model needs training.
    """
    exp_path = get_runs_dir() / experiment_name
    model_path = exp_path / "model"
    return JSONResponse(content={"response": not model_path.exists()})


@app.post("/training_model/")
async def training_model(experiment_name: str):
    """
    Train a model for the specified experiment.

    Args:
        experiment_name (str): The name of the experiment.
    """
    exp_path = get_runs_dir() / experiment_name

    train_path = exp_path / "train.csv"
    config_path = exp_path / "config.json"
    model_path = exp_path / "model"

    data = pd.read_csv(train_path)

    with config_path.open('r') as f:
        config = json.load(f)
    config = ExperimentConfig(**config)
    max_features = config.max_features if config.max_features != 'all' else None
    if config.ml_model == 'Random Forest':
        model = RandomForestMSE(n_estimators=config.n_estimators,
                                tree_params={
                                    'max_depth': config.max_depth,
                                    'max_features': max_features
                                })
    elif config.ml_model == 'Gradient Boosting':
        model = GradientBoostingMSE(n_estimators=config.n_estimators,
                                    tree_params={
                                        'max_depth': config.max_depth,
                                        'max_features': max_features
                                    })
    feature_columns = [col for col in data.columns if col != config.target_column]
    X_train, X_val, y_train, y_val = train_test_split(
        data[feature_columns], data[config.target_column]
    )
    history = model.fit(X_train, y_train, X_val=X_val, y_val=y_val, trace=True)
    model.dump(model_path)
    with (model_path / 'history.json').open('w') as f:
        json.dump(history, f)


@app.get("/convergence_history/{experiment_name}")
async def convergence_history(experiment_name: str):
    """
    Get the convergence history of a trained model for the specified experiment.

    Args:
        experiment_name (str): The name of the experiment.

    Returns:
        dict: The convergence history of the model.
    """
    exp_path = get_runs_dir() / experiment_name
    history_path = exp_path / 'model' / 'history.json'
    with history_path.open('r') as f:
        history = json.load(f)
    return history


@app.get("/predicted_values/{experiment_name}")
async def predicted_values(experiment_name: str, test_file: UploadFile = File(...)):
    """
    Get predicted values for a test dataset using a trained model.

    Args:
        experiment_name (str): The name of the experiment.
        test_file (UploadFile): The test dataset file to be uploaded (CSV format).

    Returns:
        JSONResponse: A JSON response containing the predicted values.
    """

    exp_path = get_runs_dir() / experiment_name

    config_path = exp_path / "config.json"
    model_path = exp_path / "model"

    with config_path.open('r') as f:
        config = json.load(f)
    config = ExperimentConfig(**config)
    if config.ml_model == 'Random Forest':
        model = RandomForestMSE.load(model_path)
    elif config.ml_model == 'Gradient Boosting':
        model = GradientBoostingMSE.load(model_path)

    test_file_content = await test_file.read()
    data_test = pd.read_csv(BytesIO(test_file_content))
    feature_columns = [col for col in data_test.columns if col != config.target_column]
    X_test = data_test[feature_columns].values
    predictions = model.predict(X_test)
    return JSONResponse(content=predictions.tolist())
