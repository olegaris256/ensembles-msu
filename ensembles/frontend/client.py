from typing import Any
from io import BytesIO
import numpy.typing as npt
import requests
import json
from ensembles.utils import ConvergenceHistory
import numpy as np
from ensembles.backend.schemas import ExistingExperimentsResponse, ExperimentConfig, ConvergenceHistoryResponse

class Client:
    def __init__(self, base_url: str) -> None:
        """
        Initializes the Client with a base URL for the API.

        Args:
            base_url (str): The base URL of the API.
        """

        self.base_url = base_url
        self.session = requests.Session()

    def get_names(self) -> list[str]:
        """
        Retrieves the names of all existing experiments.

        Returns:
            list[str]: A list of experiment names.
        """

        response = self.session.get(f"{self.base_url}/existing_experiments/")
        response.raise_for_status()
        return response.json()["experiment_names"]

    def register_experiment(self, experiment_config, train_file: BytesIO) -> None:
        """
        Registers a new experiment with the given configuration and training data.

        Args:
            experiment_config (Any): The configuration for the experiment.
            train_file (Any): The training data file.
        """
        response = self.session.post(
            f"{self.base_url}/registered_experiment/",
            files={'train_file': train_file.getvalue()},
            data={'experiment_config': json.dumps(experiment_config.model_dump())}
        )
        response.raise_for_status()
        return response

    def load_experiment_config(self, experiment_name) -> dict[str, Any]:
        """
        Loads the configuration of an existing experiment.

        Args:
            experiment_name (Any): The name of the experiment.

        Returns:
            ExperimentConfig: The configuration of the experiment.
        """
        response = self.session.get(
            f"{self.base_url}/experiment_config/{experiment_name}"
        )
        response.raise_for_status()
        return ExperimentConfig(**response.json())

    def is_training_needed(self, experiment_name) -> bool:
        """
        Request info about was the model ever trained.

        Args:
            experiment_name (Any): The name of the experiment.
        
        Returns:
            bool: indicator was the model ever trained.
        """
        # get needs_training
        ...

    def train_model(self, experiment_name) -> None:
        """
        Trains the model for the specified experiment.

        Args:
            experiment_name (Any): The name of the experiment.
        """
        # post training_model
        response = self.session.post(
            f"{self.base_url}/training_model", params={"experiment_name": experiment_name}
        )
        response.raise_for_status()
        

    def get_convergence_history(self, experiment_name) -> ConvergenceHistoryResponse:
        """
        Retrieves the convergence history of the specified experiment.

        Args:
            experiment_name (Any): The name of the experiment.

        Returns:
            ConvergenceHistory: The convergence history of the experiment.
        """
        # get convergence_history
        response = self.session.get(
            f"{self.base_url}/convergence_history/{experiment_name}"
        )
        return ConvergenceHistoryResponse(**response.json())

    def predict(self, experiment_name, test_file) -> npt.NDArray[Any]:
        """
        Makes predictions using the trained model of the specified experiment.

        Args:
            experiment_name (Any): The name of the experiment.
            test_file (Any): The test data file.

        Returns:
            npt.NDArray[Any]: The predictions made by the model.
        """
        response = self.session.get(
            f"{self.base_url}/predicted_values/{experiment_name}",
            files={'test_file': test_file}
        )
        response.raise_for_status()
        return np.array(response.json())