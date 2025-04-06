from typing import Any, Dict


class Store:
    def __init__(self):
        self.models: Dict[str, Any] = {}
        self.results: Dict[str, Any] = {}

    def get_model(self, model_id: str) -> Any:
        if model_id not in self.models:
            raise KeyError(f"Model with id {model_id} not found")
        return self.models[model_id]

    def set_model(self, model_id: str, model: Any):
        self.models[model_id] = model

    def get_results(self, model_id: str) -> Dict[str, Any]:
        if model_id not in self.results:
            raise KeyError(f"Results for model with id {model_id} not found")
        return self.results[model_id]

    def set_results(self, model_id: str, results: Dict[str, Any]):
        self.results[model_id] = results
