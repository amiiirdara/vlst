"""Type stubs for tabpfn (editor / Pylance only; runtime needs `pip install tabpfn`)."""

from typing import Any, Optional

class TabPFNClassifier:
    def __init__(
        self,
        *,
        n_estimators: int = ...,
        device: str = ...,
        ignore_pretraining_limits: bool = ...,
        balance_probabilities: bool = ...,
        random_state: Optional[int] = ...,
    ) -> None: ...
    def fit(self, X: Any, y: Any) -> TabPFNClassifier: ...
    def predict(self, X: Any, **kwargs: Any) -> Any: ...
    def predict_proba(self, X: Any) -> Any: ...
