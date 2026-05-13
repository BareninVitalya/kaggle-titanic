import torch
import torch.nn as nn
from typing import List
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


ACTIVATION_MAP = {
    "relu":       nn.ReLU,
    "leakyrelu":  nn.LeakyReLU,
    "gelu":       nn.GELU,
    "tanh":       nn.Tanh,
    "sigmoid":    nn.Sigmoid,
    "selu":       nn.SELU,
    "elu":        nn.ELU,
}


class TitanicMLP(nn.Module):
    """
    Гибкая MLP для задачи бинарной классификации (Titanic).

    Параметры
    ---------
    input_dim   : int        — размерность входного вектора признаков
    hidden_dims : List[int]  — список размеров скрытых слоёв, например [64, 32]
    activation  : str        — функция активации: relu | leakyrelu | gelu | tanh | selu | elu
    dropout     : float      — вероятность Dropout (0.0 = выключен)
    batchnorm   : bool       — BatchNorm1d после каждого скрытого слоя
    output_dim  : int        — 1 для BCEWithLogitsLoss (по умолчанию)

    Архитектура каждого hidden-блока:
        Linear -> [BatchNorm1d] -> Activation -> [Dropout]

    Пример (baseline 2 слоя):
        net = TitanicMLP(input_dim=9, hidden_dims=[64, 32])
        # Linear(9,64) -> ReLU -> Linear(64,32) -> ReLU -> Linear(32,1)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        activation: str = "relu",
        dropout: float = 0.0,
        batchnorm: bool = False,
        output_dim: int = 1,
    ):
        super().__init__()

        act_cls = ACTIVATION_MAP.get(activation.lower())
        if act_cls is None:
            print(
                f"Неизвестная активация '{activation}'. "
                f"Доступны: {list(ACTIVATION_MAP.keys())}"
            )

        layers: List[nn.Module] = []
        in_dim = input_dim

        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            if batchnorm:
                layers.append(nn.BatchNorm1d(h_dim))
            layers.append(act_cls())
            if dropout > 0.0:
                layers.append(nn.Dropout(p=dropout))
            in_dim = h_dim

        layers.append(nn.Linear(in_dim, output_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ─────────────────────────────────────────────────────────────────────────────
# Deep Neural Network (DNN) — helpers + sklearn wrapper + build function
# ─────────────────────────────────────────────────────────────────────────────

def _get_optimizer(name: str, parameters, lr: float, weight_decay: float):
    """Возвращает optimizer по строковому имени."""
    name = name.lower()
    if name == "adam":
        return torch.optim.Adam(parameters, lr=lr, weight_decay=weight_decay)
    if name == "adamw":
        return torch.optim.AdamW(parameters, lr=lr, weight_decay=weight_decay)
    if name == "sgd":
        return torch.optim.SGD(parameters, lr=lr, momentum=0.9,
                               weight_decay=weight_decay)
    raise ValueError(f"Неизвестный optimizer '{name}'. Доступны: adam, adamw, sgd")


def _get_scheduler(name, optimizer, params: dict, epochs: int):
    """Возвращает scheduler по строковому имени (или None)."""
    if name is None:
        return None
    name = name.lower()
    if name == "cosine":
        T_max = params.get("T_max", epochs)
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max)
    if name == "step":
        step_size = params.get("step_size", 10)
        gamma = params.get("gamma", 0.5)
        return torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=step_size, gamma=gamma
        )
    if name == "exponential":
        gamma = params.get("gamma", 0.95)
        return torch.optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=gamma
        )
    raise ValueError(f"Неизвестный scheduler '{name}'. Доступны: cosine, step, None")


def _get_loss_fn(name: str):
    """Возвращает loss-функцию по имени."""
    name = name.lower()
    if name == "bce":
        return nn.BCEWithLogitsLoss()
    raise ValueError(f"Неизвестная loss '{name}'. Пока доступна: bce")


class TitanicDNNClassifier(ClassifierMixin, BaseEstimator):
    """
    Sklearn-совместимый wrapper над TitanicMLP.

    Работает с cv_scores() и cross_val_score() без доп. изменений.
    Принимает как numpy-массивы, так и pandas DataFrame/Series.

    Параметры совпадают с DEFAULT_DNN_PARAMS из config.py и могут
    быть заменены через params= в build_dnn_model().
    """

    def __init__(
        self,
        input_dim: int = 1,
        hidden_dims=None,
        activation: str = "relu",
        dropout: float = 0.0,
        batchnorm: bool = False,
        optimizer: str = "adam",
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        batch_size: int = 32,
        epochs: int = 50,
        scheduler=None,
        scheduler_params=None,
        loss_fn: str = "bce",
        random_state: int = 42,
        verbose: bool = False,
        val_size: float = 0.2,
    ):
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.activation = activation
        self.dropout = dropout
        self.batchnorm = batchnorm
        self.optimizer = optimizer
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.epochs = epochs
        self.scheduler = scheduler
        self.scheduler_params = scheduler_params
        self.loss_fn = loss_fn
        self.random_state = random_state
        self.verbose = verbose
        self.val_size = val_size
        self.history_ = None

    # ── вспомогательные ─────────────────────────────────────────────────────

    def _to_numpy(self, X, y=None):
        if hasattr(X, "values"):
            X = X.values
        if y is not None and hasattr(y, "values"):
            y = y.values
        return (X, y) if y is not None else X

    def _build_net(self, input_dim: int):
        torch.manual_seed(self.random_state)
        self.model_ = TitanicMLP(
            input_dim=input_dim,
            hidden_dims=self.hidden_dims,
            activation=self.activation,
            dropout=self.dropout,
            batchnorm=self.batchnorm,
        )

    def _prepare_tensors(self, X, y):
        X_np, y_np = self._to_numpy(X, y)
        X_t = torch.tensor(X_np, dtype=torch.float32)
        y_t = torch.tensor(y_np, dtype=torch.float32)
        return X_t, y_t

    def _train_epoch(self, loader, model, opt, loss_fn):
        model.train()
        total_loss = 0.0
        for X_batch, y_batch in loader:
            opt.zero_grad()
            logits = model(X_batch).squeeze(1)
            loss = loss_fn(logits, y_batch)
            loss.backward()
            opt.step()
            total_loss += loss.item()
        return total_loss / len(loader)

    def _evaluate(self, X_t, y_t, model, loss_fn):
        model.eval()
        with torch.no_grad():
            logits = model(X_t).squeeze(1)
            loss = loss_fn(logits, y_t).item()
            probs = torch.sigmoid(logits).cpu().numpy()
        preds = (probs >= 0.5).astype(int)
        y_np = y_t.cpu().numpy()
        acc = (preds == y_np).mean()
        return loss, acc


    # ── sklearn API ──────────────────────────────────────────────────────────

    def fit(self, X_train, y_train, X_val=None, y_val=None):

        X_tr_t, y_tr_t = self._prepare_tensors(X_train, y_train)
        if X_val is not None and y_val is not None:
            X_val_t, y_val_t = self._prepare_tensors(X_val, y_val)
        else:
            X_val_t = y_val_t = None


        self._build_net(X_tr_t.shape[1])

        opt = _get_optimizer(self.optimizer, self.model_.parameters(),
                             self.lr, self.weight_decay)
        params = self.scheduler_params or {}
        sched = _get_scheduler(self.scheduler, opt, params, self.epochs)
        loss_fn = _get_loss_fn(self.loss_fn)

        train_ds = TensorDataset(X_tr_t, y_tr_t)
        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)

        self.history_ = {
            "train_loss": [],
            "train_acc": [],
        }
        if X_val_t is not None:
            self.history_["val_loss"] = []
            self.history_["val_acc"] = []

        for epoch in range(self.epochs):
            _ = self._train_epoch(train_loader, self.model_, opt, loss_fn)

            tr_loss, tr_acc = self._evaluate(X_tr_t, y_tr_t, self.model_, loss_fn)
            self.history_["train_loss"].append(tr_loss)
            self.history_["train_acc"].append(tr_acc)

            if X_val_t is not None:
                val_loss, val_acc = self._evaluate(X_val_t, y_val_t, self.model_, loss_fn)
                self.history_["val_loss"].append(val_loss)
                self.history_["val_acc"].append(val_acc)

            if sched is not None:
                sched.step()

            if self.verbose:
                msg = f"epoch {epoch+1}/{self.epochs} | train_loss={tr_loss:.4f} acc={tr_acc:.4f}"
                if X_val_t is not None:
                    msg += f" | val_loss={val_loss:.4f} acc={val_acc:.4f}"
                print(msg)

        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        X_np = self._to_numpy(X)
        X_t = torch.tensor(X_np, dtype=torch.float32)
        self.model_.eval()
        with torch.no_grad():
            logits = self.model_(X_t).squeeze(1)
            probs = torch.sigmoid(logits).numpy()
        return np.column_stack([1 - probs, probs])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    def score(self, X, y):
        y_np = self._to_numpy(y) if hasattr(y, "values") else np.asarray(y)
        return float((self.predict(X) == y_np).mean())