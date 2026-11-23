import copy
import itertools
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn


class ExplicitLSTMCell(nn.Module):
    """Implementação explícita de uma célula LSTM com gates e Cell State."""

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Projeções lineares para [x_t, h_{t-1}] -> 4 * hidden_size
        # Ordem dos 4 blocos: forget (f), input (i), cell candidate (g/c_tilde), output (o)
        self.x2h = nn.Linear(input_size, 4 * hidden_size)
        self.h2h = nn.Linear(hidden_size, 4 * hidden_size)

    def forward(
        self,
        x_t: torch.Tensor,
        h_prev: torch.Tensor,
        c_prev: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        gates = self.x2h(x_t) + self.h2h(h_prev)
        f_gate, i_gate, c_tilde, o_gate = gates.chunk(4, dim=-1)

        f_t = torch.sigmoid(f_gate)      # Forget gate: o que descartar de c_prev
        i_t = torch.sigmoid(i_gate)      # Input gate: o quanto inserir de c_tilde
        g_t = torch.tanh(c_tilde)        # Candidata ao novo estado da célula
        o_t = torch.sigmoid(o_gate)      # Output gate: o que expor em h_t

        c_t = f_t * c_prev + i_t * g_t   # Novo Cell State
        h_t = o_t * torch.tanh(c_t)      # Novo estado oculto

        return h_t, c_t


class SequenceLSTMClassifier(nn.Module):
    """Classificador de sequências usando células LSTM explícitas."""

    def __init__(self, input_size: int, hidden_size: int, num_classes: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.cell = ExplicitLSTMCell(input_size, hidden_size)
        self.fc_out = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x tem formato (batch_size, seq_len, input_size)
        batch_size, seq_len, _ = x.shape
        h_t = torch.zeros(batch_size, self.hidden_size, device=x.device, dtype=x.dtype)
        c_t = torch.zeros(batch_size, self.hidden_size, device=x.device, dtype=x.dtype)

        for t in range(seq_len):
            h_t, c_t = self.cell(x[:, t, :], h_t, c_t)

        # Usa o último estado oculto h_t para mapear às classes
        logits = self.fc_out(h_t)
        return logits


def evaluate_loss(
    model: nn.Module,
    x_val: torch.Tensor,
    y_val: torch.Tensor,
    criterion: nn.Module,
) -> float:
    model.eval()
    with torch.no_grad():
        logits = model(x_val)
        loss = criterion(logits, y_val)
    return float(loss.item())


def grid_search_lstm(
    param_grid: Dict[str, List[Any]],
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_val: torch.Tensor,
    y_val: torch.Tensor,
    input_size: int,
    num_classes: int,
    epochs: int = 15,
    seed: int = 42,
) -> Tuple[Dict[str, Any], nn.Module, List[Tuple[Dict[str, Any], float]]]:
    """Explora sistematicamente todas as combinações da grade de hiperparâmetros."""
    keys = list(param_grid.keys())
    value_lists = [param_grid[k] for k in keys]
    criterion = nn.CrossEntropyLoss()

    results: List[Tuple[Dict[str, Any], float]] = []
    best_params: Dict[str, Any] = {}
    best_val_loss = float("inf")
    best_model: nn.Module | None = None

    for idx, combo in enumerate(itertools.product(*value_lists), start=1):
        params = dict(zip(keys, combo))
        torch.manual_seed(seed)

        model = SequenceLSTMClassifier(
            input_size=input_size,
            hidden_size=params["hidden_size"],
            num_classes=num_classes,
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=params["lr"])

        for _ in range(epochs):
            model.train()
            optimizer.zero_grad()
            logits = model(x_train)
            loss = criterion(logits, y_train)
            loss.backward()
            optimizer.step()

        val_loss = evaluate_loss(model, x_val, y_val, criterion)
        results.append((params, val_loss))
        print(f"[{idx:02d}] params={params} -> val_loss={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_params = params
            best_model = copy.deepcopy(model)

    assert best_model is not None
    return best_params, best_model, results


def main() -> None:
    torch.manual_seed(42)

    seq_len = 12
    input_size = 6
    num_classes = 3

    # Dados sintéticos de treino e validação
    x_train = torch.randn(32, seq_len, input_size)
    y_train = torch.randint(0, num_classes, (32,))
    x_val = torch.randn(16, seq_len, input_size)
    y_val = torch.randint(0, num_classes, (16,))

    # Grade de hiperparâmetros para busca exaustiva
    param_grid = {
        "hidden_size": [16, 32],
        "lr": [1e-2, 5e-3, 1e-3],
        "epochs": [15],
    }

    # Mantém apenas chaves de modelo/otimizador na grade de busca
    search_grid = {
        "hidden_size": param_grid["hidden_size"],
        "lr": param_grid["lr"],
    }
    epochs = param_grid["epochs"][0]

    print("Iniciando Grid Search...\n")
    best_params, best_model, all_results = grid_search_lstm(
        param_grid=search_grid,
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        input_size=input_size,
        num_classes=num_classes,
        epochs=epochs,
        seed=42,
    )

    best_loss = min(loss for _, loss in all_results)
    print("\n--- Resultado do Grid Search ---")
    print(f"Total de combinações avaliadas: {len(all_results)}")
    print(f"Melhores hiperparâmetros: {best_params}")
    print(f"Menor val_loss: {best_loss:.4f}")

    # Inferência com o melhor modelo encontrado
    best_model.eval()
    sample = torch.randn(1, seq_len, input_size)
    with torch.no_grad():
        probs = torch.softmax(best_model(sample), dim=-1)
    print(f"Probabilidades previstas na amostra: {[round(p, 4) for p in probs[0].tolist()]}")


if __name__ == "__main__":
    main()