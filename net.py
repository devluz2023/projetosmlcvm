import torch
from torch import Tensor, nn


class CVMNet(nn.Module):
    def __init__(self, dim_in: int, dim_hidden: int, n_classes: int):
        super().__init__()
        self.i_layer = nn.Linear(dim_in, dim_hidden)
        self.h_layer = nn.Sequential(
            nn.Linear(dim_hidden, dim_hidden // 2),
            nn.Tanh(),
        )
        self.o_layer = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(dim_hidden // 2, n_classes),
            nn.Softmax(dim=-1),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.o_layer(self.h_layer(self.i_layer(x)))


def main() -> None:
    torch.manual_seed(42)

    model = CVMNet(dim_in=400, dim_hidden=100, n_classes=3)
    x_input = torch.eye(20).flatten()  # tensor 1D de tamanho 400 (20 * 20)

    pred = model(x_input).detach()

    print(f"Formato (shape) de pred : {tuple(pred.shape)}")
    print(f"Valores de pred        : {[round(float(v), 6) for v in pred]}")
    print(f"Soma dos valores       : {float(pred.sum()):.6f}")
    print(f"requires_grad          : {pred.requires_grad}")


if __name__ == "__main__":
    main()