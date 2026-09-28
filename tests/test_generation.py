import pytest
import torch

from model_lab.decoder_model import DecoderOnlyLanguageModel
from model_lab.generation import generate_greedy


def make_model() -> DecoderOnlyLanguageModel:
    return DecoderOnlyLanguageModel(
        vocab_size=5,
        max_sequence_length=3,
        embedding_dim=4,
        num_heads=2,
        num_layers=1,
    )


def test_zero_new_tokens_returns_input_without_model_call() -> None:
    model = make_model()
    input_ids = torch.tensor([[0, 1], [2, 3]], dtype=torch.int64)

    output = generate_greedy(model, input_ids, max_new_tokens=0)

    assert torch.equal(output, input_ids)
    assert output is not input_ids


def test_generation_appends_requested_number_of_tokens() -> None:
    torch.manual_seed(0)
    model = make_model()
    input_ids = torch.tensor([[0, 1], [2, 3]], dtype=torch.int64)

    output = generate_greedy(model, input_ids, max_new_tokens=4)

    assert output.shape == torch.Size([2, 6])
    assert torch.equal(output[:, :2], input_ids)
    assert output.dtype == torch.int64
    assert torch.all((output >= 0) & (output < model.vocab_size))


def test_greedy_choice_matches_last_position_argmax_each_step() -> None:
    class FixedModel(DecoderOnlyLanguageModel):
        def __init__(self) -> None:
            super().__init__(
                vocab_size=3,
                max_sequence_length=3,
                embedding_dim=2,
                num_heads=1,
                num_layers=1,
            )
            self.calls: list[torch.Tensor] = []

        def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
            self.calls.append(token_ids.clone())
            logits = torch.zeros(
                token_ids.shape[0], token_ids.shape[1], self.vocab_size
            )
            logits[:, -1, 2] = 5.0
            return logits

    model = FixedModel()
    input_ids = torch.tensor([[0, 1]], dtype=torch.int64)

    output = generate_greedy(model, input_ids, max_new_tokens=2)

    assert torch.equal(output, torch.tensor([[0, 1, 2, 2]], dtype=torch.int64))
    assert [call.tolist() for call in model.calls] == [[[0, 1]], [[0, 1, 2]]]


def test_generation_crops_context_to_model_limit() -> None:
    class RecordingModel(DecoderOnlyLanguageModel):
        def __init__(self) -> None:
            super().__init__(
                vocab_size=3,
                max_sequence_length=3,
                embedding_dim=2,
                num_heads=1,
                num_layers=1,
            )
            self.calls: list[torch.Tensor] = []

        def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
            self.calls.append(token_ids.clone())
            logits = torch.zeros(
                token_ids.shape[0], token_ids.shape[1], self.vocab_size
            )
            logits[:, -1, 1] = 1.0
            return logits

    model = RecordingModel()
    input_ids = torch.tensor([[0, 1, 2, 0]], dtype=torch.int64)

    output = generate_greedy(model, input_ids, max_new_tokens=2)

    assert torch.equal(output, torch.tensor([[0, 1, 2, 0, 1, 1]], dtype=torch.int64))
    assert [call.tolist() for call in model.calls] == [
        [[1, 2, 0]],
        [[2, 0, 1]],
    ]


def test_generation_does_not_create_parameter_gradients() -> None:
    model = make_model()
    input_ids = torch.tensor([[0, 1]], dtype=torch.int64)

    generate_greedy(model, input_ids, max_new_tokens=2)

    assert all(parameter.grad is None for parameter in model.parameters())


def test_generation_restores_original_training_mode() -> None:
    model = make_model()
    input_ids = torch.tensor([[0, 1]], dtype=torch.int64)
    model.train()

    generate_greedy(model, input_ids, max_new_tokens=1)

    assert model.training


@pytest.mark.parametrize(
    "bad_input",
    [
        torch.tensor([0, 1], dtype=torch.int64),
        torch.tensor([[0.0, 1.0]]),
        torch.empty((1, 0), dtype=torch.int64),
    ],
)
def test_generation_rejects_invalid_input_ids(bad_input: torch.Tensor) -> None:
    with pytest.raises((ValueError, TypeError), match="shape|int64|sequence length"):
        generate_greedy(make_model(), bad_input, max_new_tokens=1)


def test_generation_rejects_negative_token_count() -> None:
    with pytest.raises(ValueError, match="max_new_tokens"):
        generate_greedy(
            make_model(),
            torch.tensor([[0, 1]], dtype=torch.int64),
            max_new_tokens=-1,
        )
