import pytest
import torch
from torch import nn

from model_lab.character_tokenizer import CharacterTokenizer
from model_lab.decoder_model import DecoderOnlyLanguageModel
from model_lab.decoder_stack import DecoderStack
from model_lab.language_model_head import LanguageModelHead
from model_lab.position_embedding import PositionEmbedding
from model_lab.residual_normalization import LayerNormalization
from model_lab.text_dataset import TextSequenceDataset
from model_lab.token_embedding import TokenEmbedding


def make_model() -> DecoderOnlyLanguageModel:
    return DecoderOnlyLanguageModel(
        vocab_size=7,
        max_sequence_length=6,
        embedding_dim=4,
        num_heads=2,
        num_layers=2,
    )


def test_model_contains_each_decoder_pipeline_component() -> None:
    model = make_model()

    assert isinstance(model.token_embedding, TokenEmbedding)
    assert isinstance(model.position_embedding, PositionEmbedding)
    assert isinstance(model.decoder_stack, DecoderStack)
    assert isinstance(model.final_normalization, LayerNormalization)
    assert isinstance(model.language_model_head, LanguageModelHead)
    assert isinstance(model, nn.Module)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"vocab_size": 0}, "vocab_size"),
        ({"max_sequence_length": 0}, "max_sequence_length"),
        ({"embedding_dim": 0}, "embedding_dim"),
        ({"num_heads": 0}, "num_heads"),
        ({"num_layers": 0}, "num_layers"),
        ({"embedding_dim": 5, "num_heads": 2}, "divisible"),
    ],
)
def test_constructor_rejects_invalid_configuration(
    kwargs: dict[str, int], message: str
) -> None:
    configuration = {
        "vocab_size": 7,
        "max_sequence_length": 6,
        "embedding_dim": 4,
        "num_heads": 2,
        "num_layers": 2,
    }
    configuration.update(kwargs)

    with pytest.raises(ValueError, match=message):
        DecoderOnlyLanguageModel(**configuration)


def test_forward_returns_one_vocab_logit_vector_per_input_position() -> None:
    model = make_model()

    logits = model(torch.tensor([[0, 1, 2], [3, 4, 5]], dtype=torch.int64))

    assert logits.shape == torch.Size([2, 3, 7])


def test_forward_matches_explicit_decoder_pipeline() -> None:
    model = make_model()
    token_ids = torch.tensor([[0, 1, 2], [3, 4, 5]], dtype=torch.int64)

    hidden_states = model.token_embedding(token_ids)
    hidden_states = hidden_states + model.position_embedding(sequence_length=3)
    hidden_states = model.decoder_stack(hidden_states)
    hidden_states = model.final_normalization(hidden_states)
    expected = model.language_model_head(hidden_states)

    assert torch.equal(model(token_ids), expected)


def test_future_token_does_not_change_earlier_logits() -> None:
    model = make_model()
    token_ids1 = torch.tensor([[0, 1, 2, 3]], dtype=torch.int64)
    token_ids2 = torch.tensor([[0, 1, 2, 6]], dtype=torch.int64)

    logits1 = model(token_ids1)
    logits2 = model(token_ids2)

    assert torch.allclose(logits1[:, :3], logits2[:, :3])


def test_forward_preserves_model_dtype_and_device() -> None:
    model = make_model().to(dtype=torch.float64)
    token_ids = torch.tensor([[0, 1, 2]], dtype=torch.int64)

    logits = model(token_ids)

    assert logits.dtype == torch.float64
    assert logits.device == token_ids.device


def test_backward_reaches_every_model_parameter() -> None:
    model = make_model()
    token_ids = torch.tensor([[0, 1, 2], [3, 4, 5]], dtype=torch.int64)

    model(token_ids).square().sum().backward()

    for parameter in model.parameters():
        assert parameter.grad is not None


def test_forward_rejects_non_batched_token_ids() -> None:
    model = make_model()

    with pytest.raises(ValueError, match=r"\[B, T\]"):
        model(torch.tensor([0, 1, 2], dtype=torch.int64))


def test_forward_rejects_non_int64_token_ids() -> None:
    model = make_model()

    with pytest.raises(TypeError, match="int64"):
        model(torch.tensor([[0.0, 1.0, 2.0]]))


def test_forward_rejects_empty_sequence() -> None:
    model = make_model()

    with pytest.raises(ValueError, match="sequence length"):
        model(torch.empty((2, 0), dtype=torch.int64))


def test_forward_rejects_sequence_beyond_model_context() -> None:
    model = make_model()

    with pytest.raises(ValueError, match="max_sequence_length"):
        model(torch.zeros((2, 7), dtype=torch.int64))


def test_tokenizer_dataset_and_model_form_complete_forward_pipeline() -> None:
    tokenizer = CharacterTokenizer.from_text("hello world")
    token_ids = tokenizer.encode("hello world")
    dataset = TextSequenceDataset(token_ids, context_length=4)
    input_ids, _ = dataset[0]
    model = DecoderOnlyLanguageModel(
        vocab_size=tokenizer.vocab_size,
        max_sequence_length=4,
        embedding_dim=4,
        num_heads=2,
        num_layers=2,
    )

    logits = model(input_ids.unsqueeze(0))

    assert logits.shape == torch.Size([1, 4, tokenizer.vocab_size])
