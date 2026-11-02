import pytest

from scrnarefgenerator import ConditionSpec, EvaluationConfig, GeneratorConfig


def test_public_config_defaults_are_valid():
    config = GeneratorConfig(seed=7)
    spec = ConditionSpec(columns=["celltype"], joint={("T",): 2})

    assert config.seed == 7
    assert config.distribution_model == "hierarchical_nb"
    assert EvaluationConfig().auc_range == (0.55, 0.90)
    assert spec.joint == {("T",): 2}


def test_neighbor_geometry_is_the_default():
    config = GeneratorConfig()

    assert config.geometry_model == "neighbor_preserving"
    assert config.geometry_neighbors == 30
    assert config.local_profile_mix == 0.5
    assert config.group_prior_mix == 0.1
    assert config.count_noise_model == "poisson"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("local_profile_mix", 1.1),
        ("group_prior_mix", -0.1),
        ("geometry_neighbors", 0),
        ("geometry_latent_k", 1),
        ("geometry_hvg_count", 1),
    ],
)
def test_invalid_geometry_configuration_is_rejected(field, value):
    with pytest.raises(ValueError):
        GeneratorConfig(**{field: value})


def test_unknown_count_noise_model_is_rejected():
    with pytest.raises(ValueError):
        GeneratorConfig(count_noise_model="unknown")


def test_balanced_novelty_is_the_default():
    config = GeneratorConfig()

    assert config.novelty_mode == "balanced"
    assert config.novelty_strength is None
    assert config.effective_novelty_strength == pytest.approx(0.25)


@pytest.mark.parametrize(
    ("mode", "expected"),
    [("fidelity", 0.0), ("balanced", 0.25), ("novel", 0.50)],
)
def test_novelty_mode_presets(mode, expected):
    assert GeneratorConfig(novelty_mode=mode).effective_novelty_strength == pytest.approx(
        expected
    )


def test_explicit_novelty_strength_overrides_mode():
    config = GeneratorConfig(novelty_mode="fidelity", novelty_strength=0.25)
    assert config.effective_novelty_strength == pytest.approx(0.25)


def test_invalid_novelty_configuration_is_rejected():
    with pytest.raises(ValueError):
        GeneratorConfig(novelty_mode="unknown")

    with pytest.raises(ValueError):
        GeneratorConfig(novelty_strength=1.1)
