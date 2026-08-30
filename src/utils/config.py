"""Configuration loading and validation.

This module provides utilities for loading experiment configurations from YAML files
and validating them against expected schemas.
"""

import yaml
from pathlib import Path
from typing import Dict, Any
from dataclasses import dataclass, field, asdict

from src.utils.task_specs import (
    DEFAULT_ALGORITHMS,
    SELECT_TASK_CHOICES,
    TERMINATION_LATENT_CHOICES,
    normalize_algorithm_order,
)


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    embed_dim: int = 32
    residual_connections: bool = True
    agg_fn: str = "MAX"  # SUM, AVG, MIN, MAX
    num_mp_layers: int = 2
    dropout: float = 0.1
    algorithms: list[str] = field(default_factory=lambda: list(DEFAULT_ALGORITHMS))
    termination_mode: str = "head"  # head, distance
    termination_distance_latent: str = "processed"  # processed, encoded, encoded_bfs, encoded_bf, encoded_prim
    termination_distance: str = "mean_l2"  # l2, mean_l2, l1, mse
    termination_distance_threshold: float = 0.01
    termination_distance_signal: bool = True
    processor_input_adapter: bool = False
    decoder_input_mode: str = "processed_encoded"
    predecessor_input_mode: str = "processed_encoded_edge"


@dataclass
class TrainingConfig:
    """Training hyperparameters."""
    epochs: int = 500
    learning_rate: float = 1e-5
    max_grad_norm: float = 1.0
    batch_size: int = 10
    eval_interval: int = 10
    seed: int = 42
    tasks: str = "all"
    init_checkpoint: str | None = None
    processor_init_checkpoint: str | None = None
    init_checkpoint_modules: list[str] = field(default_factory=list)
    freeze_modules: list[str] = field(default_factory=list)
    reset_modules: list[str] = field(default_factory=list)
    legibility_residual_direction_weight: float = 0.0
    legibility_residual_direction_stream: str = "encoded"


@dataclass
class DataConfig:
    """Dataset configuration."""
    train_path: str = "data/train_dataset.npz"
    val_path: str = "data/val_dataset.npz"
    test_path: str = "data/test_dataset.npz"
    task_paths: Dict[str, Dict[str, str]] | None = None


@dataclass
class LoggingConfig:
    """Logging and tracking configuration."""
    use_wandb: bool = False
    wandb_project: str = "nge"
    wandb_entity: str = ""
    log_interval: int = 1
    save_checkpoints: bool = True
    checkpoint_interval: int = 50
    checkpoint_keep_last: int = 5


@dataclass
class SurrogateConfig:
    """Cross-algorithm surrogate processor configuration."""
    enabled: bool = False
    source_algorithm: str = "dijkstra"
    target_algorithm: str = "bfs"
    source_update_control: str = "normal"
    target_update_mean_control: str = "normal"
    source_run_dir: str | None = None
    target_init_run_dir: str | None = None
    source_trajectories_dir: str | None = None
    target_trajectories_dir: str | None = None
    source_dictionary_path: str | None = None
    target_dictionary_path: str | None = None
    source_group_path: str | None = None
    source_dictionary_key: str = "signed_atoms"
    basis_components: int = 6
    context_dim: int = 8
    context_mp_layers: int = 1
    adapt_hidden_dim: int = 0
    adapt_input_mode: str = "context_source"
    route_activation: str = "sparsemax"
    route_top_k: int = 4
    route_entropy_bonus_weight: float = 0.0
    residual_hidden_dim: int = 0
    residual_penalty_weight: float = 0.0


@dataclass
class ExperimentConfig:
    """Complete experiment configuration."""
    name: str = "default"
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    surrogate: SurrogateConfig = field(default_factory=SurrogateConfig)

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            'name': self.name,
            'model': asdict(self.model),
            'training': asdict(self.training),
            'data': asdict(self.data),
            'logging': asdict(self.logging),
            'surrogate': asdict(self.surrogate),
        }

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ExperimentConfig':
        """Create config from dictionary."""
        return cls(
            name=config_dict.get('name', 'default'),
            model=ModelConfig(**config_dict.get('model', {})),
            training=TrainingConfig(**config_dict.get('training', {})),
            data=DataConfig(**config_dict.get('data', {})),
            logging=LoggingConfig(**config_dict.get('logging', {})),
            surrogate=SurrogateConfig(**config_dict.get('surrogate', {})),
        )


def load_config(config_path: str | Path) -> ExperimentConfig:
    """Load experiment configuration from YAML file.

    Args:
        config_path: Path to YAML configuration file

    Returns:
        ExperimentConfig object with validated configuration

    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If config file is malformed
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)

    if config_dict is None:
        config_dict = {}

    return ExperimentConfig.from_dict(config_dict)


def save_config(config: ExperimentConfig, output_path: str | Path) -> None:
    """Save experiment configuration to YAML file.

    Args:
        config: ExperimentConfig object to save
        output_path: Path where to save the YAML file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        yaml.dump(config.to_dict(), f, default_flow_style=False, sort_keys=False)


def validate_config(config: ExperimentConfig) -> None:
    """Validate experiment configuration.

    Args:
        config: ExperimentConfig to validate

    Raises:
        ValueError: If configuration is invalid
    """
    # Validate model config
    if config.model.embed_dim <= 0:
        raise ValueError(f"embed_dim must be positive, got {config.model.embed_dim}")

    if config.model.agg_fn not in ["SUM", "AVG", "MIN", "MAX"]:
        raise ValueError(f"Invalid agg_fn: {config.model.agg_fn}")

    if config.model.num_mp_layers < 0:
        raise ValueError(
            "num_mp_layers must be non-negative, "
            f"got {config.model.num_mp_layers}"
        )

    if not 0 <= config.model.dropout < 1:
        raise ValueError(f"dropout must be in [0, 1), got {config.model.dropout}")

    normalize_algorithm_order(config.model.algorithms)

    if config.model.termination_mode not in ["head", "distance"]:
        raise ValueError(f"Invalid termination_mode: {config.model.termination_mode}")

    if config.model.termination_distance_latent not in TERMINATION_LATENT_CHOICES:
        raise ValueError(
            "termination_distance_latent must be one of: "
            f"{', '.join(TERMINATION_LATENT_CHOICES)}, "
            f"got {config.model.termination_distance_latent}"
        )

    if config.model.termination_distance not in ["l2", "mean_l2", "l1", "mse"]:
        raise ValueError(
            "termination_distance must be one of: l2, mean_l2, l1, mse. "
            f"Got {config.model.termination_distance}"
        )

    if config.model.termination_distance_threshold < 0:
        raise ValueError(
            "termination_distance_threshold must be non-negative, "
            f"got {config.model.termination_distance_threshold}"
        )

    if not isinstance(config.model.termination_distance_signal, bool):
        raise ValueError(
            "termination_distance_signal must be boolean, "
            f"got {type(config.model.termination_distance_signal).__name__}"
        )

    if config.model.decoder_input_mode not in ["processed_encoded", "processed_only"]:
        raise ValueError(
            "decoder_input_mode must be one of: processed_encoded, processed_only. "
            f"Got {config.model.decoder_input_mode}"
        )

    if config.model.predecessor_input_mode not in [
        "processed_encoded_edge",
        "processed_edge",
        "processed_only",
    ]:
        raise ValueError(
            "predecessor_input_mode must be one of: "
            "processed_encoded_edge, processed_edge, processed_only. "
            f"Got {config.model.predecessor_input_mode}"
        )

    # Validate training config
    if config.training.epochs <= 0:
        raise ValueError(f"epochs must be positive, got {config.training.epochs}")

    if config.training.learning_rate <= 0:
        raise ValueError(f"learning_rate must be positive, got {config.training.learning_rate}")

    if config.training.batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {config.training.batch_size}")

    if config.training.tasks not in SELECT_TASK_CHOICES:
        raise ValueError(
            "training.tasks must be one of: "
            f"{', '.join(SELECT_TASK_CHOICES)}. Got {config.training.tasks}"
        )
    if config.training.tasks != "all" and config.training.tasks not in config.model.algorithms:
        raise ValueError(
            "training.tasks must be 'all' or one of model.algorithms. "
            f"Got {config.training.tasks} for algorithms {config.model.algorithms}"
        )

    if config.training.init_checkpoint is not None and not isinstance(
        config.training.init_checkpoint, str
    ):
        raise ValueError("training.init_checkpoint must be a string path or null.")
    if config.training.processor_init_checkpoint is not None and not isinstance(
        config.training.processor_init_checkpoint, str
    ):
        raise ValueError(
            "training.processor_init_checkpoint must be a string path or null."
        )

    for field_name in ("init_checkpoint_modules", "freeze_modules", "reset_modules"):
        field_value = getattr(config.training, field_name)
        if not isinstance(field_value, list) or not all(
            isinstance(entry, str) and entry.strip() for entry in field_value
        ):
            raise ValueError(
                f"training.{field_name} must be a list of non-empty strings."
            )

    if config.training.init_checkpoint_modules and config.training.init_checkpoint is None:
        raise ValueError(
            "training.init_checkpoint_modules requires training.init_checkpoint."
        )

    if config.training.legibility_residual_direction_weight < 0:
        raise ValueError(
            "training.legibility_residual_direction_weight must be non-negative."
        )
    if config.training.legibility_residual_direction_stream not in (
        "encoded",
        "previous_hidden",
    ):
        raise ValueError(
            "training.legibility_residual_direction_stream must be one of: "
            "encoded, previous_hidden."
        )

    overlap = sorted(set(config.training.freeze_modules) & set(config.training.reset_modules))
    if overlap:
        raise ValueError(
            "training.freeze_modules and training.reset_modules overlap: "
            + ", ".join(overlap)
        )

    if config.logging.log_interval <= 0:
        raise ValueError(f"log_interval must be positive, got {config.logging.log_interval}")

    if config.logging.checkpoint_interval <= 0:
        raise ValueError(
            f"checkpoint_interval must be positive, got {config.logging.checkpoint_interval}"
        )

    if config.logging.checkpoint_keep_last <= 0:
        raise ValueError(
            "checkpoint_keep_last must be positive, "
            f"got {config.logging.checkpoint_keep_last}"
        )

    if config.surrogate.enabled:
        source_algorithm = normalize_algorithm_order([config.surrogate.source_algorithm])[0]
        target_algorithm = normalize_algorithm_order([config.surrogate.target_algorithm])[0]
        if config.surrogate.source_update_control not in {"normal", "zero", "shuffle"}:
            raise ValueError(
                "surrogate.source_update_control must be one of "
                f"normal, zero, shuffle. Got {config.surrogate.source_update_control}."
            )
        if config.surrogate.target_update_mean_control not in {"normal", "zero"}:
            raise ValueError(
                "surrogate.target_update_mean_control must be one of "
                "normal, zero. Got "
                f"{config.surrogate.target_update_mean_control}."
            )
        if source_algorithm == target_algorithm:
            raise ValueError("surrogate.source_algorithm and target_algorithm must differ.")
        if config.model.algorithms != [target_algorithm]:
            raise ValueError(
                "Surrogate training currently expects model.algorithms to contain only "
                f"the target algorithm {target_algorithm}. Got {config.model.algorithms}."
            )
        if config.training.tasks != target_algorithm:
            raise ValueError(
                "Surrogate training currently expects training.tasks to equal "
                f"the target algorithm {target_algorithm}."
            )
        for field_name in (
            "source_trajectories_dir",
            "target_trajectories_dir",
        ):
            value = getattr(config.surrogate, field_name)
            if not value:
                raise ValueError(f"surrogate.{field_name} is required when enabled.")
            if not Path(value).exists():
                raise ValueError(f"surrogate.{field_name} does not exist: {value}")
        if config.surrogate.source_dictionary_key not in {"atoms", "signed_atoms"}:
            raise ValueError(
                "surrogate.source_dictionary_key must be one of atoms, signed_atoms. "
                f"Got {config.surrogate.source_dictionary_key}."
            )
        for field_name in ("source_dictionary_path", "target_dictionary_path"):
            value = getattr(config.surrogate, field_name)
            if value is not None and not Path(value).exists():
                raise ValueError(f"surrogate.{field_name} does not exist: {value}")
        if config.surrogate.target_init_run_dir is not None and not Path(
            config.surrogate.target_init_run_dir
        ).exists():
            raise ValueError(
                "surrogate.target_init_run_dir does not exist: "
                f"{config.surrogate.target_init_run_dir}"
            )
        if config.surrogate.basis_components <= 0:
            raise ValueError("surrogate.basis_components must be positive.")
        if config.surrogate.context_dim <= 0:
            raise ValueError("surrogate.context_dim must be positive.")
        if config.surrogate.context_mp_layers < 0:
            raise ValueError("surrogate.context_mp_layers must be non-negative.")
        if config.surrogate.adapt_hidden_dim < 0:
            raise ValueError("surrogate.adapt_hidden_dim must be non-negative.")
        if config.surrogate.adapt_input_mode not in {
            "context_source",
            "source_only",
            "coordinate_native",
            "group_local_chart",
        }:
            raise ValueError(
                "surrogate.adapt_input_mode must be one of context_source, source_only, "
                "coordinate_native, group_local_chart. "
                f"Got {config.surrogate.adapt_input_mode}."
            )
        if config.surrogate.adapt_input_mode == "group_local_chart":
            if config.surrogate.source_group_path is None:
                raise ValueError(
                    "surrogate.source_group_path is required for group_local_chart."
                )
            if not Path(config.surrogate.source_group_path).exists():
                raise ValueError(
                    "surrogate.source_group_path does not exist: "
                    f"{config.surrogate.source_group_path}"
                )
        if config.surrogate.route_activation not in {
            "sparsemax",
            "softmax",
            "topk_softmax",
        }:
            raise ValueError(
                "surrogate.route_activation must be one of sparsemax, softmax, "
                f"topk_softmax. Got {config.surrogate.route_activation}."
            )
        if config.surrogate.route_top_k <= 0:
            raise ValueError("surrogate.route_top_k must be positive.")
        if config.surrogate.route_entropy_bonus_weight < 0:
            raise ValueError(
                "surrogate.route_entropy_bonus_weight must be non-negative."
            )
        if config.surrogate.residual_hidden_dim < 0:
            raise ValueError("surrogate.residual_hidden_dim must be non-negative.")
        if config.surrogate.residual_penalty_weight < 0:
            raise ValueError("surrogate.residual_penalty_weight must be non-negative.")

    # Validate data paths exist
    for path_name, path in [
        ('train_path', config.data.train_path),
        ('val_path', config.data.val_path),
        ('test_path', config.data.test_path)
    ]:
        if config.data.task_paths is None and not Path(path).exists():
            raise ValueError(f"Data file not found: {path_name}={path}")

    if config.data.task_paths is not None:
        if not isinstance(config.data.task_paths, dict) or not config.data.task_paths:
            raise ValueError("data.task_paths must be a non-empty mapping when provided.")
        for algorithm in config.model.algorithms:
            if algorithm not in config.data.task_paths:
                raise ValueError(
                    f"Missing data.task_paths entry for configured algorithm: {algorithm}"
                )
            split_paths = config.data.task_paths[algorithm]
            if not isinstance(split_paths, dict):
                raise ValueError(
                    f"data.task_paths.{algorithm} must be a mapping of split names to paths."
                )
            for split_key in ("train_path", "val_path", "test_path"):
                if split_key not in split_paths:
                    raise ValueError(
                        f"Missing {split_key} in data.task_paths.{algorithm}"
                    )
                if not Path(split_paths[split_key]).exists():
                    raise ValueError(
                        f"Data file not found: data.task_paths.{algorithm}.{split_key}="
                        f"{split_paths[split_key]}"
                    )
