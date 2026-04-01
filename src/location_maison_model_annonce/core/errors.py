from __future__ import annotations


class ProjectError(Exception):
    """Base exception for the training project."""


class ConfigurationError(ProjectError):
    """Raised when configuration is invalid."""


class DatasetError(ProjectError):
    """Raised when dataset generation or loading fails."""


class TrainingError(ProjectError):
    """Raised when training fails."""
