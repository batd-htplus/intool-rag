"""Core modules for RAG system.

Includes:
- Dependency injection container for managing providers
- Configuration management
- Logging setup
"""

from .container import Container, get_container, shutdown_container

__all__ = ["Container", "get_container", "shutdown_container"]
