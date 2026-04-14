"""Context engine package."""

from .api.server import create_app
from .core.engine import DefaultContextEngine
from .core.interfaces import ContextEngine
from .core.models import Agent, ContextDistillation, ExecutionPlan, Response, RetrievedContext, Task, Trace
from .evals.evaluator import Evaluator
from .factory import SmartFactory

__version__ = "0.3.0"

__all__ = [
    "Agent",
    "ContextDistillation",
    "ContextEngine",
    "create_app",
    "DefaultContextEngine",
    "Evaluator",
    "ExecutionPlan",
    "Response",
    "RetrievedContext",
    "SmartFactory",
    "Task",
    "Trace",
    "__version__",
]
