"""Process-wide adapters shared by the use cases."""

from ..core.database import engine
from ..infrastructure.sandbox import Sandbox

sandbox = Sandbox(engine)
