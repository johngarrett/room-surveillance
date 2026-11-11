"""Monotonic deadlines, checked only on fresh detection frames."""
import random
from config import Config


class Scheduler:
    def __init__(self):
        self.deadline = None
        self.settings = None

    def update(self, now: float, occupied: bool, config: Config):
        settings = (config.CAPTURE_ENABLED, config.MIN_CAPTURE_INTERVAL, config.MAX_CAPTURE_INTERVAL)
        if not occupied or not config.CAPTURE_ENABLED:
            self.deadline = None
        elif self.deadline is None or settings != self.settings:
            self.schedule(now, config)
        self.settings = settings

    def schedule(self, now: float, config: Config):
        self.deadline = now + random.uniform(config.MIN_CAPTURE_INTERVAL, config.MAX_CAPTURE_INTERVAL)

    def due(self, now: float) -> bool:
        return self.deadline is not None and now >= self.deadline
