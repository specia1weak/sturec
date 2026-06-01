from __future__ import annotations

import math


class CurriculumScheduler:
    def __init__(
            self,
            total_steps: int,
            start_ratio: float = 0.8,
            end_ratio: float = 0.2,
            schedule_type: str = "cosine",
            start_temp: float = 5.0,
            end_temp: float = 1.0,
            warmup_steps: int = 0,
    ):
        self.total_steps = max(1, int(total_steps))
        self.start_ratio = float(start_ratio)
        self.end_ratio = float(end_ratio)
        self.schedule_type = str(schedule_type)
        self.start_temp = float(start_temp)
        self.end_temp = float(end_temp)
        self.warmup_steps = max(0, int(warmup_steps))

        self._step = 0

    def step(self) -> None:
        self._step += 1

    def reset(self) -> None:
        self._step = 0

    def get_progress(self) -> float:
        raw = max(0.0, min(1.0, (self._step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)))
        return raw

    def get_sharing_ratio(self) -> float:
        progress = self.get_progress()
        if self.schedule_type == "cosine":
            ratio = self.end_ratio + 0.5 * (self.start_ratio - self.end_ratio) * (1.0 + math.cos(progress * math.pi))
        elif self.schedule_type == "linear":
            ratio = self.start_ratio + (self.end_ratio - self.start_ratio) * progress
        elif self.schedule_type == "step":
            if progress < 0.33:
                ratio = self.start_ratio
            elif progress < 0.66:
                ratio = (self.start_ratio + self.end_ratio) * 0.5
            else:
                ratio = self.end_ratio
        else:
            ratio = self.start_ratio + (self.end_ratio - self.start_ratio) * progress
        return max(self.end_ratio, min(self.start_ratio, ratio))

    def get_affinity_temperature(self) -> float:
        progress = self.get_progress()
        return self.start_temp + (self.end_temp - self.start_temp) * progress

    def get_phase(self) -> str:
        progress = self.get_progress()
        if progress < 0.33:
            return "early"
        if progress < 0.66:
            return "mid"
        return "late"

    def state_dict(self) -> dict:
        return {
            "_step": self._step,
            "total_steps": self.total_steps,
            "start_ratio": self.start_ratio,
            "end_ratio": self.end_ratio,
            "schedule_type": self.schedule_type,
            "start_temp": self.start_temp,
            "end_temp": self.end_temp,
            "warmup_steps": self.warmup_steps,
        }

    def load_state_dict(self, state: dict) -> None:
        self._step = state.get("_step", 0)
        self.total_steps = state.get("total_steps", self.total_steps)
        self.start_ratio = state.get("start_ratio", self.start_ratio)
        self.end_ratio = state.get("end_ratio", self.end_ratio)
        self.schedule_type = state.get("schedule_type", self.schedule_type)
        self.start_temp = state.get("start_temp", self.start_temp)
        self.end_temp = state.get("end_temp", self.end_temp)
        self.warmup_steps = state.get("warmup_steps", self.warmup_steps)
