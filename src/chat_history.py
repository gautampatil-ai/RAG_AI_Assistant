from typing import List, Dict, Any

class ChatHistoryTracker:
    def __init__(self):
        self.messages: List[Dict[str, Any]] = []
        self.query_count: int = 0
        self.total_response_time: float = 0.0

    def add_user_message(self, content: str):
        self.messages.append({"role": "user", "content": content})

    def add_assistant_message(self, content: str, citations: List[Dict[str, Any]] = None):
        self.messages.append({"role": "assistant", "content": content, "citations": citations or []})

    def log_query_metrics(self, time_sec: float):
        self.query_count += 1
        self.total_response_time += time_sec

    def get_average_response_time((self) -> float:
        return round(self.total_response_time / self.query_count, 2) if self.query_count > 0 else 0.0

    def clear(self):
        self.messages.clear()
        self.query_count = 0
        self.total_response_time = 0.0
