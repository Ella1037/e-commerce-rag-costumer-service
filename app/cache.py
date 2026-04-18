# app/cache.py
"""
In-memory query cache with TTL and max size.
Key: (question_normalized, method)
Value: cached answer string
"""
import time
import hashlib
from collections import OrderedDict

class QueryCache:
    def __init__(self, max_size: int = 500, ttl_seconds: int = 3600):
        self.max_size    = max_size
        self.ttl         = ttl_seconds
        self._cache: OrderedDict[str, tuple[str, float]] = OrderedDict()  # key → (answer, timestamp)
        self.hits        = 0
        self.misses      = 0

    def _make_key(self, question: str, method: str) -> str:
        normalized = question.strip().lower()
        raw        = f"{method}::{normalized}"
        return hashlib.md5(raw.encode()).hexdigest()

    def get(self, question: str, method: str) -> str | None:
        key = self._make_key(question, method)
        if key not in self._cache:
            self.misses += 1
            return None
        answer, ts = self._cache[key]
        if time.time() - ts > self.ttl:
            del self._cache[key]
            self.misses += 1
            return None
        # LRU: move to end
        self._cache.move_to_end(key)
        self.hits += 1
        return answer

    def set(self, question: str, method: str, answer: str) -> None:
        key = self._make_key(question, method)
        if key in self._cache:
            self._cache.move_to_end(key)
        self._cache[key] = (answer, time.time())
        # evict oldest if over max size
        if len(self._cache) > self.max_size:
            self._cache.popitem(last=False)

    @property
    def stats(self) -> dict:
        total = self.hits + self.misses
        return {
            "hits":      self.hits,
            "misses":    self.misses,
            "hit_rate":  round(self.hits / total, 3) if total > 0 else 0.0,
            "size":      len(self._cache),
            "max_size":  self.max_size,
        }

# module-level singleton
query_cache = QueryCache(max_size=500, ttl_seconds=3600)