import yaml
import GPUtil

class DynamicChunkSizer:
    def __init__(self, config: dict):
        cfg = config.get("dynamic_chunk", {})

        self.enable = cfg.get("enable", True)
        self.sample_bytes = cfg.get("sample_bytes", 2048)
        self.min_chunk = cfg.get("min_chunk", 10000)
        self.max_chunk = cfg.get("max_chunk", 200000)
        self.reserve_mb = cfg.get("reserve_mb", 500)
        self.fallback_chunk = cfg.get("fallback_chunk", 50000)

    def estimate_chunksize(self):
        if not self.enable:
            return self.fallback_chunk

        try:
            gpus = GPUtil.getGPUs()
            if not gpus:
                return self.fallback_chunk
            mem_free = gpus[0].memoryFree  # 单位：MB
        except Exception:
            return self.fallback_chunk

        effective_mem = max(mem_free - self.reserve_mb, 100)
        bytes_available = effective_mem * 1024 * 1024
        chunk = int(bytes_available / self.sample_bytes)
        return max(self.min_chunk, min(self.max_chunk, chunk))
