import redis
from app.core.config import settings

class RedisClient:
    _client = None

    @classmethod
    def get_client(cls):
        if cls._client is None:
            cls._client = redis.Redis(
                host=settings.redis_host,
                port=settings.redis_port,
                decode_responses=True,
            )
        # print("connect",cls._client,cls._client.ping())
        return cls._client
