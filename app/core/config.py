from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    app_name: str
    env: str

    redis_host: str
    redis_port: int

    log_level: str = "INFO"

    class Config:
        env_file = ".env"

settings = Settings()
