from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    database_url: str = "postgresql://mlb:mlb@localhost:5432/mlb"
    redis_url: str = "redis://localhost:6379"
    weather_api_key: str = ""
    odds_api_key: str = ""
    log_level: str = "INFO"

    # MLB API
    mlb_api_base_url: str = "https://statsapi.mlb.com/api/v1"
    mlb_api_rate_limit: int = 1000  # requests per hour

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}


settings = Settings()
