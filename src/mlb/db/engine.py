from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from mlb.config import settings

engine = create_engine(settings.database_url, pool_size=10, max_overflow=20)
SessionLocal = sessionmaker(bind=engine)


def get_session() -> Session:
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()
