import logging
from sqlmodel import SQLModel, create_engine, Session
import sqlite3

# Configure clean logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

DATABASE_URL = "sqlite:///chat.db"
engine = create_engine(DATABASE_URL, echo=False)

def check_connection():
    """Test basic SQLite database connection."""
    try:
        with sqlite3.connect("chat.db") as conn:
            conn.execute("SELECT 1").fetchone()
            logger.info("✅ Database connection successful")
            return True
    except Exception as e:
        logger.error(f"❌ Database connection failed: {e}")
        return False

def init_db():
    """Initialize database with basic tables."""
    try:
        # Create basic tables (skip if no models defined)
        try:
            import models
            SQLModel.metadata.create_all(engine)
            logger.info("✅ Database tables created")
        except ImportError:
            logger.info("📝 No models found - using basic database setup")
        
        logger.info("✅ Database initialization complete")
        
    except Exception as e:
        logger.error(f"❌ Database initialization failed: {e}")

def get_session():
    """Get database session for operations."""
    with Session(engine) as session:
        yield session

if __name__ == "__main__":
    if check_connection():
        init_db()
    else:
        logger.error("🚨 Database setup failed")
