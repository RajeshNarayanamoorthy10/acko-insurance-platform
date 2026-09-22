"""Run once: creates acko_platform.db with the chat_logs and quotations tables."""

from src.db.database import init_db

if __name__ == "__main__":
    init_db()
    print("Tables created successfully.")