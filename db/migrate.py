from sqlalchemy import create_engine, text
from pathlib import Path
from config import DB_URL

def main():
    engine = create_engine(DB_URL)
    sql = Path("db/schema.sql").read_text(encoding="utf-8")

    with engine.connect() as conn:
        for stml in sql.split(";"):
            if stml.strip():
                conn.execute(text(stml))
        conn.commit()

    print("DB schema applied.")

if __name__ == "__main__":
    main()