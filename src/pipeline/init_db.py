import os
import pandas as pd
from src.db.database import engine, Base, SessionLocal
from src.db.models import Book
from src.config import settings

def init_db():
    print("Creating database tables...")
    Base.metadata.create_all(bind=engine)
    
    print("Checking if we need to seed data from CSV...")
    db = SessionLocal()
    
    # Check if DB is empty
    count = db.query(Book).count()
    if count == 0:
        csv_path = os.path.join(settings.DATA_RAW_DIR, settings.RAW_DATA_FILE)
        if os.path.exists(csv_path):
            print(f"Seeding database from {csv_path}...")
            df = pd.read_csv(csv_path)
            df = df.fillna("")
            
            for _, row in df.iterrows():
                book = Book(
                    title=str(row.get('title', '')).strip(),
                    author=str(row.get('author', '')).strip(),
                    genre=str(row.get('genre', '')).strip(),
                    synopsis=str(row.get('synopsis', '')).strip()
                )
                db.add(book)
            
            db.commit()
            print(f"Successfully seeded {len(df)} books into the database.")
        else:
            print("No raw CSV found to seed.")
    else:
        print(f"Database already contains {count} books. Skipping seed.")
        
    db.close()

if __name__ == "__main__":
    init_db()
