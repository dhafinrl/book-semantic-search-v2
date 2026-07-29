import os
import pandas as pd
from src.config import settings
from src.db.database import engine

def load_data_from_db() -> pd.DataFrame:
    """Loads raw data from SQLite Database."""
    return pd.read_sql("SELECT * FROM books", engine)

def clean_and_merge(df: pd.DataFrame) -> pd.DataFrame:
    """Fills NaNs and merges columns into a single rich_text column."""
    df = df.fillna("")
    
    def create_rich_text(row):
        title = str(row.get('title', '')).strip()
        author = str(row.get('author', '')).strip()
        genre = str(row.get('genre', '')).strip()
        synopsis = str(row.get('synopsis', '')).strip()
        
        # Combine into a structured text for better semantic understanding
        rich_text = f"Title: {title}. Author: {author}. Genre: {genre}. Synopsis: {synopsis}"
        return rich_text

    df['rich_text'] = df.apply(create_rich_text, axis=1)
    return df

def preprocess_data():
    output_path = os.path.join(settings.DATA_PROCESSED_DIR, settings.PROCESSED_DATA_FILE)
    
    print("Loading data from SQLite Database...")
    df = load_data_from_db()
    
    if df.empty:
        print("Database is empty. Nothing to preprocess.")
        return
        
    print(f"Loaded {len(df)} records. Cleaning and merging...")
    processed_df = clean_and_merge(df)
    
    print(f"Saving processed data to {output_path}...")
    # Ensure processed directory exists
    os.makedirs(settings.DATA_PROCESSED_DIR, exist_ok=True)
    processed_df.to_pickle(output_path)
    
    print("Preprocessing completed successfully!")

if __name__ == "__main__":
    preprocess_data()
