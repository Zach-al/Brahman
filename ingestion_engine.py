import sqlite3
import requests
import json
import os
from pathlib import Path

class BrahmanIngestion:
    def __init__(self, db_path="data/brahman_v2.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.db_path)
        self.init_db()

    def init_db(self):
        cursor = self.conn.cursor()
        # 1. Dhatu Table: The 2000+ Roots
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS dhatus (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                root TEXT NOT NULL,
                gana INTEGER,
                pada TEXT,
                meaning TEXT,
                tags TEXT
            )
        ''')
        # 2. Sutra Table: Panini's Rules
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sutras (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                category TEXT
            )
        ''')
        self.conn.commit()

    def ingest_dhatupatha(self):
        print("="*60)
        print("PHASE 3: DHĀTUPĀṬHA INGESTION")
        print("="*60)
        
        # Pulling verified structured data from drdhaval2785
        url = "https://raw.githubusercontent.com/drdhaval2785/SanskritVerb/master/Data/dhaatugana.txt"
        
        # Gana mapping
        gana_map = {
            'भ्वादिः': 1, 'अदादिः': 2, 'जुहोत्यादिः': 3, 'दिवादिः': 4,
            'स्वादिः': 5, 'तुदादिः': 6, 'रुधादिः': 7, 'तनादिः': 8,
            'क्र्यादिः': 9, 'चुरादिः': 10
        }
        
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            lines = response.text.strip().split('\n')
            
            cursor = self.conn.cursor()
            cursor.execute("DELETE FROM dhatus") # Fresh start
            
            count = 0
            for line in lines:
                parts = line.strip().split('\t')
                if len(parts) >= 6:
                    # Index, Gana, Root, Meaning, It, Pada
                    gana_text = parts[1].strip()
                    root_raw = parts[2].strip()
                    meaning = parts[3].strip()
                    pada = parts[5].strip()
                    
                    # Extract root inside parenthesis if available, e.g., 'भू ( भू )' -> 'भू'
                    root = root_raw
                    if '(' in root_raw and ')' in root_raw:
                        root = root_raw.split('(')[1].split(')')[0].strip()
                    
                    gana_num = gana_map.get(gana_text, 1)
                    
                    cursor.execute('''
                        INSERT INTO dhatus (root, gana, pada, meaning)
                        VALUES (?, ?, ?, ?)
                    ''', (root, gana_num, pada, meaning))
                    count += 1
            
            self.conn.commit()
            print(f"✓ Successfully mapped {count} verbal roots into SQLite.")
        except Exception as e:
            print(f"✗ Ingestion Failed: {e}")

    def close(self):
        self.conn.close()

if __name__ == "__main__":
    ingestor = BrahmanIngestion()
    ingestor.ingest_dhatupatha()
    ingestor.close()
