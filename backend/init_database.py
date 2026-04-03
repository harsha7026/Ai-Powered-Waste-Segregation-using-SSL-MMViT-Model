"""
Database initialization script for MySQL.

Usage:
1. Create MySQL database first:
   mysql -u root -p
   CREATE DATABASE waste_segregation CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
   
2. Configure database credentials in .env file

3. Run this script:
   python init_database.py
"""

import sys
from app.database import init_db, check_db_connection, engine
from app.config import DATABASE_URL


def main():
    print("=" * 60)
    print("Waste Segregation - Database Initialization")
    print("=" * 60)
    print(f"\nDatabase URL: {DATABASE_URL.replace(DATABASE_URL.split('@')[0].split('//')[1], '***')}")
    
    print("\nChecking database connection...")
    if not check_db_connection():
        print("❌ Failed to connect to database!")
        print("\nPlease ensure:")
        print("1. MySQL server is running")
        print("2. Database exists (CREATE DATABASE waste_segregation;)")
        print("3. Credentials in .env or config.py are correct")
        sys.exit(1)
    
    print("✓ Database connection successful")
    
    print("\nCreating database tables...")
    if init_db():
        print("✓ Database tables created successfully")
        print("\nTables created:")
        print("  - predictions (stores all classification results)")
        
        # Test insert
        print("\nTesting database write...")
        try:
            from sqlalchemy.orm import Session
            from app.database import SessionLocal, Prediction
            from datetime import datetime
            
            db: Session = SessionLocal()
            try:
                test_prediction = Prediction(
                    predicted_class="plastic",
                    confidence=0.95,
                    timestamp=datetime.utcnow()
                )
                db.add(test_prediction)
                db.commit()
                
                # Query back
                count = db.query(Prediction).count()
                print(f"✓ Database write successful (Total records: {count})")
                
                # Clean up test record
                db.delete(test_prediction)
                db.commit()
            finally:
                db.close()
        except Exception as e:
            print(f"❌ Database write test failed: {e}")
            sys.exit(1)
        
        print("\n" + "=" * 60)
        print("Database initialization complete!")
        print("You can now start the FastAPI server.")
        print("=" * 60)
    else:
        print("❌ Failed to create database tables")
        sys.exit(1)


if __name__ == "__main__":
    main()
