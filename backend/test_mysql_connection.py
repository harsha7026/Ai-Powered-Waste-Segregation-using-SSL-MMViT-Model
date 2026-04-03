"""
Quick test script to verify MySQL connection and create database.
Run this after setting your password in .env file.
"""

import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

import mysql.connector
from mysql.connector import Error
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = int(os.getenv("DB_PORT", "3306"))
DB_USER = os.getenv("DB_USER", "root")
DB_PASSWORD = os.getenv("DB_PASSWORD", "")
DB_NAME = os.getenv("DB_NAME", "waste_segregation")

print("=" * 60)
print("MySQL Connection Test")
print("=" * 60)
print(f"Host: {DB_HOST}")
print(f"Port: {DB_PORT}")
print(f"User: {DB_USER}")
print(f"Database: {DB_NAME}")
print(f"Password: {'(set)' if DB_PASSWORD else '(empty)'}")
print()

try:
    # Try connecting without specifying database first
    print("Step 1: Testing MySQL server connection...")
    connection = mysql.connector.connect(
        host=DB_HOST,
        port=DB_PORT,
        user=DB_USER,
        password=DB_PASSWORD
    )
    
    if connection.is_connected():
        print("✅ Successfully connected to MySQL server!")
        
        cursor = connection.cursor()
        
        # Check if database exists
        print(f"\nStep 2: Checking if database '{DB_NAME}' exists...")
        cursor.execute("SHOW DATABASES")
        databases = [db[0] for db in cursor.fetchall()]
        
        if DB_NAME in databases:
            print(f"✅ Database '{DB_NAME}' already exists!")
        else:
            print(f"⚠️  Database '{DB_NAME}' does not exist. Creating it...")
            cursor.execute(f"CREATE DATABASE {DB_NAME} CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci")
            print(f"✅ Database '{DB_NAME}' created successfully!")
        
        cursor.close()
        connection.close()
        
        print("\n" + "=" * 60)
        print("✅ All checks passed! You can now run:")
        print("   python init_database.py")
        print("=" * 60)
        
except Error as e:
    print(f"❌ Connection failed: {e}")
    print("\n" + "=" * 60)
    print("Troubleshooting:")
    print("=" * 60)
    
    if "Access denied" in str(e):
        print("1. Check your MySQL password in the .env file")
        print("2. Common passwords: '', 'root', 'password', 'admin'")
        print("3. Or reset password using MySQL Workbench/phpMyAdmin")
    elif "Can't connect" in str(e):
        print("1. Ensure MySQL service is running")
        print("2. Check if port 3306 is correct")
    else:
        print("1. Check MySQL installation")
        print("2. Verify credentials")
    
    sys.exit(1)
