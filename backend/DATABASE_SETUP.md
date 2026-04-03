# MySQL Database Setup for Waste Segregation Project

## Prerequisites

1. **Install MySQL Server**
   - Windows: Download from [MySQL Downloads](https://dev.mysql.com/downloads/installer/)
   - Or use XAMPP/WAMP which includes MySQL

2. **Install Python MySQL Dependencies**
   ```bash
   cd backend
   pip install -r requirements.txt
   ```

## Database Setup Steps

### 1. Create Database

Open MySQL command line or MySQL Workbench and run:

```sql
CREATE DATABASE waste_segregation CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
```

### 2. Configure Database Connection

Create a `.env` file in the `backend/` directory (or edit existing one):

```bash
# Copy the example file
cp .env.example .env
```

Edit `.env` with your MySQL credentials:

```env
DB_HOST=localhost
DB_PORT=3306
DB_USER=root
DB_PASSWORD=your_mysql_password
DB_NAME=waste_segregation
```

### 3. Initialize Database Tables

Run the initialization script:

```bash
cd backend
python init_database.py
```

This will:
- Check database connection
- Create the `predictions` table
- Run a test insert/query

### 4. Database Schema

**predictions** table structure:
```sql
CREATE TABLE predictions (
    id INT AUTO_INCREMENT PRIMARY KEY,
    predicted_class VARCHAR(50) NOT NULL,
    confidence FLOAT NOT NULL,
    timestamp DATETIME NOT NULL,
    image_path VARCHAR(255) NULL,
    user_id VARCHAR(100) NULL,
    INDEX idx_class (predicted_class),
    INDEX idx_timestamp (timestamp)
);
```

## Usage

Once configured, predictions will automatically be saved to MySQL when you use the classifier.

### Viewing Predictions

You can query predictions directly from MySQL:

```sql
USE waste_segregation;

-- View all predictions
SELECT * FROM predictions ORDER BY timestamp DESC LIMIT 10;

-- Get statistics
SELECT 
    predicted_class, 
    COUNT(*) as count,
    AVG(confidence) as avg_confidence
FROM predictions 
GROUP BY predicted_class;
```

## Fallback Behavior

If MySQL is not available, the system will:
- Log a warning
- Continue operating (predictions won't persist)
- Analytics dashboard will show zeros

## Troubleshooting

**Connection Error:**
```
Failed to connect to database!
```
- Ensure MySQL server is running
- Check credentials in `.env`
- Verify database exists

**Import Error:**
```
ModuleNotFoundError: No module named 'mysql'
```
- Run: `pip install -r requirements.txt`

**Table Creation Failed:**
```
Failed to create database tables
```
- Ensure user has CREATE TABLE privileges
- Run: `GRANT ALL PRIVILEGES ON waste_segregation.* TO 'your_user'@'localhost';`
