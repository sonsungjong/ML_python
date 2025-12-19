# pip install psycopg2-binary

import psycopg2
from psycopg2 import OperationalError

def create_connection():
    connection = None
    try:
        connection = psycopg2.connect(
            host="115.178.75.126",
            port="5432",
            user="postgres",
            password="postgres123",
            database="postgres"
        )
        print("Connection to PostgreSQL DB successful")
    except OperationalError as e:
        print(f"The error '{e}' occurred")
    return connection

def execute_read_query(connection, query):
    cursor = connection.cursor()
    result = None
    try:
        cursor.execute(query)
        result = cursor.fetchall()
        return result
    except OperationalError as e:
        print(f"The error '{e}' occurred")

if __name__ == "__main__":
    conn = create_connection()
    if conn:
        # 테이블에서 상위 5개 데이터만 가져와서 테스트
        select_query = "SELECT * FROM BT_PRCHS_MTNC_SITU"
        try:
            data = execute_read_query(conn, select_query)
            if data:
                print("\n--- Data from BT_PRCHS_MTNC_SITU ---")
                for row in data:
                    print(row)
            else:
                print("No data found.")
        except Exception as e:
             print(f"Query execution failed: {e}")
            
        conn.close()
