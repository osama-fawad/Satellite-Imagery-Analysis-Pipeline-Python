import psycopg2

def get_db_connection():
    """Establish connection with CQRE DB"""

    return psycopg2.connect(
        host="localhost",
        port=5432,
        database="cqre",
        user="cqre",
        password="cqre"
    )