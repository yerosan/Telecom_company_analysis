import psycopg2
from psycopg2 import sql, OperationalError


class DataBaseConnection:
    def __init__(self, host, database, user, password, port=5432):
        self.host=host
        self.database=database
        self.user=user
        self.password=password
        self.port=port

    def create_connection(self):
        """
        Creates a connection to the PostgreSQL database.

        :param host: The hostname or IP address of the PostgreSQL server.
        :param database: The name of the database to connect to.
        :param user: The database user.
        :param password: The user's password.
        :param port: The port number (default: 5432).
        :return: A connection object or None if connection fails.
        """
        connection = None
        try:
            connection = psycopg2.connect(
                host=self.host,
                database=self.database,
                user=self.user,
                password=self.password,
                port=self.port
            )
            print("Connection to database successful")
        except OperationalError as e:
            print(f"The error '{e}' occurred")
        return connection

    def execute_query(self,connection, query, params=None):
        """
        Executes a single query on the connected PostgreSQL database.

        :param connection: A valid database connection object.
        :param query: The SQL query to be executed.
        :param params: Optional query parameters (for parameterized queries).
        :return: None
        """
        cursor = connection.cursor()
        try:
            cursor.execute(query, params)
            connection.commit()
            # print("Query executed successfully")
        except OperationalError as e:
            print(f"The error '{e}' occurred")
        finally:
            cursor.close()

    def fetch_data(self, connection, query, params=None):
        """
        Executes a SELECT query and fetches the data from the database.

        :param connection: A valid database connection object.
        :param query: The SELECT query to be executed.
        :param params: Optional query parameters (for parameterized queries).
        :return: List of rows retrieved from the database.
        """
        cursor = connection.cursor()
        try:
            cursor.execute(query, params)
            result = cursor.fetchall()
            columns=[desc[0] for desc in cursor.description] # extracting columns
            return result, columns
        except OperationalError as e:
            print(f"The error '{e}' occurred")
        finally:
            cursor.close()

    def close_connection(self,connection):
        """
        Closes the database connection.

        :param connection: A valid database connection object.
        :return: None
        """
        if connection:
            connection.close()
            print("Connection to database closed")
