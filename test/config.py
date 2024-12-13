import os
from os.path import join, dirname
from dotenv import load_dotenv


dotenv_path = join(os.getcwd(), ".env")
print(dotenv_path)
load_dotenv(dotenv_path)

TEST_API_TOKEN = os.getenv("TEST_API_TOKEN")

TEST_API_PORT = os.getenv("TEST_API_PORT")
TEST_API_URL = os.getenv("TEST_API_URL", "http://localhost") + ":" + TEST_API_PORT
TEST_API_BACKENDS = os.getenv("TEST_API_BACKENDS")
