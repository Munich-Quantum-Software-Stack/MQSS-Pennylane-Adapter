import os

TEST_API_TOKEN = os.getenv("TEST_API_TOKEN")

TEST_API_PORT = os.getenv("TEST_API_PORT")
TEST_API_URL = os.getenv("TEST_API_URL", "http://localhost") + ":" + TEST_API_PORT
TEST_API_BACKENDS = os.getenv("TEST_API_BACKENDS")
