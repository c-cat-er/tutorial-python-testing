import pytest
import pandas as pd

@pytest.fixture
def sample_mes_data():
    return pd.DataFrame({"level": ["INFO", "CRITICAL"], "severity": [0, 3]})