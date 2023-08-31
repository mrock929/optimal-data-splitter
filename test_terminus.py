import pytest
import pandas as pd
from terminus import Terminus


@pytest.fixture
def sample_data() -> pd.DataFrame:
    return pd.read_csv('./sample_data.csv')


def test_init(sample_data):
    """Test the Terminus class initialization"""
    terminus = Terminus(data=sample_data)

    assert terminus.val_percent == 0.1
    assert terminus.save_output is True


def test_split_val_only(sample_data):
    """Test the split data function for only a val split"""
    config = {'test_percent': 0.0, 'num_iterations': 1}
    terminus = Terminus(data=sample_data, config=config)

    output_data = terminus.split_data()

    assert output_data.loc[output_data['split'] == 'test'].shape[0] == 0
    assert output_data.loc[output_data['split'] == 'val'].shape[0] != 0


def test_split_test_only(sample_data):
    """Test the split data function for only a test split"""
    config = {'val_percent': 0.0, 'num_iterations': 1}
    terminus = Terminus(data=sample_data, config=config)

    output_data = terminus.split_data()

    assert output_data.loc[output_data['split'] == 'val'].shape[0] == 0
    assert output_data.loc[output_data['split'] == 'test'].shape[0] != 0


def test_split(sample_data):
    """Test the split data function for a generic case"""
    config = {'test_percent': 0.2, 'val_percent': 0.16,  'num_iterations': 1}
    terminus = Terminus(data=sample_data, config=config)

    output_data = terminus.split_data()

    assert output_data.loc[output_data['split'] == 'test'].shape[0] != 0
    assert output_data.loc[output_data['split'] == 'val'].shape[0] != 0


def test_update_config(sample_data):
    """Test that the config parameters are updated correctly"""
    config = {'val_percent': 0.9, 'num_iterations': 1}

    terminus = Terminus(data=sample_data, config=config)

    assert terminus.val_percent == 0.9
    assert terminus.num_iterations == 1
    assert terminus.save_output is True
