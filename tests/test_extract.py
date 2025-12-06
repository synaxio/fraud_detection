from etl.extract import extract_data

def test_extract_returns_dataframe():
    df = extract_data()
    assert not df.empty

