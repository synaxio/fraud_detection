from etl import extract

def test_extract_raw_transaction():
    data = extract.extract_and_store_raw_transactions()
    assert len(data) > 0

