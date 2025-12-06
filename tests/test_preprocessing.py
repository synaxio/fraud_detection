from training.preprocessing import build_preprocessing
import pandas as pd

def test_preprocessing_shape():
    prep = build_preprocessing()
    df = pd.DataFrame({...})
    out = prep.fit_transform(df)
    assert out.shape[0] == df.shape[0]
