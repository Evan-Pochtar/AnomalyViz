from src.createData import createSampleDataset
from src.core import runAll, aggregate
from src.data import clean

def test_runAll():
    df = createSampleDataset(n_samples=500, contamination=0.05)

    results = runAll(clean(df))
    agreement = aggregate(results)

    assert isinstance(results, dict)
    assert isinstance(agreement, dict)
    assert len(results) > 0
    assert len(agreement) > 0

    consensus = {idx: count for idx, count in agreement.items() if count > 4}
    assert len(consensus) == 25
