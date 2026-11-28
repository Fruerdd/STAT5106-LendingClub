"""Check the result-comparison rules used by Member 6."""
import importlib.util
from pathlib import Path
import gzip

spec = importlib.util.spec_from_file_location('reproduction6', Path(__file__).resolve().parents[1]/'analysis/06_reproducibility_member6.py')
r6 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r6)


def test_csv_comparison_allows_roundoff_but_rejects_count_and_category_changes(tmp_path):
    expected, actual = tmp_path/'expected.csv', tmp_path/'actual.csv'
    expected.write_text('count,metric,group\n1000000,0.7,A\n')
    actual.write_text('count,metric,group\n1000000,0.700000001,A\n')
    assert r6.compare_csv(expected, actual)[0]
    for value in ['1000001,0.7,A', '1000000,,A', '1000000,0.7,B']:
        actual.write_text('count,metric,group\n'+value+'\n')
        assert not r6.compare_csv(expected, actual)[0]


def test_decompressed_checksum_ignores_gzip_timestamp(tmp_path):
    a, b = tmp_path/'a.gz', tmp_path/'b.gz'
    a.write_bytes(gzip.compress(b'unchanged csv', mtime=1))
    b.write_bytes(gzip.compress(b'unchanged csv', mtime=2))
    assert r6.sha256(a) != r6.sha256(b)
    assert r6.sha256(a, True) == r6.sha256(b, True)
