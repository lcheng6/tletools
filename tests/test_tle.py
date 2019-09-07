from tle import TLE

import sys


def test_from_lines(tle_lines):
    print(tle_lines, file=sys.stderr)
    t = TLE.from_lines(*tle_lines)
    assert isinstance(t, TLE)


def test_to_orbit(tle):
    assert tle.to_orbit().ecc == tle.ecc


def test_asdict(tle):
    assert type(tle)(**tle.asdict()) == tle


def test_astuple(tle):
    assert type(tle)(*tle.astuple()) == tle