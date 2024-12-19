from tletools import TLE
from tletools.tle import TLEu

import astropy.units as u

def test_from_lines(tle_lines):
    t = TLE.from_lines(*tle_lines)
    assert isinstance(t, TLE)

def test_from_lines_high_M(tle_lines2):
    t = TLE.from_lines(*tle_lines2)
    assert isinstance(t, TLE)

def test_from_lines_with_units(tle_lines):
    t = TLEu.from_lines(*tle_lines)
    assert isinstance(t, TLEu)


def test_to_orbit(tle):
    assert tle.to_orbit().ecc == tle.ecc
    assert float(tle.to_orbit().inc / u.deg) == tle.inc


def test_asdict(tle):
    assert type(tle)(**tle.asdict()) == tle


def test_astuple(tle):
    assert type(tle)(*tle.astuple()) == tle

def test_to_lines(tle_lines):
    t = TLE.from_lines(*tle_lines)
    lines = t.to_lines()
    tle_joined_lines = '\n'.join(tle_lines)
    assert(lines == tle_joined_lines)

def test_orbit_to_lines(tle_lines):
    tle = TLE.from_lines(*tle_lines)
    orbit = tle.to_orbit()
    # lines = orbit.to_lines()

    lines_from_orbit = TLE.from_orbit(orbit).to_lines()
    # tle_from_orbit = TLE.from_lines(*lines)
    # didn't get right:
    # * mean anomaly
    # * mean motion
    assert tle_lines == lines_from_orbit