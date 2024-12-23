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


def test_to_lines(tle_lines, tle_from_lines_to_lines_expected):
    t = TLE.from_lines(*tle_lines)
    lines = t.to_lines()
    tle_joined_lines = '\n'.join(tle_lines)
    assert (lines == tle_from_lines_to_lines_expected)

def test_orbit_to_lines(tle_lines, tle_from_orbit_to_lines_expected):
    tle = TLE.from_lines(*tle_lines)
    orbit = tle.to_orbit()
    # lines = orbit.to_lines()

    lines_from_orbit = TLE.from_orbit(
        orbit,
        name=tle.name,
        int_desig=tle.int_desig,
        classification=tle.classification,
        norad=tle.norad,
        dn_o2=tle.dn_o2,
        ddn_o6=tle.ddn_o6,
        bstar=tle.bstar,
        rev_num=tle.rev_num,
    ).to_lines()
    assert lines_from_orbit == tle_from_orbit_to_lines_expected


def test_from_orbit_to_lines2(tle_lines2, tle_lines2_from_orbit_expected):
    tle = TLE.from_lines(*tle_lines2)
    orbit = tle.to_orbit()
    # lines = orbit.to_lines()

    lines_from_orbit = TLE.from_orbit(
        orbit,
        name=tle.name,
        int_desig=tle.int_desig,
        classification=tle.classification,
        norad=tle.norad,
        dn_o2=tle.dn_o2,
        ddn_o6=tle.ddn_o6,
        bstar=tle.bstar,
        rev_num=tle.rev_num,
    ).to_lines()
    assert tle_lines2_from_orbit_expected == lines_from_orbit
