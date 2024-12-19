'''
The module :mod:`tletools.tle` defines the classes :class:`TLE` and :class:`TLEu`.

The library offers two classes to represent a single TLE.
There is the unitless version :class:`TLE`, whose attributes are expressed in the same units
that are used in the TLE format, and there is the unitful version :class:`TLEu`,
whose attributes are quantities (:class:`astropy.units.Quantity`), a type able to represent
a value with an associated unit taken from :mod:`astropy.units`.
Here is a short example of how you can use them:

>>> tle_string = """
... ISS (ZARYA)
... 1 25544U 98067A   19249.04864348  .00001909  00000-0  40858-4 0  9990
... 2 25544  51.6464 320.1755 0007999  10.9066  53.2893 15.50437522187805
... """
>>> tle_lines = tle_string.strip().splitlines()
>>> TLE.from_lines(*tle_lines)
TLE(name='ISS (ZARYA)', norad='25544', ..., n=15.50437522, rev_num=18780)

.. autoclass:: TLE
    :members:
.. autoclass:: TLEu
'''

import attr

import numpy as np
import astropy.units as u
from astropy.time import Time

# Maybe remove them from here?
from poliastro.twobody import Orbit as _Orbit
from poliastro.bodies import Earth as _Earth

from .utils import partition, rev as u_rev, M_to_nu as _M_to_nu

DEG2RAD = np.pi / 180.
RAD2DEG = 180. / np.pi


def _conv_year(s):
    """Interpret a two-digit year string."""
    if isinstance(s, int):
        return s
    y = int(s)
    return y + (1900 if y >= 57 else 2000)


def _parse_decimal(s):
    """Parse a floating point with implicit leading dot.

    >>> _parse_decimal('378')
    0.378
    """
    return float('.' + s)


def _parse_float(s):
    """Parse a floating point with implicit dot and exponential notation.

    >>> _parse_float(' 12345-3')
    0.00012345
    >>> _parse_float('+12345-3')
    0.00012345
    >>> _parse_float('-12345-3')
    -0.00012345
    """
    return float(s[0] + '.' + s[1:6] + 'e' + s[6:8])


def _float_to_string(f: float, digits: int = 8) -> str:
    """Convert a float to a string with implicit dot and exponential notation.

    >>> _float_to_string(0.00012345, digits)
    '12345-3'
    """
    if f == 0:
        # zero gets a special string
        return "0" * digits + "-0"
    format_string = '{:.' + str(digits) + 'E}'
    s = format_string.format(f)
    # skip the first zero in the exponent, and if it is +0, make it -0 to confirm to the TLE convention
    exponent = int(s[digits + 3] + s[digits + 4] + s[digits + 5])
    exponent = exponent + 1

    # skip the decimal point, and E
    return s[0] + s[2:digits + 1] + str(exponent)

def _calculate_check_sum_on_tle_line(line: str) -> int:
    """Calculate the checksum of a TLE line.

    The checksum is calculated by taking the sum of all the digits in the line, ignoring spaces, and then taking the
    modulo 10 of that sum.

    >>> _calculate_check_sum_on_tle_line('1 25544U 98067A   19249.04864348  .00001909  00000-0  40858-4 0  9990')
    0
    """
    sum_of_digits = sum(int(c) for c in line if c.isdigit()) + sum(1 for c in line if c == '-')
    return sum_of_digits % 10


@attr.s
class TLE:
    """Data class representing a single TLE.

    A two-line element set (TLE) is a data format encoding a list of orbital
    elements of an Earth-orbiting object for a given point in time, the epoch.

    All the attributes parsed from the TLE are expressed in the same units that
    are used in the TLE format.

    :ivar str name:
        Name of the satellite.
    :ivar str norad:
        NORAD catalog number (https://en.wikipedia.org/wiki/Satellite_Catalog_Number).
    :ivar str classification:
        'U', 'C', 'S' for unclassified, classified, secret.
    :ivar str int_desig:
        International designator (https://en.wikipedia.org/wiki/International_Designator),
    :ivar int epoch_year:
        Year of the epoch.
    :ivar float epoch_day:
        Day of the year plus fraction of the day.
    :ivar float dn_o2:
        First time derivative of the mean motion divided by 2.
    :ivar float ddn_o6:
        Second time derivative of the mean motion divided by 6.
    :ivar float bstar:
        BSTAR coefficient (https://en.wikipedia.org/wiki/BSTAR).
    :ivar int set_num:
        Element set number.
    :ivar float inc:
        Inclination.
    :ivar float raan:
        Right ascension of the ascending node.
    :ivar float ecc:
        Eccentricity.
    :ivar float argp:
        Argument of perigee.
    :ivar float M:
        Mean anomaly.
    :ivar float n:
        Mean motion.
    :ivar int rev_num:
        Revolution number.
    """

    # name of the satellite
    name = attr.ib(converter=str.strip)
    # NORAD catalog number (https://en.wikipedia.org/wiki/Satellite_Catalog_Number)
    norad = attr.ib(converter=str.strip)
    classification = attr.ib()
    int_desig = attr.ib(converter=str.strip)
    epoch_year = attr.ib(converter=_conv_year)
    epoch_day = attr.ib()
    dn_o2 = attr.ib()
    ddn_o6 = attr.ib()
    bstar = attr.ib()
    set_num = attr.ib(converter=int)
    inc = attr.ib()
    raan = attr.ib()
    ecc = attr.ib()
    argp = attr.ib()
    M = attr.ib()
    n = attr.ib()
    rev_num = attr.ib(converter=int)

    def __attrs_post_init__(self):
        self._epoch = None
        self._a = None
        self._nu = None

    @property
    def epoch(self):
        """Epoch of the TLE, as an :class:`astropy.time.Time` object."""
        if self._epoch is None:
            year = np.datetime64(self.epoch_year - 1970, 'Y')
            day = np.timedelta64(int((self.epoch_day - 1) * 86400 * 10 ** 6), 'us')
            self._epoch = Time(year + day, format='datetime64', scale='utc')
        return self._epoch

    @property
    def a(self):
        """Semi-major axis."""
        if self._a is None:
            self._a = (_Earth.k.value / (self.n * np.pi / 43200) ** 2) ** (1 / 3) / 1000
        return self._a

    @property
    def nu(self):
        """True anomaly."""
        if self._nu is None:
            # Make sure the mean anomaly is between -pi and pi
            M = ((self.M + 180) % 360 - 180) * DEG2RAD
            self._nu = _M_to_nu(M, self.ecc) * RAD2DEG
        return self._nu

    @classmethod
    def from_lines(cls, name, line1, line2):
        """Parse a TLE from its constituent lines.

        All the attributes parsed from the TLE are expressed in the same units that
        are used in the TLE format.
        """
        return cls(
            name=name,
            norad=line1[2:7],
            classification=line1[7],
            int_desig=line1[9:17],
            epoch_year=line1[18:20],
            epoch_day=float(line1[20:32]),
            dn_o2=float(line1[33:43]),
            ddn_o6=_parse_float(line1[44:52]),
            bstar=_parse_float(line1[53:61]),
            set_num=line1[64:68],
            inc=float(line2[8:16]),
            raan=float(line2[17:25]),
            ecc=_parse_decimal(line2[26:33]),
            argp=float(line2[34:42]),
            M=float(line2[43:51]),
            n=float(line2[52:63]),
            rev_num=line2[63:68])

    @classmethod
    def load(cls, filename):
        """Load multiple TLEs from a file."""
        if isinstance(filename, str):
            with open(filename) as fp:
                return [cls.from_lines(*l012)
                        for l012 in partition(fp, 3)]
        else:
            return [tle for fn in filename for tle in cls.load(fn)]

    @classmethod
    def loads(cls, string):
        """Load multiple TLEs from a string."""
        return [cls.from_lines(*l012) for l012 in partition(string.split('\n'), 3)]

    def to_orbit(self, attractor=_Earth):
        '''Convert to a :class:`poliastro.twobody.orbit.Orbit` around the attractor.

        >>> tle_string = """ISS (ZARYA)
        ... 1 25544U 98067A   19249.04864348  .00001909  00000-0  40858-4 0  9990
        ... 2 25544  51.6464 320.1755 0007999  10.9066  53.2893 15.50437522187805"""
        >>> tle = TLE.from_lines(*tle_string.splitlines())
        >>> tle.to_orbit()
        6788 x 6799 km x 51.6 deg (GCRS) orbit around Earth (♁) at epoch 2019-09-06T01:10:02.796672000 (UTC)
        '''
        return _Orbit.from_classical(
            attractor=attractor,
            a=u.Quantity(self.a, u.km),
            ecc=u.Quantity(self.ecc, u.one),
            inc=u.Quantity(self.inc, u.deg),
            raan=u.Quantity(self.raan, u.deg),
            argp=u.Quantity(self.argp, u.deg),
            nu=u.Quantity(self.nu, u.deg),
            epoch=self.epoch)

    @classmethod
    def from_orbit(cls, orbit:_Orbit, name = "NOT ASSIGNED", norad = "00000", classification="U", int_desig="00000A",
                   dn_o2=0, ddn_o6=0, bstar=0, set_num=999, rev_num=999):
        '''Convert from a :class:`poliastro.twobody.orbit.Orbit` around the attractor.

        >>> from poliastro.twobody import Orbit
        >>> from astropy import units as u
        >>> from poliastro.bodies import Earth
        >>> from astropy.time import Time
        >>> orbit_epoch_str = "2024-02-01T00:29:29.126688Z"
        >>> orbit_epoch_time = Time.strptime(orbit_epoch_str, "%Y-%m-%dT%H:%M:%S.%fZ")
        >>> orbit = Orbit.from_classical(
        ...     attractor=Earth,
        ...     a = 6788 << u.km,
        ...     ecc = 0.0007999 << u.one,
        ...     inc = 51.6464 << u.deg,
        ...     raan= 320.1755 << u.deg,
        ...     argp= 10.9066 << u.deg,
        ...     nu= 53.2893 << u.deg,
        ...     epoch = orbit_epoch_time
        ... )
        >>> tle = TLE.from_orbit(orbit)
        >>> tle.to_lines()
        '1 00000U 00000A   24032.02000000  .00000000  00000-0  00000-0 0  9999\\n2 00000  51.6464 320.1755 0007999  10.9066  53.2893 15.50437522  999'
        '''
        return cls(
            name=name,
            norad=norad,
            classification=classification,
            int_desig=int_desig,
            epoch_year=orbit.epoch.to_datetime().year,
            epoch_day=orbit.epoch.to_datetime().timetuple().tm_yday + orbit.epoch.to_datetime().hour / 24 + orbit.epoch.to_datetime().minute / (
                    24 * 60) + orbit.epoch.to_datetime().second / (
                                     24 * 60 * 60) + orbit.epoch.to_datetime().microsecond / (24 * 60 * 60 * 1000000),
            dn_o2=dn_o2,
            ddn_o6=ddn_o6,
            bstar=bstar,
            set_num=set_num,
            inc=orbit.inc.to(u.deg).value,
            raan=orbit.raan.to(u.deg).value,
            ecc=orbit.ecc.value,
            argp=orbit.argp.to(u.deg).value,
            # M=orbit.n * (24 * 60 * 60 * u.s) << u.deg, # mean motion
            # n=orbit.nu.to(u.deg).value,
            M = orbit.nu.to(u.deg).value, # orbit.nu.to(u.deg).value, # mean anomaly
            n = orbit.n, # mean motion
            rev_num=rev_num
        )



    def astuple(self):
        """Return a tuple of the attributes."""
        return attr.astuple(self)

    def asdict(self, computed=False, epoch=False):
        """Return a dict of the attributes."""
        d = attr.asdict(self)
        if computed:
            d.update(a=self.a, nu=self.nu)
        if epoch:
            d.update(epoch=self.epoch)
        return d

    def to_lines(self):
        templates = [
            "{name}",
            "1 {norad}{classification} {int_desig}   {epoch_year_last_digits:02d}{epoch_day:12.8f}  {dn_o2_wo_leading_zero}  {ddn_o6_wo_e}  {bstar_wo_e} 0  {set_num:3d}",
            "2 {norad} {inc:8.4f} {raan:8.4f} {ecc_wo_leading_zero} {argp:8.4f} {M:8.4f} {n:11.8f}{rev_num:4d}",
        ]
        additional_dict = {
            'epoch_year_last_digits': self.epoch_year % 100,
            'dn_o2_wo_leading_zero': "{dn_o2:.8f}".format(dn_o2=self.dn_o2).lstrip('0'),  # dn_o2 without leading zero
            'ddn_o6_wo_e': _float_to_string(self.ddn_o6, digits=5),
            'line1_check_sum': 0,  # TODO: implement
            'bstar_wo_e': _float_to_string(self.bstar, digits=5),
            'ecc_wo_leading_zero': "{ecc:.7f}".format(ecc=self.ecc).lstrip('0').lstrip('.'),  # ecc without leading zero
            'line2_check_sum': 5,  # TODO: implement
        }
        lines = [template.format(**{**self.asdict(), **additional_dict}) for template in templates]
        line_1_mod = _calculate_check_sum_on_tle_line(lines[1])
        line_2_mod = _calculate_check_sum_on_tle_line(lines[2])
        lines[1] = lines[1] + str(line_1_mod)
        lines[2] = lines[2] + str(line_2_mod)

        return "\n".join(lines)


@attr.s
class TLEu(TLE):
    """Unitful data class representing a single TLE.

    This is a subclass of :class:`TLE`, so refer to that class for a description
    of the attributes, properties and methods.

    The only difference here is that all the attributes are quantities
    (:class:`astropy.units.Quantity`), a type able to represent a value with
    an associated unit taken from :mod:`astropy.units`.
    """

    @property
    def a(self):
        """Semi-major axis."""
        if self._a is None:
            self._a = (_Earth.k.value / self.n.to_value(u.rad / u.s) ** 2) ** (1 / 3) * u.m
        return self._a

    @property
    def nu(self):
        """True anomaly."""
        if self._nu is None:
            # Make sure the mean anomaly is between -pi and pi
            M = ((self.M.to_value(u.rad) + 180) % 360 - 180) * DEG2RAD
            ecc = self.ecc.to_value(u.one)
            nu_rad = _M_to_nu(M, ecc)
            self._nu = nu_rad * RAD2DEG * u.deg
        return self._nu

    @classmethod
    def from_lines(cls, name, line1, line2):
        """Parse a TLE from its constituent lines."""
        return cls(
            name=name,
            norad=line1[2:7],
            classification=line1[7],
            int_desig=line1[9:17],
            epoch_year=line1[18:20],
            epoch_day=float(line1[20:32]),
            dn_o2=u.Quantity(float(line1[33:43]), u_rev / u.day ** 2),
            ddn_o6=u.Quantity(_parse_float(line1[44:52]), u_rev / u.day ** 3),
            bstar=u.Quantity(_parse_float(line1[53:61]), 1 / u.earthRad),
            set_num=line1[64:68],
            inc=u.Quantity(float(line2[8:16]), u.deg),
            raan=u.Quantity(float(line2[17:25]), u.deg),
            ecc=u.Quantity(_parse_decimal(line2[26:33]), u.one),
            argp=u.Quantity(float(line2[34:42]), u.deg),
            M=u.Quantity(float(line2[43:51]), u.deg),
            n=u.Quantity(float(line2[52:63]), u_rev / u.day),
            rev_num=line2[63:68])

    def to_orbit(self, attractor=_Earth):
        """Convert to an orbit around the attractor."""
        return _Orbit.from_classical(
            attractor=attractor,
            a=self.a,
            ecc=self.ecc,
            inc=self.inc,
            raan=self.raan,
            argp=self.argp,
            nu=self.nu,
            epoch=self.epoch)
