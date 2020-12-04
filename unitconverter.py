class UnitConverter:
    """
    Used to convert units from neat input format to SI units for calculation
    """
    def __init__(self, m_unit='m_earth', d_unit='au', t_unit='days', v_unit='km/d'):
        if m_unit == 'm_earth':
            self.m = 5.9722 * 1e24
            self.mname = 'm_earth'
        else:
            self.m = 1
            self.mname = 'kg'
            print('m SI')

        if d_unit == 'km':
            self.d = 1000
            self.dname = 'km'
        elif d_unit == 'au':
            self.d = 1.496 * 1e11
            self.dname = 'au'
        else:
            self.d = 1
            self.dname = 'm'
            print('d SI')

        if t_unit == 'days':
            self.t = 86400
            self.tname = 'jd'
        else:
            self.t = 1
            self.tname = 's'
            print('t SI')

        if v_unit == 'km/d':
            self.v = 0.01157
            self.vname = 'km/d'
        elif v_unit == 'au/d':
            self.v = 1.731 * 1e6
            self.vname = 'au/d'
        else:
            self.v = 1
            self.vname = 'm/s'
            print('v SI')

