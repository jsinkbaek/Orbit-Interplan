class UnitConverter:
    """
    Used to convert units from neat input format to SI units for calculatuiton
    """
    def __init__(self, m_unit='m_earth', d_unit='km', t_unit='days'):
        if m_unit == 'm_earth':
            self.m = 5.9722 * 1e24
            self.mname = 'm_earth'
        else:
            self.m = 1
            self.mname = 'kg'
            print('no yet')
        if d_unit == 'km':
            self.d = 1000
            self.dname = 'km'
        else:
            self.d = 1
            self.dname = 'm'
            print('no yet')
        if t_unit == 'days':
            self.t = 86400
            self.tname = 'd'
        else:
            self.t = 1
            self.tname = 's'
            print('no yet')

