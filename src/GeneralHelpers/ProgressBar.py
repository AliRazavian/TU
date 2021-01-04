"""
A progress bar
"""
import sys
import re


class ProgressBar(object):
    """
    A simple progress bar from https://stackoverflow.com/questions/3160699/python-progress-bar
    """
    DEFAULT = 'Progress: %(bar)s %(percent)3d%%'
    FULL = '%(bar)s %(current)d/%(total)d (%(percent)3d%%) %(remaining)d to go'

    def __init__(self, total, width=40, fmt=DEFAULT, symbol='=', output=sys.stderr):
        assert len(symbol) == 1

        self.total = total
        self.width = width
        self.symbol = symbol
        self.output = output
        self.fmt = re.sub(r'(?P<name>%\(.+?\))d', r'\g<name>%dd' % len(str(total)), fmt)

        self.current = 0

    def __call__(self):
        remaining = self.total - self.current

        args = {
            'total': self.total,
            'bar': self.getBar(),
            'current': self.current,
            'percent': self.getProportion() * 100,
            'remaining': remaining
        }
        sys.stdout.write('\r' + self.fmt % args)
        sys.stdout.flush()

    def increment(self, no=1):
        self.current += no
        if self.current > self.total:
            self.current = self.total

    def getProportion(self):
        """
        Retrieves the proportion of the done task
        """
        return self.current / float(max(self.total, 1e-8))

    def getBar(self):
        """
        Retrieves the bar that shows the graphic progress
        """
        size = int(self.width * self.getProportion())
        return '[' + self.symbol * size + ' ' * (self.width - size) + ']'

    def done(self):
        """
        Finishes up the progress bar
        """
        self.current = self.total
        self()
        print('', file=self.output)
