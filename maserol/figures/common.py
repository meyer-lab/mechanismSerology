import logging
import sys
import time

def genFigure():
    """ Main figure generation function. """
    logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
    fdir = './output/'
    start = time.time()
    nameOut = 'figure' + sys.argv[1]

    exec('from maserol.figures.' + nameOut + ' import makeFigure', globals())
    ff = makeFigure()
    ff.savefig(fdir + nameOut + '.svg', dpi=300, bbox_inches='tight', pad_inches=0)

    logging.info(f'Figure {sys.argv[1]} is done after {time.time() - start} seconds.')