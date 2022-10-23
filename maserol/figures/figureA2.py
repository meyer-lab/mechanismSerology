"""
This is an internal figure which does not necessarily have a place in final
paper.
"""
from tensordata.zohar import data3D as zohar

from maserol.core import prepare_data
from maserol.correlation import plot_leave_out_rec_lbound_correlation

def makeFigure():
    data = prepare_data(zohar(xarray=True, logscale=False))
    return plot_leave_out_rec_lbound_correlation(data, "FcR2B", metric="rtot")
