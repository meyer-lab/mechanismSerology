from ..predictAbundKa import generate_AbundKa, infer_Lbound, model_lossfunc

def test_run():
    generate_AbundKa()
    infer_Lbound()
    model_lossfunc()
