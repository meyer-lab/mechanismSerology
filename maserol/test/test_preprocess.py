import pytest
from tensordata.atyeo import data as atyeo
from tensordata.zohar import data as zohar
from ..preprocess import prepare_data


@pytest.mark.parametrize(
    "data",
    [
        atyeo(),
        zohar(),
    ],
)
def test_prepare_data(data):
    """Test prepare_data() can rotate dims and remove irrelevant receptors."""
    cube = prepare_data(data)
    assert cube.dims[1] == "Ligand"
    assert all([x in cube.Ligand for x in ["IgG1", "IgG3"]])
    assert all(
        [
            x not in cube.Ligand
            for x in ["IgA", "IgA1", "FcRalpha", "IgM", "C1q", "SNA", "ADCC"]
        ]
    )
