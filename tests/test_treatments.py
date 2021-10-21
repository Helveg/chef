import pesticide as pst
import dbbs_models
import arbor
import unittest

class TestTreatments(unittest.TestCase):
    def test_fumigation(self):
        decor = pst.DecorSpy()
        mech = arbor.mechanism("expsyn")
        decor.place("(root)", mech, "soma_synapse")
        decor.place("(root)", arbor.iclamp(200, 600, 0.1), "iclamp")

        tree = dbbs_models.GolgiCell.as_tree(decor=decor)
        fum = pst.Fumigation([tree])
        patient.plot_morphology()
        pst.apply(fum)
