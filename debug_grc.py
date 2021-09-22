import pesticide
import dbbs_models
import arbor

decor = pesticide.DecorSpy()
mech = arbor.mechanism("expsyn")
decor.place("(root)", mech, "soma_synapse")
decor.place("(root)", arbor.iclamp(200, 600, 0.1), "iclamp")

patient = dbbs_models.GolgiCell.as_patient(decor=decor)
fum = pesticide.Fumigation([patient])
patient.plot_morphology()
pesticide.apply(fum)
