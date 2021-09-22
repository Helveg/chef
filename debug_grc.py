from chef import bake, Ingredient, DecorSpy, ProbeSoufflé
import dbbs_models
import arbor

decor = DecorSpy()
mech = arbor.mechanism("expsyn")
decor.place("(root)", mech, "soma_synapse")
decor.place("(root)", arbor.iclamp(200, 600, 0.1), "iclamp")

ingr = dbbs_models.PurkinjeCell.as_ingredient(decor=decor)
soufflé = ProbeSoufflé([ingr])
pwlin = arbor.place_pwlin(ingr.morphology)
ingr.plot_morphology()
bake(soufflé)
