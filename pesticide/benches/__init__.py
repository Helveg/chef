from dataclasses import dataclass, field
import itertools as it
import mpipool
import typing
import arbor
from time import time
import nrnsub
import glia
import pickle
from ..trees import Tree
from mpi4py.MPI import COMM_WORLD as comm

@dataclass
class Bench:
    beds: list[typing.Any] = field()
    def run_benchmarks(self, name, reps=100):
        with mpipool.MPIExecutor() as p:
            p.workers_exit()
            s = 0
            r = 0
            results = []
            metas = []
            try:
                with open(f"bench_{name}.pkl", "rb") as f:
                    (s, r, results, metas) = pickle.load(f)
                    print("Starting from rep", s, "job", r)
            except FileNotFoundError:
                print("Starting fresh")
            for i in range(s, reps):
                print("Getting jobs")
                job_main = self.get_jobs()
                if i == s:
                    it.islice(job_main, r)
                for k, (res_arb, res_nrn, meta) in enumerate(p.map(lambda j: j(), job_main)):
                    if i == s:
                        k += r
                    res = (res_arb / res_nrn)
                    print("Rep", i, "Job", k, end="\r")
                    try:
                        results[k] += res / reps
                    except:
                        results.append(res / reps)
                        metas.append(meta)
                    with open(f"bench_{name}.pkl", "wb") as f:
                        pickle.dump((i, k, results, metas), f)
            for res, meta in zip(results, metas):
                with open(str(hash(meta)) + ".txt", "w") as f:
                    f.write(f"{res}\n{meta}")

    def get_jobs(self):
        job_iterator = it.chain.from_iterable(bed.jobs() for bed in self.beds)
        # Keep a stack of job queues so that we don't have to recurse when
        # `job_or_queue` is an iterator of jobs, which may again contain an iterator
        # of jobs, etc, recursion, you get it.
        job_queue_stack = []
        while True:
            job_or_queue = next(job_iterator, None)
            if job_or_queue is None:
                try:
                    job_iterator = job_queue_stack.pop()
                except IndexError:
                    break
                continue
            try:
                queue = iter(job_or_queue)
                job_queue_stack.append(job_iterator)
                job_iterator = queue
                continue
            except TypeError:
                job = job_or_queue
            yield job


class MorphoGen:
    def morphology(self):
        tree = arbor.segment_tree()
        tree.append(
            arbor.mnpos,
            arbor.mpoint(-2, 0, 0, 2),
            arbor.mpoint(2, 0, 0, 2),
            tag=1
        )
        return "4um_sphere", arbor.morphology(tree)

    def generate(self):
        yield self.morphology


class Seed(MorphoGen):
    def morphology(self):
        tree = arbor.segment_tree()
        tree.append(
            arbor.mnpos,
            arbor.mpoint(-2, 0, 0, 2),
            arbor.mpoint(2, 0, 0, 2),
            tag=1
        )
        return "Seed", arbor.morphology(tree)


class LabelGen:
    def label(self):
        return "empty", arbor.label_dict()

    def generate(self):
        yield self.label

class NoneAndAllLabelGen(LabelGen):
    def label(self):
        return "all or none", arbor.label_dict({
            "all": "(all)",
            "none": "(region-nil)"
        })


class DecorGen:
    def decor(self, mechs):
        return "empty", arbor.decor()

    def generate(self):
        yield self.decor


class AllDecorGen(DecorGen):
    def decor(self, mechs):
        decor = arbor.decor()
        for mech in mechs:
            decor.paint('"all"', mech)
        return "all", decor

    def generate(self):
        yield self.decor


@dataclass
class MechGen:
    catalogue_name: str

    def catalogue(self):
        return glia.catalogue(self.catalogue_name)

    def mechanisms(self, combination):
        return " ".join(combination), [arbor.mechanism(mech) for mech in combination]

    def generate(self):
        mechs = list(self.catalogue())
        combs = it.chain.from_iterable(
            it.combinations(mechs, r) for r in (0, 1, 2, 3, 4, len(mechs))
        )
        # Bind `comb` as a __default__ to the lambda for each iter.
        # Otherwise all lambdas use the last iteration of `comb`.
        yield from (lambda comb=comb: self.mechanisms(comb) for i, comb in enumerate(combs))

@dataclass
class TreeListGen:
    morpho_gen: MorphoGen
    label_gen: LabelGen
    decor_gen: DecorGen
    mech_gen: MechGen

    def make_tree(self, mo, la, de, me):
        mo, la, me = mo(), la(), me()
        de = de(me[1])
        name = "|".join(f"{n}={x[0]}" for n, x in zip(("morphology", "labels", "decor", "mechanisms"), (mo, la, de, me)))
        # print("Making tree:", "name=", name, "catalogue=", self.mech_gen.catalogue(), "morph", mo[1], "labels", la[1], "decor", de[1], me[1])
        return Tree(name, self.mech_gen.catalogue(), mo[1], la[1], de[1], me[1])

    def __call__(self):
        yield from (
            lambda a=a: [self.make_tree(*a)]
            for a in it.product(
                self.morpho_gen.generate(),
                self.label_gen.generate(),
                self.decor_gen.generate(),
                self.mech_gen.generate(),
            )
        )

@dataclass
class TreatmentGen:
    treatment_cls: type

    def __call__(self):
        yield self.treatment_cls

@dataclass
class Bed:
    trees_gen: TreeListGen
    treatment_gen: TreatmentGen

    def jobs(self):
        yield from it.starmap(
            Job,
            it.product(
                self.treatment_gen(),
                self.trees_gen()
            )
        )

class Job:
    def __init__(self, treatment_f, trees_f):
        self._treatment_f = treatment_f
        self._trees_f = trees_f

    def __call__(self):
        self.trees = self._trees_f()
        self.treatment = self._treatment_f(self.trees)
        t_arb = self.time_arb()
        t_nrn = self.time_nrn()
        del self.trees, self.treatment
        return t_arb, t_nrn, [t.name for t in self._trees_f()]

    def time_nrn(self):
        from neuron import h
        # Construct the object
        cells = [tree.neuron_cell() for tree in self.treatment._trees]
        h.dt = self.treatment.dt
        t = time()
        h.finitialize()
        while h.t < self.treatment.t:
            h.fadvance()
        return time() - t

    def time_arb(self):
        import arbor
        context = arbor.context()
        dd = arbor.partition_load_balance(self.treatment, context)
        sim = arbor.simulation(self.treatment, dd, context)
        t = time()
        sim.run(self.treatment.t, self.treatment.dt)
        return time() - t

def seed(cat_name: str) -> TreeListGen:
    return TreeListGen(MorphoGen(), NoneAndAllLabelGen(), AllDecorGen(), MechGen(cat_name))
