from dataclasses import InitVar, dataclass, field
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
class Task:
    start: float = field(default_factory=time)
    end: float = field(default_factory=time)
    checkpoints: dict[str, float] = field(default_factory=dict)
    err: Exception = None

    def finish(self):
        self.end = time()

    def error(self, err):
        self.err = err
        self.end = float("nan")

    def checkpoint(self, name, start=False):
        if name in self.checkpoints:
            raise Exception(f"Duplicate checkpoint call {name}.")
        self.checkpoints[name] = t = time()
        if start:
            self.start = t

    def get(self, name):
        # If the task errored out, return `nan` for all checkpoints it
        # never hit.
        if self.err:
            return self.checkpoints.get(name, float("nan"))
        else:
            return self.checkpoints.get(name)

@dataclass
class Result:
    treatment: InitVar[typing.Any]
    job: Task
    arb: Task
    nrn: Task

    def __post_init__(self, treatment):
        self.name = repr(treatment)
        self.ramk = comm.Get_rank()

@dataclass
class Bench:
    beds: list[typing.Any] = field()
    def run_benchmarks(self, name, reps=100):
        with mpipool.MPIExecutor() as p:
            p.workers_exit()
            s = 0
            r = 0
            results = []
            times = []
            try:
                with open(f"bench_{name}.pkl", "rb") as f:
                    (s, r, results, times) = pickle.load(f)
                    print("Starting from rep", s, "job", r)
            except FileNotFoundError:
                print("Starting fresh")
            for i in range(s, reps):
                print("Getting jobs")
                job_main = self.get_jobs()
                if i == s:
                    it.islice(job_main, r)
                for k, result in enumerate(p.map(lambda j: j(), job_main)):
                    if i == s:
                        k += r
                    res_arb = result.arb.end - result.arb.get("sim-init")
                    res_nrn = result.nrn.end - result.nrn.get("sim-init")
                    res = res_nrn / res_arb
                    print("Rep", i, "Job", k, "Arbor", res_nrn / res_arb, "times faster.", end="\r")
                    try:
                        times[k] += res / reps
                        results[k].append(result)
                    except:
                        times.append(res / reps)
                        results.append([result])
                    if not k % 100:
                        with open(f"bench_{name}.pkl", "wb") as f:
                            pickle.dump((i, k, results, times), f)
            for res, avg_time in zip(results, times):
                with open(str(hash(res[0].name)) + ".txt", "w") as f:
                    f.write(f"{res[0].name}\n{avg_time}")

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
        job = Task()
        self.trees = trees = self._trees_f()
        job.checkpoint("trees-created")
        self.treatment = self._treatment_f(trees)
        job.checkpoint("treatment-created")
        arb_task = self.time_arb()
        job.checkpoint("arbor")
        nrn_task = self.time_nrn()
        result = Result(self.treatment, job, arb_task, nrn_task)
        job.checkpoint("neuron")
        job.finish()
        del self.trees, self.treatment
        return result

    def time_nrn(self):
        task = Task()
        try:
            from neuron import h
            # Construct the object
            cells = [tree.neuron_cell() for tree in self.treatment._trees]
            h.dt = self.treatment.dt
            h.finitialize()
            task.checkpoint("sim-init")
            while h.t < self.treatment.t:
                h.fadvance()
            task.finish()
        except Exception as e:
            task.error(e)
        return task

    def time_arb(self):
        task = Task()
        try:
            import arbor
            context = arbor.context()
            dd = arbor.partition_load_balance(self.treatment, context)
            sim = arbor.simulation(self.treatment, dd, context)
            task.checkpoint("sim-init")
            sim.run(self.treatment.t, self.treatment.dt)
            task.finish()
        except Exception as e:
            task.error(e)
        return task

def seed(cat_name: str) -> TreeListGen:
    return TreeListGen(MorphoGen(), NoneAndAllLabelGen(), AllDecorGen(), MechGen(cat_name))
