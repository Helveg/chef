import arbor
import numpy as np
import itertools
import functools
import types
import sys
from dataclasses import dataclass, field
import nrnsub
from time import time
from .trees import Tree

__version__ = "0.0.2"

try:
    import neuron
    def _nrn_available(default=lambda: None):
        def decorator(f):
            @functools.wraps(f)
            def w(*args, **kwargs):
                return f(*args, **kwargs)

            return w

        return decorator

    neuron.available = _nrn_available
    neuron.is_available = True
except ImportError:
    def _nrn_unavailable(default=lambda: None):
        def decorator(f):
            @functools.wraps(f)
            def w(*args, **kwargs):
                return default()

            return w

        return decorator

    sys.modules["neuron"] = neuron = types.ModuleType("neuron")
    neuron.available = _nrn_unavailable
    neuron.is_available = False


class SafetyLabel:
    def __init__(self, neuron_base=True, v_init=-65, K=302.15, rL=35.4, cm=0.01, **kwargs):
        if neuron_base:
            self._props = arbor.neuron_cable_properties()
        else:
            self._props = arbor.cable_global_properties()
        # self._props.set_property(Vm=v_init, tempK=K, rL=35.4, cm=0.01)
        grouped_by_ion = {}
        for k, v in kwargs.items():
            parts = k.split("_")
            ion = parts[0]
            prop = "_".join(parts[1:])
            ion_props = grouped_by_ion.setdefault(ion, dict())
            ion_props[prop] = v
        for ion, props in grouped_by_ion.items():
            self._props.set_ion(ion=ion, **props)

    @property
    def properties(self):
        return self._props

def default_props():
    return SafetyLabel(
        na_int_con=10.0, na_ext_con=140.0, na_rev_pot=50.0,
        k_int_con=54.4, k_ext_con=2.5, k_rev_pot=-77.0,
        ca_int_con=0.00005, ca_ext_con=2.0, ca_rev_pot=132.5,
        cal_int_con=0.00005, cal_ext_con=2.0, cal_rev_pot=132.5, cal_valence=2,
        h_valence=1.0, h_int_con=1.0, h_ext_con=1.0, h_rev_pot=-34.0,
    )


class Treatment:
    """
    Treatments contain all information on a simulation.
    """
    t = 10
    dt = 0.025

    def __repr__(self):
        return f"<{type(self).__name__} trees=[{', '.join(t.name for t in self._trees)}]>"


class Nursery(arbor.recipe, Treatment):
    """
    Nurse a tree. Ok, runs a single cell model :rolls_eyes:
    """
    def __init__(self, trees, cat=None, props=None):
        super().__init__()
        if len(trees) != 1:
            raise Exception(f"Nursery is a single tree treatment. {len(trees)} given.")
        self._trees = trees
        self._tree = trees[0]
        if props is None:
            props = default_props()
        if cat is None:
            cat = self._tree.catalogue
        self._props = props
        self._catalogue = cat
        self._props.properties.register(cat)

    def num_cells(self):
        return 1

    def probes(self, gid):
        return []
        [
            arbor.cable_probe_membrane_voltage(rm)
            for rm in tree.get_region_midpoints()
        ]

    def num_sources(self, gid):
        return 0

    def cell_kind(self, gid):
        return arbor.cell_kind.cable

    def cell_description(self, gid):
        return self._tree.arbor_cell

    def global_properties(self, kind):
        return self._props.properties


class Fumigation(arbor.recipe, Treatment):
    """
    Probe all the things.
    """
    def __init__(self, trees, props=None):
        super().__init__()
        if props is None:
            props = default_props()
        self._trees = trees
        self._props = props.properties
        self._catalogue = cat = arbor.default_catalogue()
        for tree in trees:
            cat.extend(tree.catalogue, "")
        self._props.register(cat)

    def num_cells(self):
        return len(self._trees)

    def probes(self, gid):
        tree = self._trees[gid]
        return [
            arbor.cable_probe_membrane_voltage(rm)
            for rm in tree.get_region_midpoints()
        ]

    def num_sources(self, gid):
        return 1

    def cell_kind(self, gid):
        return arbor.cell_kind.cable

    def cell_description(self, gid):
        return self._trees[gid].arbor_cell

    def global_properties(self, kind):
        return self._props


class DecorSpy(arbor.decor):
    def __init__(self):
        super().__init__()
        self.painted = []
        self.placed = []

    def paint(self, *args, **kwargs):
        super().paint(*args, **kwargs)
        self.painted.append((args, kwargs))

    def place(self, *args, **kwargs):
        super().place(*args, **kwargs)
        self.placed.append((args, kwargs))


def apply(recipe, duration=1000):
    context = arbor.context()
    domains = arbor.partition_load_balance(recipe, context)
    sim = arbor.simulation(recipe, domains, context)
    sim.record(arbor.spike_recording.all)
    handles = []
    for gid, tree in enumerate(recipe._trees):
        Vm_probes = {
            f"Vm_{r}": sim.sample((gid, j), arbor.regular_schedule(0.1))
            for j, r in enumerate(tree.get_regions())
        }
        handles.append(Vm_probes)

    sim.run(tfinal=duration)

    spikes = sim.spikes()
    import plotly.graph_objs as go
    for gid, probes in enumerate(handles):
        fig = go.Figure()
        for name, probe in probes.items():
            print("Probe", name)
            results = sim.samples(probe)
            if not results:
                print("No data for", name)
                continue
            data, meta = results[0]
            fig.add_scatter(x=data[:, 0], y=data[:, 1], name=name + " " + str(meta))
        fig.show()
    return data[:, 0], data[:, 1]


import arbor

# (1) Create a morphology with a single (cylindrical) segment of length=diameter=6 μm
tree = arbor.segment_tree()
tree.append(arbor.mnpos, arbor.mpoint(-3, 0, 0, 3), arbor.mpoint(3, 0, 0, 3), tag=1)

# (2) Define the soma and its midpoint
labels = arbor.label_dict({'soma':   '(tag 1)',
                           'midpoint': '(location 0 0.5)'})

# (3) Create cell and set properties
decor = arbor.decor()
decor.set_property(Vm=-40)
decor.paint('"soma"', 'hh')
decor.place('"midpoint"', arbor.iclamp( 10, 2, 0.8), "iclamp")
decor.place('"midpoint"', arbor.spike_detector(-10), "detector")
cell = arbor.cable_cell(tree, labels, decor)

tree = Tree("test", arbor.default_catalogue(), arbor.morphology(tree), labels, decor)

# (4) Define a recipe for a single cell and set of probes upon it.
# This constitutes the corresponding generic recipe version of
# `single_cell_model.py`.

class single_recipe(arbor.recipe, Treatment):
    def __init__(self, cells):
        # (4.1) The base C++ class constructor must be called first, to ensure that
        # all memory in the C++ class is initialized correctly.
        arbor.recipe.__init__(self)
        self.the_cell = cells[0]
        self._trees = cells
        self.the_probes = [arbor.cable_probe_membrane_voltage('"midpoint"')]
        self.the_probes = []
        self.the_props = arbor.neuron_cable_properties()
        self.the_cat = arbor.default_catalogue()
        self.the_props.register(self.the_cat)

    def num_cells(self):
        # (4.2) Override the num_cells method
        return 1

    def cell_kind(self, gid):
        # (4.3) Override the cell_kind method
        return arbor.cell_kind.cable

    def cell_description(self, gid):
        # (4.4) Override the cell_description method
        return self.the_cell.arbor_cell

    def probes(self, gid):
        # (4.5) Override the probes method
        return self.the_probes

    def global_properties(self, kind):
        # (4.6) Override the global_properties method
        return self.the_props

# (5) Instantiate recipe with a voltage probe located on "midpoint".

recipe = single_recipe([tree])
