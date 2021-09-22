import arbor
from arbor import (
    cable_cell as arb_cable_cell,
    catalogue as arb_catalogue,
    decor as arb_decor,
    morphology as arb_morphology,
    label_dict as arb_label_dict,
    place_pwlin as arb_place_pwlin,
)
import numpy as np
import itertools
from dataclasses import dataclass, field

__version__ = "0.0.2"


class Fumigation(arbor.recipe):
    """
    Probe all the things.
    """
    def __init__(self, ingredients, v_init=-65, K=302.15):
        super().__init__()
        self._ingredients = ingredients
        self._props = arbor.neuron_cable_properties()
        self._props.set_property(Vm=v_init, tempK=K, rL=35.4, cm=0.01)
        self._props.set_ion(ion='na', int_con=10,   ext_con=140, rev_pot=50)
        self._props.set_ion(ion='k',  int_con=54.4, ext_con=2.5, rev_pot=-77)
        self._props.set_ion(ion='ca', int_con=0.00005, ext_con=2, rev_pot=132.5)
        self._props.set_ion(ion='h', valence=1, int_con=1.0, ext_con=1.0, rev_pot=-34)
        self._catalogue = cat = arbor.default_catalogue()
        for ingr in ingredients:
            cat.extend(ingr.catalogue, "")
        self._props.register(cat)

    def num_cells(self):
        return len(self._ingredients)

    def probes(self, gid):
        ingr = self._ingredients[gid]
        return [
            print(rm) or arbor.cable_probe_membrane_voltage(rm)
            for rm in ingr.get_region_midpoints()
        ]

    def num_sources(self, gid):
        return 1

    def cell_kind(self, gid):
        return arbor.cell_kind.cable

    def cell_description(self, gid):
        return self._ingredients[gid].cable_cell

    def global_properties(self, kind):
        return self._props


@dataclass
class Patient:
    """
    All information on a single cell model
    """
    name: str = field()
    catalogue: tuple[arb_catalogue, str] = field()
    morphology: arb_morphology = field()
    labels: arb_label_dict = field()
    decor: arb_decor = field()
    cable_cell: arb_cable_cell = None
    pwlin: arb_place_pwlin = None

    def __post_init__(self):
        self.decor.place("(root)", arbor.spike_detector(-10), "soma_spike_detector")
        self.cable_cell = arbor.cable_cell(self.morphology, self.labels, self.decor)
        self.pwlin = arbor.place_pwlin(self.morphology)

    def get_region_midpoints(self):
        return [f"(on-components 0.5 (region \"{r}\"))" for r in self.get_regions()]

    def get_painted_regions(self):
        return set(a[0][1:-1] for a, kw in self.decor.painted if len(a) > 1 and isinstance(a[1], arbor.mechanism))

    def get_regions(self):
        return set(self.labels)

    def plot_morphology(self):
        import plotly.graph_objs as go
        import plotly.io as pio

        pio.templates.default = "simple_white"
        regions = self.get_regions()

        fig = go.Figure()
        dims = ("x", "y", "z")
        origins = {k: float("+inf") for k in dims}
        range = float("-inf")
        for region, (x, y, z) in zip(
            regions, map(self.get_region_pw_xyz, regions)
        ):
            fig.add_trace(go.Scatter3d(x=x, y=y, z=z, name=region, marker_size=1))
            for k, d in zip(dims, (x, y, z)):
                _min = min(v for v in d if v is not None)
                _max = max(v for v in d if v is not None)
                range = max(abs(_max - _min), range)
                origins[k] = min(_min, origins[k])

        for k, o in origins.items():
            fig.layout.scene[f"{k}axis_range"] = [o, o + range]

        fig.show()


    def get_region_segments(self, region):
        return self.pwlin.segments(self.cable_cell.cables(region))

    def get_region_pw_xyz(self, region):
        segments = self.get_region_segments(region)
        x = [*itertools.chain(*((s.prox.x, s.dist.x, None) for s in segments))]
        y = [*itertools.chain(*((s.prox.y, s.dist.y, None) for s in segments))]
        z = [*itertools.chain(*((s.prox.z, s.dist.z, None) for s in segments))]
        return x, y, z




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
    for gid, ingr in enumerate(recipe._ingredients):
        Vm_probes = {
            f"Vm_{r}": sim.sample((gid, j), arbor.regular_schedule(0.1))
            for j, r in enumerate(ingr.get_regions())
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
