import arbor
import functools
import math
from dataclasses import dataclass, field
import itertools as it
import numpy as np

class NrnCell:
    def __init__(self):
        self.all = []

    def init_vm(self):
        self.recorders = [make_recorder(s) for s in self.all]

    def vm(self):
        return [np.array(list(s)) for s in self.recorders]

def _dims(segments):
    d = lambda s, d: (getattr(s.prox, d) - getattr(s.dist, d)) ** 2
    eucl = [sum(d(s, l) for l in ("x", "y", "z")) ** (1/2) for s in segments]
    radii = [(s.prox.radius + s.dist.radius) / 2 for s in segments]
    r = sum(r * e for r, e in zip(radii, eucl)) / sum(eucl) / len(segments)
    return sum(eucl), r * 2

def make_section(segments):
    import neuron
    sec = neuron.h.Section()
    sec.L, sec.diam = _dims(segments)
    return sec

def make_recorder(section):
    import neuron
    v = neuron.h.Vector()
    v.record(section(0.5)._ref_v)
    return v


@dataclass
class Tree:
    """
    Trees contain all information on a cable cell.
    """
    name: str = field()
    catalogue: tuple[arbor.catalogue, str] = field()
    morphology: arbor.morphology = field()
    labels: arbor.label_dict = field()
    decor: arbor.decor = field()
    mechanisms: list[arbor.mechanism] = field(default_factory=list)
    pwlin: arbor.place_pwlin = None

    def __post_init__(self):
        self.decor.place("(root)", arbor.spike_detector(-10), "soma_spike_detector")
        # self.pwlin = arbor.place_pwlin(self.morphology)

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
        return self.pwlin.segments(self.arbor_cell.cables(region))

    def get_region_pw_xyz(self, region):
        segments = self.get_region_segments(region)
        x = [*it.chain(*((s.prox.x, s.dist.x, None) for s in segments))]
        y = [*it.chain(*((s.prox.y, s.dist.y, None) for s in segments))]
        z = [*it.chain(*((s.prox.z, s.dist.z, None) for s in segments))]
        return x, y, z

    @functools.cached_property
    def arbor_cell(self):
        return arbor.cable_cell(self.morphology, self.labels, self.decor)

    def neuron_cell(self):
        import neuron, glia
        mechs = self.mechanisms
        templ = self.morphology
        cell = NrnCell()
        num = templ.num_branches
        cell.all = [
            make_section(templ.branch_segments(i))
            for i in range(num)
        ]
        for i, sec in enumerate(cell.all):
            p = templ.branch_parent(i)
            if p != 4294967295:
                sec.connect(cell.all[p])
        name = self.name
        # Hack to replace CV lists. Hardcode the `all` and `none` situations
        if "decor=all" in name:
            for mech in mechs:
                # mech is actually a `density` so pick up its mech
                mech = mech.mech
                for sec in cell.all:
                    gname = mech.name
                    var = None
                    tried = [(gname, var)]
                    parts = mech.name.split("_")
                    itr = iter(range(1, len(parts) + 1))
                    while (i := next(itr, None)) is not None:
                        try:
                            glia.insert(sec, gname, variant=var)
                            break
                        except Exception as e:
                            gname = "_".join(parts[:i])
                            var = "_".join(parts[i:])
                            tried.append((gname, var))
                    else:
                        raise Exception(f"No glia asset found for {mech.name} {tried}")
        cell.init_vm()
        return cell
