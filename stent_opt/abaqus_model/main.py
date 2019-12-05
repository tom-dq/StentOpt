
import typing

from stent_opt.abaqus_model import amplitude, base, step, load, instance, part, surface, element, output_requests


# Todo
#   - Sym
#   - Balloon



class AbaqusModel:
    name: str
    instances: typing.Dict[str, instance.Instance]
    steps: typing.List[step.StepBase]
    step_loads: typing.Set[typing.Tuple[step.StepBase, load.LoadBase]]

    _main_sep_line: str = "** -----------------------------"

    def __init__(self, name: str):
        self.name = name
        self.instances = dict()
        self.steps = list()
        self.step_loads = set()

    def add_instance(self, one_instance: instance.Instance):
        if one_instance.name in self.instances:
            raise ValueError(f"Already had an instance named {one_instance.name}")

        self.instances[one_instance.name] = one_instance

    def add_step(self, one_step: step.StepBase):
        self.steps.append(one_step)

    def add_load_starting_from(self, starting_step: step.StepBase, one_load: load.LoadBase):
        """Add a load at a step, and for all the following steps."""
        on_this_one = False
        for one_step in self.steps:
            on_this_one = on_this_one or one_step == starting_step
            if on_this_one:
                self.step_loads.add( (one_step, one_load))

        if not on_this_one:
            raise ValueError(f"Did not find {starting_step} in AbaqusModel.steps")

    def add_load_specific_steps(self, active_steps: typing.Iterable[step.StepBase], one_load: load.LoadBase):
        """Add a load but only at particular steps (so you can turn it off after a while)."""
        for one_step in active_steps:
            self.step_loads.add((one_step, one_load))

            if one_step not in self.steps:
                raise ValueError(f"Did not find {one_step} in AbaqusModel.steps")

    def get_parts(self) -> typing.Iterable[ part.Part]:
        """Iterate through the parts referenced by any instance in the model."""
        seen = set()
        for one_instance in self.instances.values():
            maybe_part = one_instance.base_part
            if maybe_part not in seen:
                yield maybe_part

            seen.add(maybe_part)

    def get_only_instance(self) -> instance.Instance:

        if len(self.instances) != 1:
            raise ValueError(f"Expected a single instance, got {len(self.instances)}.")

        return list(self.instances.values())[0]


    def produce_inp_lines(self) -> typing.Iterable[str]:
        yield from self._produce_inp_lines_header()
        yield from base.inp_heading("PARTS")
        for part in self.get_parts():
            yield from part.produce_inp_lines()

        yield from self._produce_inp_lines_assembly()
        yield from self._produce_inp_lines_amplitude()
        yield from self._produce_inp_lines_material()
        yield from self._produce_inp_lines_steps()


    def _produce_inp_lines_header(self) -> typing.Iterable[str]:
        yield "*Heading"
        yield f"** Job name: Job-1 Model name: {self.name}"
        yield "** Generated by: Abaqus/CAE 2016"
        yield "*Preprint, echo=NO, model=NO, history=NO, contact=NO"

    def _produce_inp_lines_assembly(self) -> typing.Iterable[str]:
        yield from base.inp_heading("ASSEMBLY")
        yield f"*Assembly, name=Assembly"
        yield "**"
        for instance_name, one_instance in self.instances.items():
            yield from one_instance.make_inp_lines()

        yield "**"

        for one_instance in self.instances.values():
            yield from one_instance.produce_equation_inp_line()

        yield "*End Assembly"


    def _produce_inp_lines_amplitude(self) -> typing.Iterable[str]:
        def generate_referenced_amplitudes():
            seen = set()
            seen.add(None)  # If there's no amplitude, it will be a None.
            all_loads = self._get_sorted_loads()
            for one_load in all_loads:
                if one_load.amplitude not in seen:
                    yield one_load.amplitude
                    seen.add(one_load.amplitude)

        for amp in generate_referenced_amplitudes():
            yield from amp.make_inp_lines()


    def _produce_inp_lines_material(self) -> typing.Iterable[str]:
        yield from base.inp_heading("MATERIALS")
        for one_part in self.get_parts():
            yield from one_part.common_material.produce_inp_lines()

    def _get_sorted_loads(self):
        all_loads = set(one_load for _, one_load in self.step_loads)

        def sort_key(one_load: load.LoadBase):
            return one_load.sortable()

        return sorted(all_loads, key=sort_key)


    def _produce_inp_lines_steps(self) ->  typing.Iterable[str]:
        yield self._main_sep_line
        for idx_step, one_step in enumerate(self.steps):

            yield from base.inp_heading(f"STEP: {one_step.name}")
            yield from one_step.produce_inp_lines()

            is_first_step = idx_step == 0
            if is_first_step:
                yield "** Mass Scaling: Semi-Automatic"
                yield "**               Whole Model"
                yield "*Variable Mass Scaling, dt=1e-05, type=set equal dt, frequency=1"

            yield from base.inp_heading("LOADS")

            for one_load in self._get_sorted_loads():
                all_load_events = self._step_load_actions(one_load)
                relevant_load_events = [action for a_step, action in all_load_events if a_step == one_step]
                if len(relevant_load_events) == 0:
                    pass

                elif len(relevant_load_events) == 1:
                    action = relevant_load_events.pop()
                    yield from one_load.produce_inp_lines(action)

                else:
                    raise ValueError(f"Got more than one thing to do with {one_load} and {one_step}... {relevant_load_events}")

            yield from output_requests.produce_inp_lines(output_requests.general_components)

            # Have to end the step here, after the loads have been output.

            # This outputs stuff we can read without starting CAE
            yield "*FILE OUTPUT, NUMBER INTERVAL=1"
            yield "*EL FILE"
            yield "S"

            # abaqus.bat job=job-12 interactive     >>> Run
            # abaqus.bat job=job-12 convert=select  >>> Make .fil from .sel
            # abaqus.bat ascfil job=job-12          >>> Make asciii .fin from .fil


            yield "*End Step"

    def _step_load_actions(self, one_load: load.LoadBase):
        """Get the load action for the steps (turn on, turn off, etc)."""

        active_at_last_step = False
        for idx, one_step in enumerate(self.steps):
            active_at_this_step = (one_step, one_load) in self.step_loads
            is_first_step = idx == 0

            def get_event_this_combination():
                if active_at_this_step and is_first_step:
                    return load.Action.create_first_step

                elif active_at_this_step and not active_at_last_step:
                    return load.Action.create_subsequent_step

                elif not active_at_this_step and active_at_last_step:
                    return load.Action.remove

            this_action = get_event_this_combination()
            if this_action:
                yield (one_step, this_action)

            active_at_last_step = active_at_this_step


def make_test_model() -> AbaqusModel:

    # one_part = part.make_part_test()
    one_instance = instance.make_instance_test()

    model = AbaqusModel("TestModel")
    model.add_instance(one_instance)

    all_steps = [step.make_test_step(x) for x in (1, 2, 3)]
    for one_step in all_steps:
        model.add_step(one_step)

    # Make a surface
    one_elem_set = one_instance.base_part.element_sets["OneElem"]
    surf_data = [ (one_elem_set, surface.SurfaceFace.S2) , ]
    one_surface = surface.Surface(name="TestSurface", sets_and_faces=surf_data)
    one_instance.add_surface(one_surface)


    # Add the loads
    one_amplitude = amplitude.make_test_amplitude()
    some_node_set = one_instance.base_part.get_everything_set(base.SetType.node)
    load_point = load.make_test_load_point(some_node_set)
    load_dist = load.make_test_pressure(one_surface, one_amplitude)
    model.add_load_starting_from(all_steps[0], load_point)
    model.add_load_specific_steps(all_steps[0:2], load_dist)

    return model

if __name__ == "__main__":
    model = make_test_model()
    with open(r"C:\temp\aba_out.inp", "w") as fOut:
        for l in model.produce_inp_lines():
            print(l)
            fOut.write(l + "\n")


# Export








