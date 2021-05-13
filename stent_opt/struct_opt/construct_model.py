import collections
import statistics
import typing

from stent_opt.abaqus_model import main, material, section, part, instance, node, element, surface, amplitude, \
    boundary_condition, load, step, interaction_property, interaction, base
from stent_opt.struct_opt import optimisation_parameters, design
from stent_opt.struct_opt.design import StentDesign, GlobalPartNames, GlobalNodeSetNames, Actuation, StentParams, \
    GlobalSurfNames

from stent_opt.struct_opt import patch_manager


def make_a_stent(optim_params: optimisation_parameters.OptimParams, stent_design: StentDesign, full_model: bool, sub_model_infos: typing.Iterable[patch_manager.SubModelInfoBase]):

    model = main.AbaqusModel("StentModel")

    element_dimensions = stent_design.stent_params.stent_element_dimensions
    node_num_idx_pos = design.generate_nodes(stent_design.stent_params)
    node_pos = {iNode: xyz for iNode, _, xyz in node_num_idx_pos}
    model.abaqus_output_time_interval = optim_params.abaqus_output_time_interval
    model.abaqus_target_increment = optim_params.abaqus_target_increment

    # Potentially modify the parameters if it's 2D
    if element_dimensions == 3:
        stent_params = stent_design.stent_params
        section_thickness = None
        transform_to_cyl = True

    elif element_dimensions == 2:
        sp_divs = stent_design.stent_params.divs.copy_with_updates(R=1)
        stent_params = stent_design.stent_params.copy_with_updates(divs=sp_divs)

        section_thickness = stent_params.radial_thickness
        transform_to_cyl = False

    else:
        raise ValueError(stent_design.stent_params.stent_element_dimensions)

    def make_stent_part():

        stress_strain_table = (
            material.Point(stress=205., strain=0.0),
            material.Point(stress=515., strain=0.6),
        )
        steel = material.MaterialElasticPlastic(
            name="Steel",
            density=7.85e-09,
            elast_mod=196000.0,
            elast_possion=0.3,
            plastic=stress_strain_table,
        )

        common_section = section.SolidSection(
            name="SolidSteel",
            mat=steel,
            thickness=section_thickness,
            enhanced_hourglass=section.HourglassControlEnhanced(name="EC-1"),
        )

        stent_part = part.Part(
            name=GlobalPartNames.STENT,
            common_section=common_section,
            transform_to_cyl=transform_to_cyl,
        )

        # Node offsets
        offset_single = stent_design.stent_params.get_span_of_patch_connectivity_buffered(optim_params.patch_hops).to_xyz()
        n_offset_rows = int(1.3 * len(sub_model_infos)**0.5)

        # Make the nodes.


        for sub_mod_idx, sub_model_info in enumerate(sub_model_infos):
            if optim_params.offset_submodels:
                off_row, off_col = divmod(sub_mod_idx, n_offset_rows)

            else:
                off_row, off_col = 0, 0

            this_offset = base.XYZ(off_col * offset_single.x, off_row*offset_single.y, 0.0)
            for iNode, one_node_polar in node_pos.items():
                stent_part.add_node_validated(iNode,this_offset + one_node_polar.to_xyz(), node_elem_offset=sub_model_info.node_elem_offset)

        # Make the elements.
        for sub_model_info in sub_model_infos:
            for idx, elem_num, one_elem in design.generate_stent_part_elements(stent_params):
                if idx in stent_design.active_elements and sub_model_info.elem_in_submodel(elem_num):
                    stent_part.add_element_validate(elem_num, one_elem, node_elem_offset=sub_model_info.node_elem_offset)

        # Remove any nodes which are not used
        stent_part.squeeze_unused_nodes()

        one_instance = instance.Instance(base_part=stent_part)

        if element_dimensions == 2:
            # Create the node sets on the boundary
            used_node_nums = stent_part.get_used_node_nums()
            iNode_to_idx_active = {iNode: idx for iNode, idx, _ in node_num_idx_pos if iNode in used_node_nums}

            if full_model:
                # Z may not span the whole thing...
                min_idx_z = min(idx.Z for idx in iNode_to_idx_active.values())

                def get_boundary_node_set(node_idx: design.PolarIndex):
                    # TODO - make "is_in_bottom_chunk" generalisable and more robust.
                    vert_distant = (node_idx.Z - min_idx_z) * stent_params.single_element_z_span
                    is_in_bottom_chunk = vert_distant < 0.5

                    if node_idx.Th == 0 and is_in_bottom_chunk: yield design.GlobalNodeSetNames.PlanarStentTheta0
                    if node_idx.Th == stent_params.divs.Th-1 and is_in_bottom_chunk: yield design.GlobalNodeSetNames.PlanarStentThetaMax
                    if node_idx.Z == min_idx_z: yield design.GlobalNodeSetNames.PlanarStentZMin

                boundary_set_name_to_nodes = collections.defaultdict(set)
                for iNode, idx in iNode_to_idx_active.items():
                    for node_set in get_boundary_node_set(idx):
                        boundary_set_name_to_nodes[node_set.name].add(iNode)

                for node_set_name, nodes in boundary_set_name_to_nodes.items():
                    one_node_set = node.NodeSet(stent_part, node_set_name, frozenset(nodes))
                    stent_part.add_node_set(one_node_set)

        model.add_instance(one_instance)

    def make_balloon_part():
        # Material properties are from Dylan's model.
        rubber = material.MaterialElasticPlastic(
            name="Rubber",
            density=1.1E-009,
            elast_mod=920.0,
            elast_possion=0.4,
            plastic=None,
        )

        common_section_balloon = section.MembraneSection(
            name="RubberMembrane",
            mat=rubber,
            thickness=0.02,
            enhanced_hourglass=None,
        )

        balloon_part = part.Part(
            name=GlobalPartNames.BALLOON,
            common_section=common_section_balloon,
            transform_to_cyl=True,
        )

        # Make the nodes and keep track of the leading/trailing edges.
        nodes_polar = {}
        boundary_set_name_to_nodes = collections.defaultdict(set)
        for iNode, (n_p, boundary_sets) in enumerate(design.generate_nodes_balloon_polar(stent_params=stent_params), start=1):
            nodes_polar[iNode] = n_p
            for boundary_set in boundary_sets:
                boundary_set_name_to_nodes[boundary_set.name].add(iNode)

        for node_set_name, nodes in boundary_set_name_to_nodes.items():
            one_node_set = node.NodeSet(balloon_part, node_set_name, frozenset(nodes))
            balloon_part.add_node_set(one_node_set)

        elems_all = {iElem: e for idx, iElem, e in design.generate_plate_elements_all(divs=stent_params.balloon.divs, elem_type=element.ElemType.M3D4R)}

        for iNode, one_node_polar in nodes_polar.items():
            balloon_part.add_node_validated(iNode, one_node_polar.to_xyz())

        for iElem, one_elem in elems_all.items():
            balloon_part.add_element_validate(iElem, one_elem)

        one_instance = instance.Instance(base_part=balloon_part)
        model.add_instance(one_instance)

    def make_cyl_part():
        cyl_mat = material.MaterialElasticPlastic(
            name="Cyl",
            density=1.1E-007,
            elast_mod=1e3,
            elast_possion=0.4,
            plastic=None,
        )

        common_section_cyl = section.SurfaceSection(
            name="CylSurf",
            mat=None,
            surf_density=1e-3,
            enhanced_hourglass=None,
        )

        cyl_inner_part = part.Part(
            name=GlobalPartNames.CYL_INNER,
            common_section=common_section_cyl,
            transform_to_cyl=True,
        )

        nodes_polar = design.generate_nodes_inner_cyl(stent_params=stent_params)

        elems_all = {iElem: e for idx, iElem, e in design.generate_plate_elements_all(divs=stent_params.cylinder.divs, elem_type=element.ElemType.SFM3D4R)}

        for iNode, one_node_polar in nodes_polar.items():
            cyl_inner_part.add_node_validated(iNode, one_node_polar.to_xyz())

        for iElem, one_elem in elems_all.items():
            cyl_inner_part.add_element_validate(iElem, one_elem)

        # All the nodes go in the boundary set.
        all_node_set = node.NodeSet(cyl_inner_part, GlobalNodeSetNames.RigidCyl.name, frozenset(nodes_polar.keys()))
        cyl_inner_part.add_node_set(all_node_set)

        one_instance = instance.Instance(base_part=cyl_inner_part)

        model.add_instance(one_instance)

    make_stent_part()

    if stent_params.actuation == Actuation.balloon:
        make_balloon_part()

    elif stent_params.actuation == Actuation.rigid_cylinder:
        make_cyl_part()

    return model


def create_surfaces(stent_params: StentParams, model: main.AbaqusModel):
    """Sets up the standard surfaces on the model."""

    stent_instance = model.get_only_instance_base_part_name(GlobalPartNames.STENT)
    stent_part = stent_instance.base_part

    elem_indexes_all = {iElem: i for iElem, i in design.generate_elem_indices(divs=stent_params.divs)}
    elem_indexes = {iElem: i for iElem, i in elem_indexes_all.items() if iElem in stent_part.elements}

    def create_elem_surface(one_instance: instance.Instance, elem_nums, name, one_surf: surface.SurfaceFace):
        one_part = one_instance.base_part
        elem_set_name = f"ElemSet_{name}"
        the_elements_raw = {iElem: one_part.elements[iElem] for iElem in elem_nums}
        elem_set = element.ElementSet(one_part, elem_set_name, element.Elements(the_elements_raw))
        data = [
            (elem_set, one_surf)
        ]
        the_surface = surface.Surface(name, data)

        one_part.add_element_set(elem_set)
        one_instance.add_surface(the_surface)

    if stent_params.stent_element_dimensions == 2:
        # Just have to make the leading edge
        max_theta = max(i.Th for i in elem_indexes.values())
        max_theta_elems = {iElem for iElem, i in elem_indexes.items() if i.Th == max_theta}
        create_elem_surface(stent_instance, max_theta_elems, GlobalSurfNames.PLANAR_LEADING_EDGE, surface.SurfaceFace.S2)

    elif stent_params.stent_element_dimensions == 3:
        # Have to make a few surfaces

        def create_elem_surface_S1(elem_nums, name):
            create_elem_surface(stent_instance, elem_nums, name, surface.SurfaceFace.S1)

        # Get the minimum radius surface
        min_r = min(i.R for i in elem_indexes.values())
        min_r_elems = {iElem for iElem, i in elem_indexes.items() if i.R == min_r}
        create_elem_surface_S1(min_r_elems, GlobalSurfNames.INNER_MIN_RADIUS)

        # Minimum R at any point
        all_elems_through_thickness = collections.defaultdict(set)
        for iElem, i in elem_indexes.items():
            key = (i.Th, i.Z)
            all_elems_through_thickness[key].add( (i.R, iElem) )

        min_elem_through_thickness = set()
        for _, elems in all_elems_through_thickness.items():
            min_r_and_elem = min(elems)
            min_r, min_elem = min_r_and_elem[:]
            min_elem_through_thickness.add(min_elem)

        create_elem_surface_S1(min_elem_through_thickness, GlobalSurfNames.INNER_SURFACE)

        # Minimum R, as long as the R in on the inner half.
        all_r = [i.R for i in elem_indexes.values()]
        median_r = statistics.median(all_r)
        min_bottom_half = set()
        for _, elems in all_elems_through_thickness.items():
            min_r_and_elem = min(elems)
            min_r, min_elem = min_r_and_elem[:]
            if min_r <= median_r:
                min_bottom_half.add(min_elem)

        create_elem_surface_S1(min_bottom_half, GlobalSurfNames.INNER_BOTTOM_HALF)

        if stent_params.actuation == Actuation.balloon:
            instance_balloon = model.get_only_instance_base_part_name(GlobalPartNames.BALLOON)
            all_balloon_elems = instance_balloon.base_part.elements.keys()
            create_elem_surface(instance_balloon, all_balloon_elems, GlobalSurfNames.BALLOON_INNER, surface.SurfaceFace.SNEG)

        elif stent_params.actuation == Actuation.rigid_cylinder:
            instance_cyl = model.get_only_instance_base_part_name(GlobalPartNames.CYL_INNER)
            all_cyl_elements = instance_cyl.base_part.elements.keys()
            create_elem_surface(instance_cyl, all_cyl_elements, GlobalSurfNames.CYL_INNER, surface.SurfaceFace.SNEG)


def apply_loads(
        optim_params: optimisation_parameters.OptimParams,
        stent_params: StentParams,
        full_model: bool,
        sub_model_infos: typing.Iterable[patch_manager.SubModelInfoBase],
        model: main.AbaqusModel
):
    _create_steps(optim_params, model)
    if full_model:
        _apply_loads_full(optim_params, stent_params, model)

    else:
        _apply_boundary_conds_submodel(sub_model_infos, model)




def _apply_boundary_conds_submodel(sub_model_infos: typing.Iterable[patch_manager.SubModelInfoBase], model: main.AbaqusModel):
    stent_instance = model.get_only_instance_base_part_name(GlobalPartNames.STENT)
    stent_part = stent_instance.base_part
    step1 = model.steps[0]

    used_node_nums = stent_part.get_used_node_nums()

    for sub_model_info in sub_model_infos:
        for node_num, dof_to_amp in sub_model_info.boundary_node_enforced_displacements():
            node_num_offset = node_num + sub_model_info.node_elem_offset
            if node_num_offset in used_node_nums:
                one_bound_node_set = node.NodeSet(stent_part, f"Nodeset-N{node_num_offset}", frozenset({node_num_offset}))
                stent_part.add_node_set(one_bound_node_set)

                for dof, amp in dof_to_amp.items():
                    patch_bound_disp = boundary_condition.BoundaryDispRot(
                        name=f"Bound-{node_num_offset}-{dof}",
                        with_amplitude=amp,
                        components=(
                            boundary_condition.DispRotBoundComponent(node_set=one_bound_node_set, dof=dof, value=1.0),
                        ),
                    )

                    model.add_load_starting_from(step1, patch_bound_disp)


def _apply_loads_full(optim_params: optimisation_parameters.OptimParams, stent_params: StentParams, model: main.AbaqusModel):
    if stent_params.actuation == Actuation.rigid_cylinder:
        _apply_loads_enforced_disp_rigid_cyl(optim_params, stent_params, model)

    elif stent_params.actuation == Actuation.enforced_displacement_plane:
        _apply_loads_enforced_disp_2d_planar(optim_params, stent_params, model)

    else:
        _apply_loads_pressure(optim_params, stent_params, model)


def _create_steps(optim_params: optimisation_parameters.OptimParams, model: main.AbaqusModel):
    step_expand = optim_params.analysis_step_type(
        name=f"ExpandHold",
        step_time=optim_params.time_expansion,
    )
    model.add_step(step_expand)

    if optim_params.simulation_has_second_step:
        if optim_params.post_expansion_behaviour == optimisation_parameters.PostExpansionBehaviour.release:
            step2_name = "Release"

        elif optim_params.post_expansion_behaviour == optimisation_parameters.PostExpansionBehaviour.oscillate:
            step2_name = "Oscillate"

        else:
            raise ValueError(optim_params.post_expansion_behaviour)

        step_two = optim_params.analysis_step_type(
            name=step2_name,
            step_time=optim_params.time_released,
        )
        model.add_step(step_two)


def _apply_loads_enforced_disp_2d_planar(optim_params: optimisation_parameters.OptimParams, stent_params: StentParams, model: main.AbaqusModel):

    # Maximum displacement
    max_displacement = stent_params.theta_arc_initial * (stent_params.expansion_ratio - 1.0)

    amp_data = (
        amplitude.XY(0.0, 0.0),
        amplitude.XY(0.8*optim_params.time_expansion, 1.0),
        amplitude.XY(optim_params.time_expansion, 1.0),
    )
    amp = amplitude.Amplitude("Amp-1", amp_data)

    stent_instance = model.get_only_instance_base_part_name(GlobalPartNames.STENT)
    stent_part = stent_instance.base_part
    expand_disp = boundary_condition.BoundaryDispRot(
        name="ExpandDisp",
        with_amplitude=amp,
        components=(
            boundary_condition.DispRotBoundComponent(node_set=stent_part.node_sets[GlobalNodeSetNames.PlanarStentTheta0.name], dof=1, value=0.0),
            boundary_condition.DispRotBoundComponent(node_set=stent_part.node_sets[GlobalNodeSetNames.PlanarStentThetaMax.name], dof=1, value=max_displacement),
        ),
    )

    if optim_params.post_expansion_behaviour == optimisation_parameters.PostExpansionBehaviour.release:
        step_two_disp = boundary_condition.BoundaryDispRot(
            name="ReleaseDisp",
            with_amplitude=None,
            components=(
                boundary_condition.DispRotBoundComponent(node_set=stent_part.node_sets[GlobalNodeSetNames.PlanarStentTheta0.name], dof=1, value=0.0),
            ),
        )

    elif optim_params.post_expansion_behaviour == optimisation_parameters.PostExpansionBehaviour.oscillate:
        HEART_RATE_HZ = 1.2
        HEART_RATE_TESTING = 20.0
        OSC_AMP = 0.05  # For now, just 5% of the initial arc length

        amp_osc = amplitude.AmplitudePeriodic("Amp-Osc", HEART_RATE_TESTING, 0.0, 1.0, OSC_AMP)
        step_two_disp = boundary_condition.BoundaryDispRot(
            name="OscillateDisp",
            with_amplitude=amp_osc,
            components=(
                boundary_condition.DispRotBoundComponent(
                    node_set=stent_part.node_sets[GlobalNodeSetNames.PlanarStentTheta0.name], dof=1, value=0.0),
                boundary_condition.DispRotBoundComponent(
                    node_set=stent_part.node_sets[GlobalNodeSetNames.PlanarStentThetaMax.name], dof=1,
                    value=max_displacement),
            ),
        )

    hold_base1 = boundary_condition.BoundaryDispRot(
        name="HoldBaseA",
        with_amplitude=None,
        components=(
            boundary_condition.DispRotBoundComponent(node_set=stent_part.node_sets[GlobalNodeSetNames.PlanarStentZMin.name], dof=2, value=0.0),
        ),
    )

    hold_base2 = boundary_condition.BoundaryDispRot(
        name="HoldBaseB",
        with_amplitude=None,
        components=(
            boundary_condition.DispRotBoundComponent(node_set=stent_part.node_sets[GlobalNodeSetNames.PlanarStentZMin.name], dof=2, value=0.0),
        ),
    )


    model.add_load_specific_steps([model.steps[0]], expand_disp)
    if optim_params.simulation_has_second_step: model.add_load_specific_steps([model.steps[1]], step_two_disp)
    model.add_load_specific_steps([model.steps[0]], hold_base1)
    if optim_params.simulation_has_second_step: model.add_load_specific_steps([model.steps[1]], hold_base2)

    if optim_params.release_stent_after_expansion:
        # Rebound pressure (kind of like the blood vessel squeezing in).
        leading_edge = stent_instance.surfaces[GlobalSurfNames.PLANAR_LEADING_EDGE]
        actuated_line_length = leading_edge.num_elem() * stent_params.single_element_z_span
        pressure_load = 0.2 / actuated_line_length

        amp_data = (
            amplitude.XY(0.0, 0.0),
            amplitude.XY(0.1 * optim_params.time_released, 0.0),
            amplitude.XY(0.6 * optim_params.time_released, pressure_load),
            amplitude.XY(optim_params.time_released, pressure_load),
        )

        amp_pressure = amplitude.Amplitude("Amp-Press", amp_data)

        inner_pressure = load.PressureLoad(
            name="Pressure",
            with_amplitude=amp_pressure,
            on_surface=leading_edge,
            value=pressure_load,
        )

        model.add_load_specific_steps([model.steps[1]], inner_pressure)


def _apply_loads_enforced_disp_rigid_cyl(optim_params: optimisation_parameters.OptimParams, stent_params: StentParams, model: main.AbaqusModel):
    init_radius = stent_params.actuation_surface_ratio
    final_radius = stent_params.r_min * stent_params.expansion_ratio
    dr = final_radius - init_radius


    one_step = step.StepDynamicExplicit(
        name=f"Expand",
        step_time=optim_params.time_expansion,
        bulk_visc_b1=step.FALLBACK_VISC_B1,
        bulk_visc_b2=step.FALLBACK_VISC_B2,
    )

    model.add_step(one_step)

    amp_data = (
        amplitude.XY(0.0, 0.0),
        amplitude.XY(0.75 * optim_params.time_expansion, dr),
        amplitude.XY(optim_params.time_expansion, 0),
    )

    amp = amplitude.Amplitude("Amp-1", amp_data)

    cyl_inst = model.get_only_instance_base_part_name(GlobalPartNames.CYL_INNER)
    node_set = cyl_inst.base_part.node_sets[GlobalNodeSetNames.RigidCyl.name]

    enf_disp = boundary_condition.BoundaryDispRot(
        name="ExpRad",
        with_amplitude=amp,
        components=(
            boundary_condition.DispRotBoundComponent(node_set=node_set, dof=1, value=1.0),
            boundary_condition.DispRotBoundComponent(node_set=node_set, dof=2, value=0.0),
            boundary_condition.DispRotBoundComponent(node_set=node_set, dof=3, value=0.0),
        ),
    )

    model.add_load_specific_steps([one_step], enf_disp)


def _apply_loads_pressure(optim_params: optimisation_parameters.OptimParams, stent_params: StentParams, model: main.AbaqusModel):
    # Some nominal amplitude from Dylan's model.

    if stent_params.balloon:
        max_pressure = 2.0
        surf = model.get_only_instance_base_part_name(GlobalPartNames.BALLOON).surfaces[GlobalSurfNames.BALLOON_INNER]

    else:
        max_pressure = 0.2
        surf = model.get_only_instance_base_part_name(GlobalPartNames.STENT).surfaces[GlobalSurfNames.INNER_BOTTOM_HALF]

    one_step = step.StepDynamicExplicit(
        name=f"Expand",
        step_time=optim_params.time_expansion,
        bulk_visc_b1=step.FALLBACK_VISC_B1,
        bulk_visc_b2=step.FALLBACK_VISC_B2,
    )

    model.add_step(one_step)

    amp_data = (
        amplitude.XY(0.0, 0.0),
        amplitude.XY(0.65*optim_params.time_expansion, 0.16*max_pressure),
        amplitude.XY(0.75*optim_params.time_expansion, 0.5*max_pressure),
        amplitude.XY(0.9*optim_params.time_expansion, max_pressure),
        amplitude.XY(optim_params.time_expansion, 0.05*max_pressure),
    )

    amp = amplitude.Amplitude("Amp-1", amp_data)

    one_load = load.PressureLoad(
        name="Pressure",
        with_amplitude=amp,
        on_surface=surf,
        value=1.0,
    )

    model.add_load_specific_steps([one_step], one_load)


def add_interaction(stent_params: StentParams, model: main.AbaqusModel):
    """Simple interaction between the balloon/cyl and the stent."""

    NO_INTERACTION = (Actuation.direct_pressure, Actuation.enforced_displacement_plane)
    if stent_params.actuation not in NO_INTERACTION:

        if stent_params.actuation == Actuation.balloon:
            inner_surf = model.get_only_instance_base_part_name(GlobalPartNames.BALLOON).surfaces[
                GlobalSurfNames.BALLOON_INNER]

        elif stent_params.actuation == Actuation.rigid_cylinder:
            inner_surf = model.get_only_instance_base_part_name(GlobalPartNames.CYL_INNER).surfaces[
                GlobalSurfNames.CYL_INNER]

        else:
            raise ValueError(stent_params.actuation)

        outer_surf = model.get_only_instance_base_part_name(GlobalPartNames.STENT).surfaces[
            GlobalSurfNames.INNER_SURFACE]


        int_prop = interaction_property.SurfaceInteraction(
            name="FrictionlessContact",
            lateral_friction=0.0,
        )

        one_int_general = interaction.GeneralContact(
            name="Interaction",
            int_property=int_prop,
            included_surface_pairs=(
                (inner_surf, outer_surf),
            )
        )

        one_int_specified = interaction.ContactPair(
            name="Interaction",
            int_property=int_prop,
            mechanical_constraint=interaction.MechanicalConstraint.Penalty,
            surface_pair=(inner_surf, outer_surf),
        )

        model.interactions.add(one_int_general)


def apply_boundaries(stent_params: StentParams, model: main.AbaqusModel):
    if stent_params.stent_element_dimensions == 2:
        pass

    elif stent_params.stent_element_dimensions == 3:
        _apply_boundaries_3d(stent_params, model)

    else:
        raise ValueError(stent_params.stent_element_dimensions)


def _apply_boundaries_3d(stent_params: StentParams, model: main.AbaqusModel):
    """Find the node pairs and couple them, and apply sym conditions."""

    # Stent has coupled boundaries
    stent_instance = model.get_only_instance_base_part_name(GlobalPartNames.STENT)
    stent_part = stent_instance.base_part

    # Only generate the equations on the active nodes.
    uncoupled_nodes_in_part = set()
    for elem in stent_part.elements.values():
        uncoupled_nodes_in_part.update(elem.connection)

    if not uncoupled_nodes_in_part:
        raise ValueError()

    for n1, n2 in design.gen_active_pairs(stent_params, uncoupled_nodes_in_part):
        stent_instance.add_node_couple(n1, n2, False)
        uncoupled_nodes_in_part.remove(n1)
        uncoupled_nodes_in_part.remove(n2)

    # Fix a single node in Theta and Z to stop rigid body rotation
    arbitrary_uncoupled_node = min(uncoupled_nodes_in_part)
    rigid_node_set = node.NodeSet(stent_part, "RigidRest", frozenset( [arbitrary_uncoupled_node] ))
    stent_part.add_node_set(rigid_node_set)
    bc_components = (
        boundary_condition.DispRotBoundComponent(node_set=rigid_node_set, dof=2, value=0.0),
        boundary_condition.DispRotBoundComponent(node_set=rigid_node_set, dof=3, value=0.0),
    )
    rigid_bc = boundary_condition.BoundaryDispRot(
        name="RigidRestraint",
        with_amplitude=None,
        components=bc_components,
    )
    model.boundary_conditions.add(rigid_bc)

    # Balloon has sym boundaries.
    sym_conds = {
        GlobalNodeSetNames.BalloonTheta0: boundary_condition.BoundaryType.YSYMM,    # YSYMM fixes DoF 2 (i.e., Theta)
        GlobalNodeSetNames.BalloonThetaMax: boundary_condition.BoundaryType.YSYMM,
        GlobalNodeSetNames.BalloonZ0: boundary_condition.BoundaryType.ZSYMM,        # ZSYMM fixes DoF 3 (i.e., Z)
        GlobalNodeSetNames.BalloonZMax: boundary_condition.BoundaryType.ZSYMM,
    }

    if stent_params.actuation == Actuation.balloon:
        balloon_part = model.get_only_instance_base_part_name(GlobalPartNames.BALLOON).base_part

        for sym_name, boundary_type in sym_conds.items():
            relevant_set = balloon_part.node_sets[sym_name.name]
            one_bc = boundary_condition.BoundarySymEncastre(
                name=f"{sym_name} - Sym",
                with_amplitude=None,
                boundary_type=boundary_type,
                node_set=relevant_set,
            )

            model.boundary_conditions.add(one_bc)


def write_model(model: main.AbaqusModel, fn_inp):
    with open(fn_inp, "w") as fOut:
        for l in model.produce_inp_lines():
            fOut.write(l + "\n")


def make_stent_model(
        optim_params: optimisation_parameters.OptimParams,
        stent_design: StentDesign,
        sub_model_infos: typing.Iterable[patch_manager.SubModelInfoBase],
        fn_inp: str):

    if all(sub_model_info.is_sub_model for sub_model_info in sub_model_infos):
        full_model = False

    elif all(sub_model_info.is_full_model for sub_model_info in sub_model_infos):
        full_model = True

    else:
        raise ValueError("Ambiguous")

    model = make_a_stent(optim_params, stent_design, full_model, sub_model_infos)
    if full_model:
        create_surfaces(stent_design.stent_params, model)
    apply_loads(optim_params, stent_design.stent_params, full_model, sub_model_infos, model)
    if full_model:
        add_interaction(stent_design.stent_params, model)
        apply_boundaries(stent_design.stent_params, model)

    print(fn_inp, f"Volume Ratio={stent_design.volume_ratio()}", sep='\t')
    write_model(model, fn_inp)