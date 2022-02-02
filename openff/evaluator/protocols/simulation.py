"""
A collection of base classes for protocols for running molecular simulations.
These should be subclassed when defining protocols for specific packages, such
as OpenMM or Gromacs.
"""
import abc

from openff.units import unit

from openff.evaluator.attributes import UNDEFINED
from openff.evaluator.forcefield.system import ParameterizedSystem
from openff.evaluator.thermodynamics import Ensemble, ThermodynamicState
from openff.evaluator.utils.observables import ObservableFrame
from openff.evaluator.workflow import Protocol
from openff.evaluator.workflow.attributes import (
    InequalityMergeBehaviour,
    InputAttribute,
    OutputAttribute,
)


class BaseEnergyMinimisation(Protocol, abc.ABC):
    """A base class for protocols which will minimise the potential
    energy of a given system.
    """

    input_coordinate_file = InputAttribute(
        docstring="The coordinates to minimise.", type_hint=str, default_value=UNDEFINED
    )
    parameterized_system = InputAttribute(
        docstring="The parameterized system object which encodes the systems potential "
        "energy function.",
        type_hint=ParameterizedSystem,
        default_value=UNDEFINED,
    )

    tolerance = InputAttribute(
        docstring="The energy tolerance to which the system should be minimized.",
        type_hint=unit.Quantity,
        default_value=10 * unit.kilojoules / unit.mole,
    )
    max_iterations = InputAttribute(
        docstring="The maximum number of iterations to perform. If this is 0, "
        "minimization is continued until the results converge without regard to "
        "how many iterations it takes.",
        type_hint=int,
        default_value=0,
    )

    enable_pbc = InputAttribute(
        docstring="If true, periodic boundary conditions will be enabled.",
        type_hint=bool,
        default_value=True,
    )

    output_coordinate_file = OutputAttribute(
        docstring="The file path to the minimised coordinates.", type_hint=str
    )


class BaseSimulation(Protocol, abc.ABC):
    """A base class for protocols which will perform a molecular
    simulation in a given ensemble and at a specified state.
    """

    steps_per_iteration = InputAttribute(
        docstring="The number of steps to propogate the system by at "
        "each iteration. The total number of steps performed "
        "by this protocol will be `total_number_of_iterations * "
        "steps_per_iteration`.",
        type_hint=int,
        merge_behavior=InequalityMergeBehaviour.LargestValue,
        default_value=1000000,
    )
    total_number_of_iterations = InputAttribute(
        docstring="The number of times to propogate the system forward by the "
        "`steps_per_iteration` number of steps. The total number of "
        "steps performed by this protocol will be `total_number_of_iterations * "
        "steps_per_iteration`.",
        type_hint=int,
        merge_behavior=InequalityMergeBehaviour.LargestValue,
        default_value=1,
    )

    output_frequency = InputAttribute(
        docstring="The frequency (in number of steps) with which to write to the "
        "output statistics and trajectory files.",
        type_hint=int,
        merge_behavior=InequalityMergeBehaviour.SmallestValue,
        default_value=3000,
    )
    checkpoint_frequency = InputAttribute(
        docstring="The frequency (in multiples of `output_frequency`) with which to "
        "write to a checkpoint file, e.g. if `output_frequency=100` and "
        "`checkpoint_frequency==2`, a checkpoint file would be saved every "
        "200 steps.",
        type_hint=int,
        merge_behavior=InequalityMergeBehaviour.SmallestValue,
        optional=True,
        default_value=10,
    )

    timestep = InputAttribute(
        docstring="The timestep to evolve the system by at each step.",
        type_hint=unit.Quantity,
        merge_behavior=InequalityMergeBehaviour.SmallestValue,
        default_value=2.0 * unit.femtosecond,
    )

    thermodynamic_state = InputAttribute(
        docstring="The thermodynamic conditions to simulate under",
        type_hint=ThermodynamicState,
        default_value=UNDEFINED,
    )
    ensemble = InputAttribute(
        docstring="The thermodynamic ensemble to simulate in.",
        type_hint=Ensemble,
        default_value=Ensemble.NPT,
    )

    thermostat_friction = InputAttribute(
        docstring="The thermostat friction coefficient.",
        type_hint=unit.Quantity,
        merge_behavior=InequalityMergeBehaviour.SmallestValue,
        default_value=1.0 / unit.picoseconds,
    )

    input_coordinate_file = InputAttribute(
        docstring="The file path to the starting coordinates.",
        type_hint=str,
        default_value=UNDEFINED,
    )
    parameterized_system = InputAttribute(
        docstring="The parameterized system object which encodes the systems potential "
        "energy function.",
        type_hint=ParameterizedSystem,
        default_value=UNDEFINED,
    )

    enable_pbc = InputAttribute(
        docstring="If true, periodic boundary conditions will be enabled.",
        type_hint=bool,
        default_value=True,
    )

    allow_gpu_platforms = InputAttribute(
        docstring="If true, the simulation will be performed using a GPU if available, "
        "otherwise it will be constrained to only using CPUs.",
        type_hint=bool,
        default_value=True,
    )
    high_precision = InputAttribute(
        docstring="If true, the simulation will be run using double precision.",
        type_hint=bool,
        default_value=False,
    )

    gradient_parameters = InputAttribute(
        docstring="An optional list of parameters to differentiate the evaluated "
        "energies with respect to.",
        type_hint=list,
        default_value=lambda: list(),
    )

    output_coordinate_file = OutputAttribute(
        docstring="The file path to the coordinates of the final system configuration.",
        type_hint=str,
    )
    trajectory_file_path = OutputAttribute(
        docstring="The file path to the trajectory sampled during the simulation.",
        type_hint=str,
    )

    observables = OutputAttribute(
        docstring="The observables collected during the simulation.",
        type_hint=ObservableFrame,
    )
