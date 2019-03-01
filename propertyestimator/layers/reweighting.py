"""
The simulation reweighting estimation layer.
"""
import abc
import logging
import pickle
from os import path

from openmmtools.utils import SubhookedABCMeta

from propertyestimator.layers import register_calculation_layer, PropertyCalculationLayer
from propertyestimator.utils.serialization import serialize_force_field
from propertyestimator.workflow import WorkflowGraph, Workflow
from propertyestimator.workflow.workflow import IWorkflowProperty


class IReweightable(SubhookedABCMeta):

    @property
    @abc.abstractmethod
    def multi_component_property(self): pass


@register_calculation_layer()
class ReweightingLayer(PropertyCalculationLayer):
    """A calculation layer which aims to calculate physical properties by
    reweighting the results of previous calculations.

    .. warning :: This class is still heavily under development and is subject to
                 rapid changes.
    """

    @staticmethod
    def schedule_calculation(calculation_backend, storage_backend, layer_directory,
                             data_model, callback, synchronous=False):

        # Make a local copy of the target force field.
        target_force_field = storage_backend.retrieve_force_field(data_model.force_field_id)

        target_force_field_path = path.join(layer_directory, data_model.force_field_id)

        with open(target_force_field_path, 'wb') as file:
            pickle.dump(serialize_force_field(target_force_field), file)

        stored_data_paths = ReweightingLayer._retrieve_stored_data(data_model.queued_properties,
                                                                   storage_backend, layer_directory)

        workflow_graph = ReweightingLayer._build_workflow_graph(layer_directory,
                                                                data_model.queued_properties,
                                                                target_force_field_path,
                                                                stored_data_paths,
                                                                data_model.options)

        reweighting_futures = workflow_graph.submit(calculation_backend)

        PropertyCalculationLayer._await_results(calculation_backend, storage_backend, layer_directory,
                                                data_model, callback, reweighting_futures, synchronous)

    @staticmethod
    def _retrieve_stored_data(physical_properties, storage_backend, layer_directory):
        """Extract all of the stored data from the backend which may be
        used in reweighting

        Parameters
        ----------
        physical_properties: list of PhysicalProperty
            The physical properties to attempt to estimate.
        storage_backend: PropertyEstimatorStorage
            The storage backend to retrieve the data from.
        layer_directory: str
            The directory in which to store the retrieved data.

        Returns
        -------
        dict of str and tuple(str, str)
            A dictionary partitioned by substance identifiers, whose values
            are a tuple of a path to a stored simulation data object, and
            its corresponding force field path.
        """

        data_paths = {}

        for physical_property in physical_properties:

            if not isinstance(physical_property, IReweightable):
                # Only properties which implement the IReweightable
                # interface can be reweighted
                continue

            existing_data = storage_backend.retrieve_simulation_data(physical_property.substance,
                                                                     physical_property.multi_component_property)

            if len(existing_data) == 0:
                continue

            # Take data from the storage backend and save it in the working directory.
            for substance_id in existing_data:

                if substance_id not in data_paths:
                    data_paths[substance_id] = []

                for data in existing_data[substance_id]:

                    data_path = path.join(layer_directory, data.unique_id)
                    force_field_path = path.join(layer_directory, data.force_field_id)

                    path_tuple = (data_path, force_field_path)

                    if path_tuple in data_paths[substance_id]:
                        continue

                    with open(data_path, 'wb') as file:
                        pickle.dump(data, file)

                    if not path.isfile(force_field_path):

                        existing_force_field = storage_backend.retrieve_force_field(data.force_field_id)

                        with open(force_field_path, 'wb') as file:
                            pickle.dump(serialize_force_field(existing_force_field), file)

                    data_paths[substance_id].append(path_tuple)

        return data_paths

    @staticmethod
    def _build_workflow_graph(working_directory, properties, target_force_field_path,
                              stored_data_paths, options):
        """Construct a workflow graph, containing all of the workflows which should
        be followed to estimate a set of properties by reweighting.

        Parameters
        ----------
        working_directory: str
            The local directory in which to store all local,
            temporary calculation data from this graph.
        properties : list of PhysicalProperty
            The properties to attempt to compute.
        target_force_field_path : str
            The path to the target force field parameters to use in the workflow.
        stored_data_paths: dict of str and tuple(str, str)
            A dictionary partitioned by substance identifiers, whose values
            are a tuple of a path to a stored simulation data object, and
            its corresponding force field path.
        options: PropertyEstimatorOptions
            The options to run the workflows with.
        """
        workflow_graph = WorkflowGraph(working_directory)

        for property_to_calculate in properties:

            if (not isinstance(property_to_calculate, IReweightable) or
                not isinstance(property_to_calculate, IWorkflowProperty)):
                # Only properties which implement the IReweightable and
                # IWorkflowProperty interfaces can be reweighted
                continue

            property_type = type(property_to_calculate).__name__

            if property_type not in options.workflow_schemas:

                logging.warning('The reweighting layer does not support {} '
                                'workflows.'.format(property_type))

                continue

            if ReweightingLayer.__name__ not in options.workflow_schemas[property_type]:
                continue

            schema = options.workflow_schemas[property_type][ReweightingLayer.__name__]

            global_metadata = Workflow.generate_default_metadata(property_to_calculate,
                                                                 target_force_field_path, options)

            substance_identifiers = [property_to_calculate.substance.identifier]

            if property_to_calculate.multi_component_property:

                substance_identifiers.extend([component.identifier for
                                              component in property_to_calculate.substance])

            global_metadata['full_system_data'] = []
            global_metadata['component_data'] = {}

            has_data_for_property = True

            for substance_identifier in substance_identifiers:

                if substance_identifier not in stored_data_paths:

                    has_data_for_property = False
                    break

                stored_data = stored_data_paths[substance_identifier]

                if substance_identifier == property_to_calculate.substance.identifier:
                    global_metadata['full_system_data'] = stored_data
                else:
                    global_metadata['component_data'][substance_identifier] = stored_data

            if not has_data_for_property:
                continue

            workflow = Workflow(property_to_calculate, global_metadata)
            workflow.schema = schema

            workflow_graph.add_workflow(workflow)

        return workflow_graph
