import itertools
from copy import deepcopy

import numpy as np


def flatten_list(nested_list):
    return list(itertools.chain(*nested_list))


def redefine_distance(distance_function, distance_same_camera):
    """
    In order to not match trackers of the same camera, we need to make sure that the
    distance between objects of the same camera is at least:
        - distance_threshold: if join_distance_by is 'max'
        - amount_of_cameras**2 * distance_threshold: if join_distance_by is "mean"
    """

    def new_distance(tracked_object_1, tracked_object_2):
        if tracked_object_1.camera_name == tracked_object_2.camera_name:
            return distance_same_camera
        else:
            return distance_function(tracked_object_1, tracked_object_2)

    return new_distance


class ClustersList:
    def __init__(self, clusters):
        self.clusters = clusters
        self.all_track_ids = set.union(
            *[set(cluster.tracked_ids) for cluster in clusters]
        )

    def __len__(self):
        return len(self.clusters)


class Cluster:
    def __init__(self, tracked_object=None, id=None):
        """
        Class that relates trackers from different videos

        Attributes:
        - Cluster.tracked_objects: dict of the form {str: TrackedObject}
            where str will indicate the name of the camera/video
        - Cluster.tracked_ids: list of tuples of the form (str, int)
            where str is the name of the camera, and the int is the TrackedObject.id
        """

        self.id = id

        if tracked_object is not None:
            self.tracked_objects = {tracked_object.camera_name: tracked_object}
            self.tracked_ids = [(tracked_object.camera_name, tracked_object.id)]
        else:
            self.tracked_objects = {}
            self.tracked_ids = []

        self.grow_votes = 0
        self.split_votes = 0

        self.has_been_covered = False
        self.total_covered = 0

    def __len__(self):
        return len(self.tracked_objects)


class MultiCameraClusterizer:
    def __init__(
        self,
        distance_function,
        distance_threshold: float,
        join_distance_by: str = "mean",
        max_votes_grow: int = 5,
        max_votes_split: int = 5,
        memory: int = 3,
    ):
        """
        Associate trackers from different cameras/videos.

        Arguments:
         - distance_function: function that takes two TrackedObject instances and returns a non negative number.
            This indicates how you meassure the distance between two tracked objects of different videos.

         - distance_threshold: float.
            How far two clusters (group of trackers) need to be to not join them.

         - join_distance_by: str.
            String indicating how we 'merge' distance between trackers to construct a distance between clusters.
            Can be either 'max' or 'mean'.

         - max_votes_grow: int.
            For how many updates should we wait before increasing the size of a cluster, whenever
            a cluster we have is strictly inside another we currently see.

         - max_votes_split: int.
            For how many updates should we wait before increasing the size of a cluster, whenever
            a cluster we have is never inside another we currently see.

         - memory: int.
            Merge the information of the current update with past updates to generate clusters and vote (to grow, split or neither).
            This parameter indicates how far into the past we should look.
        """
        if max_votes_grow < 1:
            raise ValueError("max_votes_grow parameter needs to be >= 1")
        if max_votes_split < 1:
            raise ValueError("max_votes_split parameter needs to be >= 1")
        if memory < 0:
            raise ValueError("memory parameter needs to be >= 0")

        self.last_cluster_id = 0

        if join_distance_by not in ["mean", "max"]:
            raise ValueError(
                f"join_distance_by argument should be either 'mean' or 'max'."
            )

        # distance function should always return non negative numbers
        self.distance_function = distance_function
        self.distance_threshold = distance_threshold
        self.join_distance_by = join_distance_by

        self.clusters = []
        self.past_clusters = []

        self.memory = memory

        self.max_votes_grow = max_votes_grow
        self.max_votes_split = max_votes_split

    def update(self, trackers_by_camera):
        # trackers_by_camera = [list(tracked_objects_camera_0), list(tracked_objects_camera_1),...]

        # In case number of camera is variable, I will redefine the distance function
        distance_same_camera = (
            len(trackers_by_camera) ** 2 * self.distance_threshold + 1
        )
        distance_function = redefine_distance(
            self.distance_function, distance_same_camera
        )

        current_clusters = flatten_list(trackers_by_camera)

        if len(current_clusters) > 0:
            distance_matrix = (
                np.zeros((len(current_clusters), len(current_clusters)))
                + distance_same_camera
            )

            # this could have been better optimized, since we don't need to iterate over tracked_objects of the same camera
            for n, tracked_object_1 in enumerate(current_clusters):
                for m, tracked_object_2 in enumerate(current_clusters[:n]):
                    distance_matrix[n, m] = distance_function(
                        tracked_object_1, tracked_object_2
                    )
                    distance_matrix[m, n] = distance_matrix[n, m]

            # change the type from TrackedObject to Cluster, to initialize individual clusters
            current_clusters = [
                Cluster(tracked_object) for tracked_object in current_clusters
            ]

            min_distance = distance_matrix.min()

            # join clusters iteratively by looking at distances
            while min_distance < self.distance_threshold:

                flattened_arg_min = distance_matrix.argmin()

                number_cluster_A = flattened_arg_min // distance_matrix.shape[1]
                number_cluster_B = flattened_arg_min % distance_matrix.shape[1]

                cluster_A = current_clusters[number_cluster_A]
                cluster_B = current_clusters[number_cluster_B]

                if self.join_distance_by == "max":
                    distance_to_joined_cluster = np.maximum(
                        distance_matrix[number_cluster_A],
                        distance_matrix[number_cluster_B],
                    )
                elif self.join_distance_by == "mean":
                    size_cluster_A = len(cluster_A)
                    size_cluster_B = len(cluster_B)
                    distance_to_joined_cluster = (
                        size_cluster_A * distance_matrix[number_cluster_A]
                        + size_cluster_B * distance_matrix[number_cluster_B]
                    ) / (size_cluster_A + size_cluster_B)
                else:
                    raise ValueError(
                        f"MultiCameraClusterizer.join_distance_by was changed to a value that is not 'mean' or 'max'."
                    )

                distance_matrix[number_cluster_A] = distance_to_joined_cluster
                distance_matrix[:, number_cluster_A] = distance_to_joined_cluster

                distance_matrix[number_cluster_B] = self.distance_threshold + 1
                distance_matrix[:, number_cluster_B] = self.distance_threshold + 1

                current_clusters[number_cluster_A].tracked_objects.update(
                    cluster_B.tracked_objects
                )
                current_clusters[number_cluster_A].tracked_ids.extend(
                    cluster_B.tracked_ids
                )
                current_clusters[number_cluster_B] = None

                min_distance = distance_matrix.min()

            current_clusters = [
                cluster for cluster in current_clusters if cluster is not None
            ]
            current_clusters = sorted(current_clusters, key=len)

            current_clusters = ClustersList(current_clusters)

            if len(self.past_clusters) == self.memory:
                self.past_clusters.pop(0)

            self.past_clusters.append(deepcopy(current_clusters))

            # Let's intersect the past clusters
            past_cluster_number = 0
            while past_cluster_number < len(self.past_clusters) - 1:
                past_cluster = self.past_clusters[past_cluster_number]
                accumulated_intersections_since_we_started_with_this_past_cluster = 0
                covered_current_cluster = False

                # lists as [[('cam0', '2'), ('cam2', '2')], [('cam0', '1'), ('cam1', '1'), ('cam2', '1')]]
                past_cluster_as_list = [
                    cluster.tracked_ids for cluster in past_cluster.clusters
                ]

                sorted(past_cluster_as_list, key=len, reverse=True)

                cluster_B_number = 0
                for n in range(len(current_clusters)):
                    current_clusters.clusters[n].has_been_covered = False
                while cluster_B_number < len(past_cluster_as_list):
                    cluster_B = past_cluster_as_list[cluster_B_number]
                    accumulated_intersections_since_we_started_with_cluster_B = 0
                    covered_cluster_B = False

                    cluster_A_number = 0
                    while cluster_A_number < len(current_clusters):
                        cluster_A = current_clusters.clusters[cluster_A_number]

                        if not cluster_A.has_been_covered:
                            intersection = set(cluster_A.tracked_ids).intersection(
                                set(cluster_B)
                            )
                            if len(intersection) > 0:
                                accumulated_intersections_since_we_started_with_cluster_B += len(
                                    intersection
                                )
                                accumulated_intersections_since_we_started_with_this_past_cluster += len(
                                    intersection
                                )

                                covered_cluster_B = (
                                    accumulated_intersections_since_we_started_with_cluster_B
                                    == len(cluster_B)
                                )
                                covered_current_cluster = (
                                    accumulated_intersections_since_we_started_with_this_past_cluster
                                    == len(current_clusters.all_track_ids)
                                )

                                if len(cluster_A) > len(intersection):
                                    # split cluster_A

                                    missing_indices = (
                                        set(cluster_A.tracked_ids) - intersection
                                    )
                                    missing_trackers = {}
                                    for (camera_name, track_id) in missing_indices:
                                        missing_trackers[
                                            camera_name
                                        ] = cluster_A.tracked_objects.pop(camera_name)

                                    cluster_A.tracked_ids = list(intersection)
                                    cluster_A.has_been_covered = True

                                    missing_cluster = Cluster()
                                    missing_cluster.tracked_objects = missing_trackers
                                    missing_cluster.tracked_ids = list(missing_indices)

                                    current_clusters.clusters[
                                        cluster_A_number
                                    ] = cluster_A
                                    current_clusters.clusters.append(missing_cluster)
                                else:
                                    cluster_A.has_been_covered = True

                                if covered_current_cluster:
                                    # if the past_cluster already covered the current_cluster, then go to next past_cluster
                                    cluster_A_number = len(current_clusters) + 1
                                    cluster_B_number = len(past_cluster_as_list) + 1
                                    past_cluster_number += 1
                                elif covered_cluster_B:
                                    # if I covered cluster_B with current_cluster, then go to next cluster_B in past_cluster
                                    cluster_A_number = len(current_clusters) + 1
                                    cluster_B_number += 1
                        cluster_A_number += 1
                    cluster_B_number += 1
                past_cluster_number += 1

            # Now current_clusters combines the information of the recent past

            if len(self.clusters) == 0:
                # if we had no clusters previously, just use the new ones
                for n in range(len(current_clusters)):
                    current_clusters.clusters[n].id = self.last_cluster_id
                    self.last_cluster_id += 1
                self.clusters = current_clusters.clusters
            else:
                # if we had clusters, we need to do intersections to see which current clusters correspond to and old one
                # this will allow us to update the old cluster (and creating new ones if we need to)

                # compute the matrix of intersection of clusters
                cardinal_intersection_matrix = []
                intersection_matrix_ids = []

                cluster_number = 0
                new_self_clusters = []
                while cluster_number < len(self.clusters):
                    new_row_interesection_ids = [None] * len(current_clusters)
                    new_row_cardinal_intersection = np.zeros((len(current_clusters),))

                    cluster = self.clusters[cluster_number]

                    copy_cluster = deepcopy(cluster)
                    copy_cluster.tracked_ids = []
                    copy_cluster.tracked_objects = {}

                    size_of_cluster = len(cluster)
                    size_covered = 0

                    current_cluster_number = 0
                    while current_cluster_number < len(current_clusters):
                        current_cluster = current_clusters.clusters[
                            current_cluster_number
                        ]

                        if current_cluster.total_covered == len(current_cluster):
                            current_cluster_number += 1
                            continue

                        intersection = set(cluster.tracked_ids).intersection(
                            set(current_cluster.tracked_ids)
                        )
                        copy_cluster.tracked_ids.extend(list(intersection))
                        for (camera_name, track_id) in intersection:
                            copy_cluster.tracked_objects[
                                camera_name
                            ] = current_cluster.tracked_objects[camera_name]

                        new_row_interesection_ids[current_cluster_number] = intersection

                        intersection_size = len(intersection)
                        size_covered += intersection_size
                        new_row_cardinal_intersection[
                            current_cluster_number
                        ] = intersection_size

                        current_cluster.total_covered += intersection_size
                        current_cluster.has_been_covered = (
                            current_cluster.total_covered == len(current_cluster)
                        )

                        if size_covered == size_of_cluster:
                            # all other intersections with current_clusters should be empty afterwards for this cluster
                            current_cluster_number = len(current_clusters) + 1
                        else:
                            current_cluster_number += 1

                    if size_covered > 0:
                        new_self_clusters.append(copy_cluster)
                        cardinal_intersection_matrix.append(
                            new_row_cardinal_intersection
                        )
                        intersection_matrix_ids.append(new_row_interesection_ids)

                    cluster_number += 1

                self.clusters = new_self_clusters

                # once I have the matrix of intersections, I check if my clusters need to grow or be splitted
                cluster_number = 0
                while cluster_number < len(self.clusters):

                    cluster = self.clusters[cluster_number]

                    number_current_cluster_with_biggest_intersection = (
                        cardinal_intersection_matrix[cluster_number].argmax()
                    )
                    intersection_size = cardinal_intersection_matrix[cluster_number][
                        number_current_cluster_with_biggest_intersection
                    ]
                    current_cluster_with_biggest_intersection = (
                        current_clusters.clusters[
                            number_current_cluster_with_biggest_intersection
                        ]
                    )

                    if intersection_size < size_of_cluster:
                        cluster.split_votes = min(
                            self.max_votes_split, cluster.split_votes + 1
                        )
                        cluster.grow_votes = max(0, cluster.grow_votes - 1)
                    elif (
                        len(current_cluster_with_biggest_intersection)
                        == size_of_cluster
                    ):
                        cluster.split_votes = max(0, cluster.split_votes - 1)
                        cluster.grow_votes = max(0, cluster.grow_votes - 1)
                    else:
                        cluster.grow_votes = min(
                            self.max_votes_grow, cluster.grow_votes + 1
                        )
                        cluster.split_votes = max(0, cluster.split_votes - 1)

                    if cluster.grow_votes == self.max_votes_grow:
                        # if the votes to grow are enough, then we will expand our cluster
                        # we might need to steal ids from other clusters, so first we will remove those from the others
                        cluster.grow_votes = 0
                        cluster.split_votes = 0

                        other_cluster_number = 0
                        while other_cluster_number < len(self.clusters):
                            if (other_cluster_number != cluster_number) and (
                                cardinal_intersection_matrix[other_cluster_number][
                                    number_current_cluster_with_biggest_intersection
                                ]
                                > 0
                            ):
                                intersection_with_other_cluster = (
                                    intersection_matrix_ids[other_cluster_number][
                                        number_current_cluster_with_biggest_intersection
                                    ]
                                )

                                # eliminate the tracked_ids that appear in the other cluster
                                self.clusters[other_cluster_number].tracked_ids = list(
                                    set(self.clusters[other_cluster_number].tracked_ids)
                                    - intersection_with_other_cluster
                                )

                                # eliminate the corresponding tracked_objects in the other cluster
                                for (
                                    camera_name,
                                    track_id,
                                ) in intersection_with_other_cluster:
                                    self.clusters[
                                        other_cluster_number
                                    ].tracked_objects.pop(camera_name)

                                # keep in the bigger cluster, the smallest id
                                original_id = cluster.id
                                original_other_id = self.clusters[
                                    other_cluster_number
                                ].id
                                if len(self.clusters[other_cluster_number]) <= len(
                                    cluster
                                ):
                                    cluster.id = min(original_id, original_other_id)
                                else:
                                    cluster.id = max(original_id, original_other_id)
                                self.clusters[other_cluster_number].id = (
                                    original_id + original_other_id - cluster.id
                                )

                                # update the matrices of intersection for the other clusters
                                intersection_matrix_ids[other_cluster_number][
                                    number_current_cluster_with_biggest_intersection
                                ] = None
                                cardinal_intersection_matrix[other_cluster_number][
                                    number_current_cluster_with_biggest_intersection
                                ] = 0

                            other_cluster_number += 1

                        cluster.tracked_objects = (
                            current_cluster_with_biggest_intersection.tracked_objects
                        )
                        cluster.tracked_ids = (
                            current_cluster_with_biggest_intersection.tracked_ids
                        )
                        self.clusters[cluster_number] = cluster

                        intersection_matrix_ids[cluster_number][
                            number_current_cluster_with_biggest_intersection
                        ] = set(current_cluster_with_biggest_intersection.tracked_ids)

                    elif cluster.split_votes == self.max_votes_split:
                        # if we have enough votes to split our cluster
                        # we update the old cluster with the information of the biggest current cluster inside
                        # for the other current clusters that intersect it, we create new clusters
                        cluster.split_votes = 0
                        cluster.grow_votes = 0

                        other_current_cluster_number = 0

                        oldest_tracker_age_in_other_cluster = -1
                        cluster_number_with_oldest_tracker = None
                        while other_current_cluster_number < len(current_clusters):
                            if (
                                other_current_cluster_number
                                != number_current_cluster_with_biggest_intersection
                            ) and (
                                cardinal_intersection_matrix[cluster_number][
                                    other_current_cluster_number
                                ]
                                > 0
                            ):
                                intersection = intersection_matrix_ids[cluster_number][
                                    other_current_cluster_number
                                ]

                                # create new clusters for smaller subclusters inside
                                new_cluster = Cluster(None, self.last_cluster_id)
                                self.last_cluster_id += 1

                                new_cluster.tracked_ids = list(intersection)

                                for (camera_name, track_id) in intersection:
                                    new_cluster.tracked_objects[
                                        camera_name
                                    ] = cluster.tracked_objects[camera_name]
                                    if (
                                        oldest_tracker_age_in_other_cluster
                                        < cluster.tracked_objects[camera_name].age
                                    ):
                                        oldest_tracker_age_in_other_cluster = (
                                            cluster.tracked_objects[camera_name].age
                                        )
                                        cluster_number_with_oldest_tracker = len(
                                            self.clusters
                                        )

                                self.clusters.append(new_cluster)

                                # need to create new rows for the new cluster
                                new_row_interesection_ids = [None] * len(
                                    current_clusters
                                )
                                new_row_cardinal_intersection = np.zeros(
                                    (len(current_clusters),)
                                )

                                new_row_interesection_ids[
                                    other_current_cluster_number
                                ] = intersection
                                new_row_cardinal_intersection[
                                    other_current_cluster_number
                                ] = len(intersection)

                                cardinal_intersection_matrix.append(
                                    new_row_cardinal_intersection
                                )
                                intersection_matrix_ids.append(
                                    new_row_interesection_ids
                                )

                            other_current_cluster_number += 1

                        # update the old cluster with only the biggest subcluster inside
                        intersection = intersection_matrix_ids[cluster_number][
                            number_current_cluster_with_biggest_intersection
                        ]
                        cluster.tracked_objects = {}
                        cluster.tracked_ids = list(intersection)
                        oldest_tracker_age_in_cluster = -1
                        for camera_name, track_id in intersection:
                            cluster.tracked_objects[
                                camera_name
                            ] = current_cluster_with_biggest_intersection.tracked_objects[
                                camera_name
                            ]
                            oldest_tracker_age_in_cluster = max(
                                oldest_tracker_age_in_cluster,
                                cluster.tracked_objects[camera_name].age,
                            )

                        # keep the smallest id in the cluster that has the oldest tracker.
                        if (
                            oldest_tracker_age_in_cluster
                            < oldest_tracker_age_in_other_cluster
                        ):
                            other_cluster_id = self.clusters[
                                cluster_number_with_oldest_tracker
                            ].id
                            cluster_id = cluster.id

                            # We already know cluster_id < other_cluster_id, so min(cluster_id, other_cluster_id) = cluster_id
                            cluster.id = other_cluster_id
                            self.clusters[
                                cluster_number_with_oldest_tracker
                            ].id = cluster_id

                        # need to update the old cluster_number rows in the intersection matrix
                        new_row_interesection_ids = [None] * len(current_clusters)
                        new_row_cardinal_intersection = np.zeros(
                            (len(current_clusters),)
                        )

                        new_row_interesection_ids[
                            number_current_cluster_with_biggest_intersection
                        ] = intersection
                        new_row_cardinal_intersection = intersection_size

                        cardinal_intersection_matrix[
                            cluster_number
                        ] = new_row_cardinal_intersection
                        intersection_matrix_ids[
                            cluster_number
                        ] = new_row_interesection_ids

                        self.clusters[cluster_number] = cluster

                    cluster_number += 1

                # create new clusters with remaining ids that were not used
                all_ids_in_self_clusters = set(
                    flatten_list([cluster.tracked_ids for cluster in self.clusters])
                )
                for current_cluster in current_clusters.clusters:
                    difference_ids = (
                        set(current_cluster.tracked_ids) - all_ids_in_self_clusters
                    )

                    if len(difference_ids) > 0:
                        new_cluster = Cluster(None, self.last_cluster_id)
                        self.last_cluster_id += 1

                        for (camera_name, track_id) in difference_ids:
                            new_cluster.tracked_objects[
                                camera_name
                            ] = current_cluster.tracked_objects[camera_name]
                        new_cluster.tracked_ids = list(difference_ids)

                        self.clusters.append(new_cluster)

                # just in case there is an empty cluster, filter it
                # this might happen since we stole ids from others when growing
                # also might happen since and old cluster might have not intersected with any new cluster
                self.clusters = [
                    cluster for cluster in self.clusters if len(cluster) > 0
                ]

        return self.clusters
