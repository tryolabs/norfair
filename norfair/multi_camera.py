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
        try:
            self.all_track_ids = set.union(
                *[set(cluster.tracked_ids) for cluster in clusters]
            )
        except TypeError:
            self.all_track_ids = set()

    def __len__(self):
        return len(self.clusters)


class Cluster:
    def __init__(self, tracked_object=None, fake_id=None):
        """
        Class that relates trackers from different videos

        Attributes:
        - Cluster.id: number identifying the cluster
        - Cluster.tracked_objects: dict of the form {str: TrackedObject}
            where str will indicate the name of the camera/video
        - Cluster.tracked_ids: list of tuples of the form (str, int)
            where str is the name of the camera, and the int is the TrackedObject.id
        """

        self.fake_id = fake_id
        self.id = None

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
        self.reid_hit_counter = {}
        self.age = 0

    def __len__(self):
        return len(self.tracked_ids)


def cluster_intersection_matrix(current_clusters, clusters):

    cardinal_intersection_matrix = []
    intersection_matrix_ids = []

    cluster_number = 0
    while cluster_number < len(clusters):
        new_row_interesection_ids = [set()] * len(current_clusters)
        new_row_cardinal_intersection = np.zeros((len(current_clusters),))

        cluster = clusters[cluster_number]

        size_of_cluster = len(cluster)
        size_covered = 0

        current_cluster_number = 0
        while current_cluster_number < len(current_clusters):
            current_cluster = current_clusters.clusters[current_cluster_number]

            if current_cluster.total_covered == len(current_cluster):
                current_cluster_number += 1
                continue

            intersection = set(cluster.tracked_ids).intersection(
                set(current_cluster.tracked_ids)
            )
            new_row_interesection_ids[current_cluster_number] = intersection

            # update the old cluster with the new tracker information
            for camera_name, track_id in intersection:
                cluster.reid_hit_counter[camera_name] = 0
                cluster.tracked_objects[camera_name] = current_cluster.tracked_objects[
                    camera_name
                ]

            intersection_size = len(intersection)
            new_row_cardinal_intersection[current_cluster_number] = intersection_size

            size_covered += intersection_size
            current_cluster.total_covered += intersection_size

            if size_covered == size_of_cluster:
                # all other intersections with current_clusters should be empty afterwards for this cluster
                current_cluster_number = len(current_clusters) + 1
            else:
                current_cluster_number += 1

        cardinal_intersection_matrix.append(new_row_cardinal_intersection)
        intersection_matrix_ids.append(new_row_interesection_ids)

        clusters[cluster_number] = cluster

        cluster_number += 1

    return intersection_matrix_ids, cardinal_intersection_matrix, clusters


def generate_current_clusters(
    trackers_by_camera,
    distance_function,
    distance_threshold,
    join_distance_by="mean",
    use_only_living_trackers=False,
):

    # In case number of camera is variable, I will redefine the distance function
    distance_same_camera = len(trackers_by_camera) ** 2 * distance_threshold + 1
    distance_function = redefine_distance(distance_function, distance_same_camera)

    current_clusters = flatten_list(trackers_by_camera)

    # use only alive trackers:
    if use_only_living_trackers:
        current_clusters = [obj for obj in current_clusters if obj.live_points.any()]

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
        while min_distance < distance_threshold:

            flattened_arg_min = distance_matrix.argmin()

            number_cluster_A = flattened_arg_min // distance_matrix.shape[1]
            number_cluster_B = flattened_arg_min % distance_matrix.shape[1]

            cluster_A = current_clusters[number_cluster_A]
            cluster_B = current_clusters[number_cluster_B]

            if join_distance_by == "max":
                distance_to_joined_cluster = np.maximum(
                    distance_matrix[number_cluster_A],
                    distance_matrix[number_cluster_B],
                )
            elif join_distance_by == "mean":
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

            distance_matrix[number_cluster_B] = distance_threshold + 1
            distance_matrix[:, number_cluster_B] = distance_threshold + 1

            current_clusters[number_cluster_A].tracked_objects.update(
                cluster_B.tracked_objects
            )
            current_clusters[number_cluster_A].tracked_ids.extend(cluster_B.tracked_ids)
            current_clusters[number_cluster_B] = None

            min_distance = distance_matrix.min()

        current_clusters = [
            cluster for cluster in current_clusters if cluster is not None
        ]
        current_clusters = sorted(current_clusters, key=len)

        return ClustersList(current_clusters)
    else:
        return ClustersList([])


def intersect_past_clusters(past_clusters):
    current_clusters = deepcopy(past_clusters[0])
    past_cluster_number = 1
    while past_cluster_number < len(past_clusters):
        past_cluster = past_clusters[past_cluster_number]
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
                        accumulated_intersections_since_we_started_with_cluster_B += (
                            len(intersection)
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

                            missing_indices = set(cluster_A.tracked_ids) - intersection
                            missing_trackers = {}
                            for (camera_name, track_id) in missing_indices:
                                missing_trackers[
                                    camera_name
                                ] = cluster_A.tracked_objects.pop(camera_name)

                            cluster_A.tracked_ids = list(intersection)

                            missing_cluster = Cluster()
                            missing_cluster.tracked_objects = missing_trackers
                            missing_cluster.tracked_ids = list(missing_indices)

                            current_clusters.clusters.append(missing_cluster)

                        cluster_A.has_been_covered = True
                        current_clusters.clusters[cluster_A_number] = cluster_A

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

    return current_clusters


def update_cluster_votes(
    cluster_number, cardinal_intersection_matrix, current_clusters, clusters
):

    cluster = clusters[cluster_number]
    if len(current_clusters) > 0:

        visible_trackers = [
            (camera_name, track_id)
            for (camera_name, track_id) in cluster.tracked_ids
            if cluster.reid_hit_counter[camera_name] == 0
        ]

        number_current_cluster_with_biggest_intersection = cardinal_intersection_matrix[
            cluster_number
        ].argmax()

        if (cardinal_intersection_matrix[cluster_number] > 0).sum() > 1:
            # in this case, the visible trackers are splitted into several current clusters, so vote to split
            cluster.split_votes = cluster.split_votes + 1
            cluster.grow_votes = max(0, cluster.grow_votes - 1)
            return cluster, number_current_cluster_with_biggest_intersection

        elif (
            len(visible_trackers)
            < len(
                current_clusters.clusters[
                    number_current_cluster_with_biggest_intersection
                ]
            )
        ) and (len(visible_trackers) > 0):
            # in this case, the visible trackers are strictly contained in one of the current clusters
            cluster.grow_votes = cluster.grow_votes + 1
            cluster.split_votes = max(0, cluster.split_votes - 1)
            return cluster, number_current_cluster_with_biggest_intersection

    # the visible trackers are exactly one of the current clusters
    cluster.split_votes = max(0, cluster.split_votes - 1)
    cluster.grow_votes = max(0, cluster.grow_votes - 1)

    return cluster, None


def remove_current_cluster_from_clusters(
    clusters,
    number_current_cluster_with_biggest_intersection,
    intersection_matrix_ids,
    cardinal_intersection_matrix,
):
    n = 0
    cluster_numbers_with_oldest_tracker = []
    oldest_age = -1
    while n < len(clusters):
        if (
            cardinal_intersection_matrix[n][
                number_current_cluster_with_biggest_intersection
            ]
            > 0
        ):

            intersection = intersection_matrix_ids[n][
                number_current_cluster_with_biggest_intersection
            ]
            cluster = clusters[n]

            cluster.tracked_ids = list(set(cluster.tracked_ids) - intersection)
            for (
                camera_name,
                track_id,
            ) in intersection:
                tracked_object = clusters[n].tracked_objects.pop(camera_name)
                if tracked_object.age == oldest_age:
                    cluster_numbers_with_oldest_tracker.append(n)
                elif tracked_object.age > oldest_age:
                    cluster_numbers_with_oldest_tracker = [n]
                    oldest_age = tracked_object.age

            # update the matrices of intersection for the other clusters
            intersection_matrix_ids[n][
                number_current_cluster_with_biggest_intersection
            ] = set()
            cardinal_intersection_matrix[n][
                number_current_cluster_with_biggest_intersection
            ] = 0

            clusters[n] = cluster

        n += 1

    return (
        clusters,
        cluster_numbers_with_oldest_tracker,
        intersection_matrix_ids,
        cardinal_intersection_matrix,
    )


def swap_cluster_ids(clusters, cluster_number, cluster_number_with_oldest_tracker):

    old_cluster = clusters[cluster_number_with_oldest_tracker]
    old_cluster_fake_id = old_cluster.fake_id

    cluster = clusters[cluster_number]
    cluster_fake_id = cluster.fake_id

    if old_cluster_fake_id < cluster_fake_id:
        old_cluster_age = old_cluster.age
        old_cluster_id = old_cluster.id

        cluster_age = cluster.age
        cluster_id = cluster.id
        cluster.id = old_cluster_id
        cluster.fake_id = old_cluster_fake_id
        cluster.age = old_cluster_age
        old_cluster.id = cluster_id
        old_cluster.fake_id = cluster_fake_id
        old_cluster.age = cluster_age

        clusters[cluster_number_with_oldest_tracker] = old_cluster
        clusters[cluster_number] = cluster
    return clusters


class MultiCameraClusterizer:
    def __init__(
        self,
        distance_function,
        distance_threshold: float,
        join_distance_by: str = "mean",
        max_votes_grow: int = 5,
        max_votes_split: int = 5,
        memory: int = 3,
        initialization_delay: int = 4,
        reid_hit_counter_max: int = 0,
        use_only_living_trackers: bool = False,
    ):
        """
        Associate trackers from different cameras/videos.

        Arguments:
         - distance_function: function that takes two TrackedObject instances and returns a non negative number.
            This indicates how you meassure the distance between two tracked objects of different videos.

         - distance_threshold: float.
            How far two clusters (group of trackers) need to be to not join them.

         - join_distance_by: str.
            String indicating how we combine distance between trackers to construct a distance between clusters.
            Each cluster will have several TrackedObject instances, so in our approach we can either take
            the maximum distance between their TrackedObject instances, or the average distance.
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

        - initialization_delay: int.
            When a new cluster is created, we wait a few frames before returning it in the update method, so that new clusters
            have the chance to be merged with other existing clusters.

        - reid_hit_counter_max: int.
            If doing reid in the tracking, then provide the reid_hit_counter_max so that the MultiCameraClusterizer instance knows
            for how long to keep storing clusters of tracked objects that have dissapeared.

        - use_only_living_trackers: bool.
            Filter tracked objects that have no alive points. This can be useful since tracked objects that are not alive might have
            position that will not match well with their position in a different camera.
        """
        if max_votes_grow < 1:
            raise ValueError("max_votes_grow parameter needs to be >= 1")
        if max_votes_split < 1:
            raise ValueError("max_votes_split parameter needs to be >= 1")
        if memory < 0:
            raise ValueError("memory parameter needs to be >= 0")

        self.last_cluster_fake_id = 0
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

        # I will give the trackers at least enough time to merge their cluster with another
        self.initialization_delay = initialization_delay + max_votes_grow

        self.reid_hit_counter_max = reid_hit_counter_max + 1
        self.use_only_living_trackers = use_only_living_trackers

    def update(self, trackers_by_camera):

        # increase all reid_hit_counter_by_one
        cluster_number = 0
        while cluster_number < len(self.clusters):
            cluster = self.clusters[cluster_number]
            cluster.tracked_objects = {}
            for camera_name, track_id in cluster.tracked_ids:
                if cluster.reid_hit_counter[camera_name] > self.reid_hit_counter_max:
                    cluster.tracked_ids.remove((camera_name, track_id))
                else:
                    cluster.reid_hit_counter[camera_name] += 1
            cluster_number += 1

        # generate current clusters
        current_clusters = generate_current_clusters(
            trackers_by_camera,
            self.distance_function,
            self.distance_threshold,
            self.join_distance_by,
            self.use_only_living_trackers,
        )
        self.past_clusters.insert(0, deepcopy(current_clusters))

        if len(self.past_clusters) > self.memory:
            self.past_clusters = self.past_clusters[: self.memory]

        # Let's intersect the past clusters
        current_clusters = intersect_past_clusters(self.past_clusters)

        # compute intersection of current_cluster with self.clusters
        (
            intersection_matrix_ids,
            cardinal_intersection_matrix,
            self.clusters,
        ) = cluster_intersection_matrix(current_clusters, self.clusters)
        # once I have the matrix of intersections, I check if my clusters need to grow or be splitted
        cluster_number = 0
        while cluster_number < len(self.clusters):

            (
                cluster,
                number_current_cluster_with_biggest_intersection,
            ) = update_cluster_votes(
                cluster_number,
                cardinal_intersection_matrix,
                current_clusters,
                self.clusters,
            )

            if cluster.grow_votes == self.max_votes_grow:
                cluster.grow_votes = 0
                cluster.split_votes = 0
                self.clusters[cluster_number] = cluster

                big_current_cluster = current_clusters.clusters[
                    number_current_cluster_with_biggest_intersection
                ]
                (
                    self.clusters,
                    cluster_numbers_with_oldest_tracker,
                    intersection_matrix_ids,
                    cardinal_intersection_matrix,
                ) = remove_current_cluster_from_clusters(
                    self.clusters,
                    number_current_cluster_with_biggest_intersection,
                    intersection_matrix_ids,
                    cardinal_intersection_matrix,
                )
                cluster = self.clusters[cluster_number]

                # remove track ids from cluster that have a common camera_name with big_current_cluster
                camera_names_in_big_current_cluster = [
                    camera_name
                    for (camera_name, track_id) in big_current_cluster.tracked_ids
                ]
                cluster.tracked_ids = [
                    (camera_name, track_id)
                    for (camera_name, track_id) in cluster.tracked_ids
                    if camera_name not in camera_names_in_big_current_cluster
                ]

                cluster.tracked_ids.extend(big_current_cluster.tracked_ids)
                for (
                    camera_name,
                    tracked_object,
                ) in big_current_cluster.tracked_objects.items():
                    cluster.tracked_objects[camera_name] = tracked_object

                self.clusters[cluster_number] = cluster

                # keep the smallest id with the oldest object
                self.clusters = swap_cluster_ids(
                    self.clusters,
                    cluster_number,
                    np.array(cluster_numbers_with_oldest_tracker).min(),
                )

                # update the matrix of intersections so that the current cluster is now contained in self.clusters[cluster_number]
                intersection_matrix_ids[cluster_number][
                    number_current_cluster_with_biggest_intersection
                ] = set(big_current_cluster.tracked_ids)
                cardinal_intersection_matrix[cluster_number][
                    number_current_cluster_with_biggest_intersection
                ] = len(big_current_cluster.tracked_ids)
            elif cluster.split_votes == self.max_votes_split:
                cluster.grow_votes = 0
                cluster.split_votes = 0

                # create the aditional clusters
                additional_clusters = []
                cluster_number_with_oldest_tracker = None
                oldest_tracker_age_in_additional_cluster = -1
                for current_cluster_number, tracked_ids in enumerate(
                    intersection_matrix_ids[cluster_number]
                ):

                    if len(tracked_ids) > 0:
                        # we remove the tracked_ids subcluster by subcluster, to keep the non visible tracked objects in the cluster
                        cluster.tracked_ids = list(
                            set(cluster.tracked_ids) - tracked_ids
                        )

                        new_cluster = Cluster(None, self.last_cluster_fake_id)
                        self.last_cluster_fake_id += 1

                        new_cluster.tracked_ids = list(tracked_ids)

                        for camera_name, track_id in tracked_ids:
                            new_cluster.tracked_objects[
                                camera_name
                            ] = cluster.tracked_objects[camera_name]
                            new_cluster.reid_hit_counter[camera_name] = 0

                            if (
                                oldest_tracker_age_in_additional_cluster
                                < cluster.tracked_objects[camera_name].age
                            ):
                                oldest_tracker_age_in_additional_cluster = (
                                    cluster.tracked_objects[camera_name].age
                                )
                                cluster_number_with_oldest_tracker = (
                                    current_cluster_number
                                )

                        additional_clusters.append(new_cluster)
                    else:
                        additional_clusters.append(None)

                cluster.tracked_ids.extend(
                    additional_clusters[cluster_number_with_oldest_tracker].tracked_ids
                )
                cluster.tracked_objects = additional_clusters[
                    cluster_number_with_oldest_tracker
                ].tracked_objects
                cluster.reid_hit_counter.update(
                    additional_clusters[
                        cluster_number_with_oldest_tracker
                    ].reid_hit_counter
                )

                self.clusters[cluster_number] = cluster
                for current_cluster_number, new_cluster in enumerate(
                    additional_clusters
                ):
                    if additional_clusters[current_cluster_number] is None:
                        continue
                    # update the intersection matrices
                    new_row_interesection_ids = [set()] * len(current_clusters)
                    new_row_cardinal_intersection = np.zeros((len(current_clusters),))

                    new_row_interesection_ids[current_cluster_number] = set(
                        new_cluster.tracked_ids
                    )
                    new_row_cardinal_intersection[current_cluster_number] = len(
                        new_cluster.tracked_ids
                    )

                    if current_cluster_number == cluster_number_with_oldest_tracker:
                        intersection_matrix_ids[
                            cluster_number
                        ] = new_row_interesection_ids
                        cardinal_intersection_matrix[
                            cluster_number
                        ] = new_row_cardinal_intersection
                    else:

                        self.clusters.append(new_cluster)
                        # need to create new rows for the new cluster
                        cardinal_intersection_matrix.append(
                            new_row_cardinal_intersection
                        )
                        intersection_matrix_ids.append(new_row_interesection_ids)
            else:
                self.clusters[cluster_number] = cluster
            cluster_number += 1

        # remove empty clusters
        self.clusters = [cluster for cluster in self.clusters if len(cluster) > 0]

        # create new clusters with remaining ids that were not used
        all_ids_in_self_clusters = set(
            flatten_list([cluster.tracked_ids for cluster in self.clusters])
        )
        for current_cluster in current_clusters.clusters:
            difference_ids = set(current_cluster.tracked_ids) - all_ids_in_self_clusters
            if len(difference_ids) > 0:
                new_cluster = Cluster(None, self.last_cluster_fake_id)
                self.last_cluster_fake_id += 1

                for (camera_name, track_id) in difference_ids:
                    new_cluster.tracked_objects[
                        camera_name
                    ] = current_cluster.tracked_objects[camera_name]
                new_cluster.tracked_ids = list(difference_ids)

                self.clusters.append(new_cluster)

        # update clusters age, and assign id to old enough clusters
        cluster_number = 0
        while cluster_number < len(self.clusters):
            cluster = self.clusters[cluster_number]
            cluster.age += 1
            if (cluster.age > self.initialization_delay) and (cluster.id is None):
                cluster.id = self.last_cluster_id
                self.last_cluster_id += 1

            # check that their reid_hit_counter make sense
            new_reid_hit_counter = {}
            for camera_name, track_id in cluster.tracked_ids:
                try:
                    new_reid_hit_counter[camera_name] = cluster.reid_hit_counter[
                        camera_name
                    ]
                except KeyError:
                    new_reid_hit_counter[camera_name] = 0

            cluster.reid_hit_counter = new_reid_hit_counter

            self.clusters[cluster_number] = cluster
            cluster_number += 1

        return [
            cluster
            for cluster in self.clusters
            if ((cluster.id is not None) and (len(cluster.tracked_objects) > 0))
        ]
