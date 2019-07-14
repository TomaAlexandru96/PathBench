"""

The MVC pattern was cloned from https://github.com/wesleywerner/mvc-game-design

"""

import copy
from typing import Type, List, Tuple, Callable, Dict, Any

import torch
from torch import nn
import matplotlib.pyplot as plt

from algorithms.classic.a_star import AStar
from algorithms.algorithm import Algorithm
from algorithms.classic.bug1 import Bug1
from algorithms.classic.bug2 import Bug2
from algorithms.classic.dijkstra import Dijkstra
from algorithms.classic.rrt import RRT
from algorithms.classic.testing.a_star_testing import AStarTesting
from algorithms.basic_testing import BasicTesting
from algorithms.classic.testing.combined_online_lstm_testing import CombinedOnlineLSTMTesting
from algorithms.classic.testing.dijkstra_testing import DijkstraTesting
from algorithms.classic.testing.wavefront_testing import WavefrontTesting
from algorithms.classic.testing.way_point_navigation_testing import WayPointNavigationTesting
from algorithms.classic.wavefront import Wavefront
from algorithms.configuration.configuration import Configuration
from algorithms.lstm.LSTM_CAE_tile_by_tile import CAE, LSTMCAEModel
from algorithms.lstm.LSTM_tile_by_tile import OnlineLSTM, BasicLSTMModule
from algorithms.lstm.MP_CAE_tile_by_tile import MPCAEModel
from algorithms.lstm.MP_tile_by_tile import BasicMPModule, OnlineMP
from algorithms.lstm.a_star_heuristic_augmentation import AStarHeuristicAugmentation
from algorithms.lstm.a_star_waypoint import WayPointNavigation
from algorithms.lstm.combined_online_LSTM import CombinedOnlineLSTM
from algorithms.lstm.map_processing import MapProcessing
from algorithms.lstm.trainer import Trainer
from analyzer.analyzer import Analyzer
from generator.generator import Generator
from maps import Maps
from simulator.models.map import Map
from simulator.services.debug import DebugLevel
from simulator.services.services import Services, GenericServices
from simulator.simulator import Simulator


class MainRunner:
    main_services: Services

    def __init__(self, configuration: Configuration):
        self.main_services: Services = Services(configuration)
        self.run = self.main_services.debug.debug_func(DebugLevel.BASIC)(self.run)

    def run(self):
        """
        model: BasicLSTMModule = self.main_services.resources.model_dir.load("caelstm_section_cae_training_block_map_10000_model")

        def convert_map_sample(mp_name, idx):
            if any(map(lambda t: mp_name in t, model.config["training_data"])):
                mp = self.main_services.resources.maps_dir.load(mp_name + "_10/3")
                features = MapProcessing.extract_features(mp, model.config["data_single_features"])
                features = MapProcessing.combine_features(features).view(model.config["in_dim"]).to(self.main_services.torch.device)
                return features

        def convert_sample(img):
            features = CAE.normalize_data(img)
            converted = model.forward(features).view(model.config["in_dim"]).data.tolist()
            feature_maps = model.encode(features)
            latent = feature_maps[0].view((10, 10)).data.tolist()
            feature_maps = list(map(lambda f: f.squeeze(), feature_maps[1:-1]))
            return features.tolist(), converted, latent, feature_maps

        def plot_map(idx, with_f_map):
            maps = ["uniform_random_fill", "block_map", "house"]
            data = list(filter(lambda x: x is not None, map(lambda m: convert_map_sample(m, idx), maps)))
            plot(data, idx, with_f_map)

        def plot_feature_map(img_name, f_map_idx, f_map):
            if f_map.shape[0] == 8:
                size = 2
                size2 = 4

            if f_map.shape[0] == 16:
                size = 4
                size2 = 4

            if f_map.shape[0] == 32:
                size = 4
                size2 = 8

            if f_map.shape[0] == 64:
                size = 8
                size2 = 8

            fig, axes = plt.subplots(size, size2, figsize=(9, 9))

            for i in range(size):
                for j in range(size2):
                    axes[i][j].imshow(f_map[i + j].tolist(), cmap="gray")
                    axes[i][j].axis('off')

            plt.show()

        def plot(imgs, idx, with_f_map):
            data = list(map(lambda x: convert_sample(x), imgs))

            fig, axes = plt.subplots(len(data), 3, figsize=(9, 3.5 * len(data)))

            if len(data) == 1:
                axes[0].title.set_text('Original')
                axes[0].imshow(data[0][0], cmap="gray_r")

                axes[1].title.set_text('Converted')
                axes[1].imshow(data[0][1], cmap="gray_r")

                axes[2].title.set_text('Latent Space')
                axes[2].imshow(data[0][2], cmap="gray")
            else:
                for i in range(len(data)):
                    axes[i][0].title.set_text('Original')
                    axes[i][0].imshow(data[i][0], cmap="gray_r")

                    axes[i][1].title.set_text('Converted')
                    axes[i][1].imshow(data[i][1], cmap="gray_r")

                    axes[i][2].title.set_text('Latent Space')
                    axes[i][2].imshow(data[i][2], cmap="gray")
            plt.show()

            if with_f_map:
                for i in range(len(data)):
                    name = "map_" + str(idx) + "_" + str(i)
                    for q in range(4):
                        plot_feature_map(name, 3 - q, data[i][3][3 - q])

        plot_map(0, True)
        """

        if self.main_services.settings.generator:
            Generator.main(self)
        elif self.main_services.settings.trainer:
            Trainer.main(self)
        elif self.main_services.settings.analyzer:
            Analyzer.main(self)
        elif self.main_services.settings.load_simulator:
            simulator: Simulator = Simulator(self.main_services)
            simulator.start()

        if self.main_services.settings.clear_cache:
            self.main_services.resources.cache_dir.clear()


generic_services: Services = GenericServices()


def start_main_runner(modify: Callable[[Configuration], None] = None):
    c: Configuration = Configuration()

    # Simulator settings
    c.simulator_graphics = True
    # c.simulator_key_frame_speed, c.simulator_key_frame_skip = 0.0001, 0
    # c.simulator_key_frame_speed, c.simulator_key_frame_skip = 0.0001, 0
    # c.simulator_key_frame_speed, c.simulator_key_frame_skip = 0.0001, 20
    # c.simulator_key_frame_speed, c.simulator_key_frame_skip = 0.0001, 100000
    c.simulator_write_debug_level: DebugLevel = DebugLevel.BASIC

    # Environment settings
    # MAPS
    # c.simulator_grid_display, c.simulator_initial_map = False, "map12"
    # c.simulator_grid_display, c.simulator_initial_map = True, "map10"
    # c.simulator_grid_display, c.simulator_initial_map = False, "map1"
    # c.simulator_grid_display, c.simulator_initial_map = False, "map2"
    # c.simulator_grid_display, c.simulator_initial_map = False, "map3"
    # c.simulator_grid_display, c.simulator_initial_map = False, "map4"
    # c.simulator_grid_display, c.simulator_initial_map = False, Maps.pixel_map_empty
    # c.simulator_grid_display, c.simulator_initial_map = True, Maps.pixel_map_small_obstacles.convert_to_dense_map()
    # c.simulator_grid_display, c.simulator_initial_map = True, Maps.grid_map_no_solution
    # c.simulator_grid_display, c.simulator_initial_map = True, Maps.grid_map_small_one_obstacle3
    # c.simulator_grid_display, c.simulator_initial_map = True, Maps.grid_yt
    # c.simulator_grid_display, c.simulator_initial_map = True, Maps.grid_map_complex_obstacle
    # c.simulator_grid_display, c.simulator_initial_map = True, Maps.grid_map_small_one_obstacle

    # CURRENT MAPS
    # c.simulator_grid_display, c.simulator_initial_map = False, Maps.pixel_map_one_obstacle.convert_to_dense_map()
    # c.simulator_grid_display, c.simulator_initial_map = True, "uniform_random_fill_10/0"
    c.simulator_grid_display, c.simulator_initial_map = True, "block_map_10/6"
    # c.simulator_grid_display, c.simulator_initial_map = True, "house_10/6"
    # c.simulator_grid_display, c.simulator_initial_map = True, "house_10000/2"
    # c.simulator_grid_display, c.simulator_initial_map = True, "block_map_10/3"
    # c.simulator_grid_display, c.simulator_initial_map = True, "uniform_random_fill_10000/25"
    # c.simulator_grid_display, c.simulator_initial_map = True, "house_10000/9"
    # c.simulator_grid_display, c.simulator_initial_map = True, Maps.grid_map_labyrinth
    # c.simulator_grid_display, c.simulator_initial_map = True, Maps.grid_map_labyrinth2
    # c.simulator_grid_display, c.simulator_initial_map = True, Maps.grid_map_one_obstacle.convert_to_dense_map()

    # ALGORITHMS
    # c.simulator_algorithm_type, c.simulator_testing_type = Wavefront, WavefrontTesting
    # c.simulator_algorithm_type, c.simulator_testing_type = Dijkstra, DijkstraTesting
    # c.simulator_algorithm_type, c.simulator_testing_type = Bug1, BasicTesting
    # c.simulator_algorithm_type, c.simulator_testing_type = Bug2, BasicTesting
    # c.simulator_algorithm_type, c.simulator_testing_type = RRT, BasicTesting
    # c.simulator_algorithm_type, c.simulator_testing_type, c.simulator_algorithm_parameters = AStarHeuristicAugmentation, BasicTesting, ([(OnlineLSTM, ([200], {"load_name": "caelstm_section_lstm_training_block_map_10000_model"}))], {})
    # c.simulator_algorithm_type, c.simulator_testing_type, c.simulator_algorithm_parameters = AStarHeuristicAugmentation, BasicTesting, ([(CombinedOnlineLSTM, ([], {}))], {})

    # CURRENT ALGORITHMS
    # c.simulator_algorithm_type, c.simulator_testing_type = AStar, AStarTesting
    # tile_by_tile uniform_random_fill
    # c.simulator_algorithm_type, c.simulator_testing_type, c.simulator_algorithm_parameters = OnlineLSTM, BasicTesting, ([], {"load_name": "tile_by_tile_training_uniform_random_fill_10000_model"})
    # tile_by_tile block_map
    # c.simulator_algorithm_type, c.simulator_testing_type, c.simulator_algorithm_parameters = OnlineLSTM, BasicTesting, ([], {"load_name": "tile_by_tile_training_block_map_10000_model"})
    # tile_by_tile house
    # c.simulator_algorithm_type, c.simulator_testing_type, c.simulator_algorithm_parameters = OnlineLSTM, BasicTesting, ([], {"load_name": "tile_by_tile_training_house_10000_model"})
    # tile_by_tile random & block
    # c.simulator_algorithm_type, c.simulator_testing_type, c.simulator_algorithm_parameters = OnlineLSTM, BasicTesting, ([], {"load_name": "tile_by_tile_training_uniform_random_fill_10000_block_map_10000_model"})
    # tile_by_tile random & block & house
    # c.simulator_algorithm_type, c.simulator_testing_type, c.simulator_algorithm_parameters = OnlineLSTM, BasicTesting, ([], {"load_name": "tile_by_tile_training_uniform_random_fill_10000_block_map_10000_house_10000_model"})

    # cae uniform_random_fill
    # c.simulator_algorithm_type, c.simulator_testing_type, c.simulator_algorithm_parameters = OnlineLSTM, BasicTesting, ([], {"load_name": "old_caelstm_section_lstm_training_uniform_random_fill_10000_model"})
    # c.simulator_algorithm_type, c.simulator_testing_type, c.simulator_algorithm_parameters = OnlineLSTM, BasicTesting, ([], {"load_name": "caelstm_section_lstm_training_uniform_random_fill_10000_model"})
    # cae block_map
    # c.simulator_algorithm_type, c.simulator_testing_type, c.simulator_algorithm_parameters = OnlineLSTM, BasicTesting, ([], {"load_name": "caelstm_section_lstm_training_block_map_10000_model"})
    # cae house
    # c.simulator_algorithm_type, c.simulator_testing_type, c.simulator_algorithm_parameters = OnlineLSTM, BasicTesting, ([], {"load_name": "caelstm_section_lstm_training_house_10000_model"})
    # cae random & block
    # c.simulator_algorithm_type, c.simulator_testing_type, c.simulator_algorithm_parameters = OnlineLSTM, BasicTesting, ([], {"load_name": "caelstm_section_lstm_training_uniform_random_fill_10000_block_map_10000_model"})
    # cae random & block & house
    # c.simulator_algorithm_type, c.simulator_testing_type, c.simulator_algorithm_parameters = OnlineLSTM, BasicTesting, ([], {"load_name": "caelstm_section_lstm_training_uniform_random_fill_10000_block_map_10000_house_10000_model"})

    # c.simulator_algorithm_type, c.simulator_testing_type = CombinedOnlineLSTM, CombinedOnlineLSTMTesting
    # c.simulator_algorithm_type, c.simulator_testing_type, c.simulator_algorithm_parameters = WayPointNavigation, WayPointNavigationTesting, ([], {"global_kernel": (OnlineLSTM, ([], {"load_name": "caelstm_section_lstm_training_block_map_10000_model"}))})
    # c.simulator_algorithm_type, c.simulator_testing_type, c.simulator_algorithm_parameters = WayPointNavigation, WayPointNavigationTesting, ([], {"global_kernel": (CombinedOnlineLSTM, ([], {})), "global_kernel_max_it": 20})

    # c.simulator_algorithm_type, c.simulator_testing_type, c.simulator_algorithm_parameters = OnlineMP, BasicTesting, ([], {"load_name": "tile_by_tile_training_uniform_random_fill_10000_model"})
    c.simulator_algorithm_type, c.simulator_testing_type, c.simulator_algorithm_parameters = OnlineMP, BasicTesting, ([], {"load_name": "mp_caelstm_section_mp_training_uniform_random_fill_10000_block_map_10000_house_10000_model"})

    # ** In order of execution **

    # Generator
    c.generator = False

    # generation
    c.generator_gen_type = "house"
    c.generator_nr_of_examples = 0

    # labelling
    c.generator_labelling_atlases = [] # ["uniform_random_fill_10000", "block_map_10000", "house_10000"]  # "house_10", "block_map_10", "uniform_random_fill_10"
    c.generator_labelling_features = [
        "agent_position",
        "direction_to_goal",
        "distance_to_goal",
        "raycast_8",
        "agent_goal_angle",
        "valid_moves",
        "direction_to_goal_normalized",
        "distance_to_goal_normalized",
        "raycast_8_normalized",
        # "local_map",
        # "global_map",
    ]
    c.generator_single_labelling_features = [
        "global_map",
    ]
    c.generator_labelling_labels = [
        "next_position",
        "next_position_index",
    ]
    c.generator_single_labelling_labels = [
        "global_map",
    ]

    # augmentation
    c.generator_aug_labelling_features = []  # "cae_encoded_features"
    c.generator_aug_labelling_labels = []
    c.generator_aug_single_labelling_features = []
    c.generator_aug_single_labelling_labels = []

    c.generator_modify = None
    """
    def f() -> Tuple[str, Callable[[Map], Map]]:
        def f(mp):
            mp.reset()
            mp.move_agent(Point(24, 10), True)
            return mp
        return "training_10000/0", f
    c.generator_modify = f
    """

    # Trainer
    c.trainer = False
    # c.trainer_model = BasicLSTMModule
    # c.trainer_model = BasicMPModule
    c.trainer_model = MPCAEModel
    # c.trainer_model = CAE
    # c.trainer_model = LSTMCAEModel

    c.trainer_pre_process_data_only = False
    c.trainer_bypass_and_replace_pre_processed_cache = False

    c.trainer_custom_config = None
    """
    c.trainer_custom_config = {}
    """

    # Custom behaviour settings
    c.analyzer = True

    # Simulator
    c.load_simulator = True

    # Cache
    c.clear_cache = False

    if modify:
        modify(c)

    MainRunner(c).run()


if __name__ == '__main__':
    start_main_runner()
