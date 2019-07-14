import copy
import math
from typing import List, Any, Tuple, Dict

import torch
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from pandas.tests.extension.numpy_.test_numpy_nested import np
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, TensorDataset, Subset

from algorithms.basic_testing import BasicTesting
from algorithms.configuration.maps.map import Map
from algorithms.lstm.LSTM_CAE_tile_by_tile import CAE
from algorithms.lstm.LSTM_tile_by_tile import BasicLSTMModule, OnlineLSTM
from algorithms.lstm.ML_model import MLModel, EvaluationResults
from algorithms.lstm.MP_tile_by_tile import BasicMPModule
from algorithms.lstm.map_processing import MapProcessing
from simulator.services.services import Services
from torchvision import transforms, datasets


class MPCAEModel(BasicMPModule):
    __cached_encoded_map: torch.Tensor

    def __init__(self, services: Services, config: Dict[str, any]):
        super().__init__(services, config)

        self.__encoder = self.__get_encoder()
        self.__cached_encoded_map = None

        if "with_init_fn" in self.config and self.config["with_init_fn"]:
            self.fc = nn.Linear(in_features=114, out_features=self.config["lstm_input_size"])
            self.bn = nn.BatchNorm1d(num_features=self.config["lstm_input_size"])
            self.dp = nn.Dropout()

    def pre_process_data(self) -> Tuple[Dataset, Dataset]:
        data_features, data_labels = super().pre_process_data()

        if "agent_position" in self.config["data_features"]:
            data_features.subsets[0].data[:, :, -2:] /= 64

        return data_features, data_labels

    def init_running_algorithm(self, mp: Map) -> None:
        super().init_running_algorithm(mp)

        raw_features_img: Dict[str, torch.Tensor] = MapProcessing.extract_features(mp, self.config["data_single_features"])
        transformed_features_img: torch.Tensor = MapProcessing.combine_features(raw_features_img)
        self.__cached_encoded_map = self.__encode_image(transformed_features_img).view((1, -1))

    def __get_encoder(self) -> CAE:
        if "custom_encoder" in self.config and self.config["custom_encoder"]:
            encoder_name: str = self.config["custom_encoder"]
        else:
            encoder_name: str = "caelstm_section_cae_" + self.training_suffix() + "_model"
        return self._services.resources.model_dir.load(encoder_name)

    def __encode_image(self, img: torch.Tensor, seq_size: int = 1) -> torch.Tensor:
        enc_in = CAE.normalize_data(img)
        enc_out = self.__encoder.encode(enc_in)[0].unsqueeze(1).repeat(1, seq_size, 1)
        return enc_out

    def __combine_features(self, x: torch.Tensor, img: torch.Tensor) -> torch.Tensor:
        x = x.to(self._services.torch.device)
        img = img.to(self._services.torch.device)
        enc_out = self.__encode_image(img, x.shape[1])
        return torch.cat((x, enc_out), 2)

    def batch_start(self, inputs: Tuple[Tuple, Tuple], labels: Tuple[torch.Tensor, torch.Tensor]) -> \
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # augment data with encoder result
        x_concat = self.__combine_features(inputs[0][0], inputs[1][0])
        out = self.forward(x_concat).view(-1, 8)
        ls = labels[0].view(-1)
        l: torch.Tensor = self.config["loss"](out, ls)
        return l, out, ls

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        init_shape = x.shape
        if "with_init_fn" in self.config and self.config["with_init_fn"]:
            x = F.relu(self.dp(self.bn(self.fc(x.view(-1, self.config["input_size"]))))).view(init_shape[0], init_shape[1], -1)
        return super().forward(x)

    # cache obstacle once
    def forward_features(self, mp: Map) -> torch.Tensor:
        raw_features_x: Dict[str, torch.Tensor] = MapProcessing.extract_features(mp, self.config["data_features"])
        transformed_features_x: torch.Tensor = MapProcessing.combine_features(raw_features_x)

        if "agent_position" in self.config["data_features"]:
            transformed_features_x[-2:-1] /= mp.size.width
            transformed_features_x[-1:] /= mp.size.height

        inp = torch.cat((transformed_features_x.view((1, -1)), self.__cached_encoded_map), 1)
        res = self.forward(inp)
        _, mov_idx = torch.max(res.squeeze(), 0)
        return Map.EIGHT_POINTS_MOVE_VECTOR[mov_idx].to_tensor()

    @staticmethod
    def get_config() -> Dict[str, Any]:
        return {
              "data_features": [
                  "raycast_8_normalized",
                  "distance_to_goal_normalized",
                  "direction_to_goal_normalized",
                  "agent_goal_angle",
              ],
              "data_single_features": [
                  "global_map",
              ],
              "data_labels": [
                  "next_position_index",
              ],
              "custom_encoder": None, # "caelstm_section_cae_training_uniform_random_fill_10000_block_map_10000_house_10000_model",
              "save_name": "mp_caelstm_section_mp",
              "training_data": [
                  "training_uniform_random_fill_10000",
                  "training_block_map_10000",
                  "training_house_10000",
              ], # training_uniform_random_fill_10000_block_map_10000_house_10000, "training_uniform_random_fill_10000_block_map_10000", "training_house_10000", "training_uniform_random_fill_10000", "training_block_map_10000",
              "epochs": 10,
              "num_layers": 2,
              #"with_init_fn": False,
              #"input_size": 114,
              "lstm_input_size": 112,
              "lstm_output_size": 8,
              "loss": nn.CrossEntropyLoss(),
              "optimizer": lambda model: torch.optim.Adam(model.parameters(), lr=0.01),
        }
