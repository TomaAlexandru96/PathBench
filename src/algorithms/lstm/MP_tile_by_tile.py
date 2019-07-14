from typing import Dict, Optional, Any, Tuple, List

import torch
from torch import nn
from torch.nn import LSTM
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, PackedSequence
from torch.utils.data import Dataset, TensorDataset

from algorithms.algorithm import Algorithm
from algorithms.basic_testing import BasicTesting
from algorithms.configuration.entities.goal import Goal
from algorithms.configuration.maps.map import Map
from algorithms.lstm.LSTM_tile_by_tile import BasicLSTMModule
from algorithms.lstm.ML_model import MLModel, SingleTensorDataset, PackedDataset
from algorithms.lstm.map_processing import MapProcessing
from simulator.services.services import Services
from simulator.views.map_displays.entities_map_display import EntitiesMapDisplay
from simulator.views.map_displays.online_lstm_map_display import OnlineLSTMMapDisplay
from structures import Point


# 1. torch.nn.utils.rnn.pad_packed_sequence(out) - Done
# 2. try to do batch second - No improvement
# 3. add none action - Not needed
# 4. add previous action - Quite hard to implement with packed
class BasicMPModule(MLModel):
    _hidden_state: Tuple[torch.Tensor, torch.Tensor]
    _lstm_layer: LSTM

    def __init__(self, services: Services, config: Dict[str, any]):
        super().__init__(services, config)

        self._hidden_state = None
        self._normalisation_layer1 = nn.BatchNorm1d(num_features=self.config["lstm_input_size"])
        self._fc_core = nn.Linear(in_features=self.config["lstm_input_size"], out_features=self.config["lstm_output_size"])
        self._normalisation_layer2 = nn.BatchNorm1d(num_features=self.config["lstm_output_size"])
        self._fc = nn.Linear(in_features=self.config["lstm_output_size"], out_features=self.config["lstm_output_size"])

    def init_running_algorithm(self, mp: Map) -> None:
        pass

    def pre_process_data(self) -> Tuple[Dataset, Dataset]:
        data_features, data_labels = super().pre_process_data()
        data_labels.data = data_labels.data.long()
        return data_features, data_labels

    def batch_start(self, inputs: Tuple[torch.Tensor, torch.Tensor], labels: Tuple[torch.Tensor, torch.Tensor]) -> \
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self.to(self._services.torch.device)
        inp = inputs[0].to(self._services.torch.device)
        ls = labels[0].view((-1)).to(self._services.torch.device)

        out = self.forward(inp).view(-1, 8)
        l: torch.Tensor = self.config["loss"](out, ls)
        return l, out, ls

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normalized_x = self._normalisation_layer1(x.view((-1, x.shape[-1])))
        out_fc_core = self._fc_core.forward(normalized_x)
        normalized_lstm_out = self._normalisation_layer2(out_fc_core)
        out = self._fc(normalized_lstm_out)
        return out

    def forward_features(self, mp: Map) -> torch.Tensor:
        raw_features: Dict[str, torch.Tensor] = MapProcessing.extract_features(mp, self.config["data_features"])
        transformed_features: torch.Tensor = MapProcessing.combine_features(raw_features)

        inp = transformed_features.view((1, -1))
        res = self.forward(inp)
        _, mov_idx = torch.max(res.squeeze(), 0)
        return Map.EIGHT_POINTS_MOVE_VECTOR[mov_idx].to_tensor()

    @staticmethod
    def get_config() -> Dict[str, Any]:
        return {
            "data_features": [
                "distance_to_goal_normalized",
                "raycast_8_normalized",
                "direction_to_goal_normalized",
                "agent_goal_angle",
            ],
            "data_labels": [
                "next_position_index",
            ],
            "save_name": "mp_tile_by_tile",
            "training_data": [
                "training_uniform_random_fill_10000",
                "training_block_map_10000",
                "training_house_10000",
            ],
            # training_uniform_random_fill_10000_block_map_10000_house_10000, "training_uniform_random_fill_10000_block_map_10000", "training_house_10000", "training_uniform_random_fill_10000", "training_block_map_10000",
            "epochs": 100,
            "num_layers": 2,
            "lstm_input_size": 12,
            "lstm_output_size": 8,
            "loss": nn.CrossEntropyLoss(),  # nn.MSELoss(),
            "optimizer": lambda model: torch.optim.Adam(model.parameters(), lr=0.01),
        }


class OnlineMP(Algorithm):
    _load_name: str
    _max_it: float

    def __init__(self, services: Services, testing: BasicTesting = None, max_it: float = float('inf'),
                 load_name: str = None):
        super().__init__(services, testing)

        if not load_name:
            raise NotImplementedError("load_name needs to be supplied")

        self._load_name = load_name
        self._max_it = max_it

    def set_display_info(self):
        return super().set_display_info() + [
            OnlineLSTMMapDisplay(self._services)
        ]

    # noinspection PyUnusedLocal
    def _find_path_internal(self) -> None:
        model: BasicMPModule = self._services.resources.model_dir.load(self._load_name)
        model.init_running_algorithm(self._get_grid())
        history_frequency: Dict[Point, int] = {}
        last_agent_pos: Point = self._get_grid().agent.position
        stuck_threshold = 5

        it = 0
        while it < self._max_it:
            # goal reached if radius intersects
            if self._get_grid().is_agent_in_goal_radius():
                self.move_agent(self._get_grid().goal.position)
                break

            next_move: Point = Point.from_tensor(model.forward_features(self._get_grid()))
            self.move_agent(self._get_grid().apply_move(next_move, self._get_grid().agent.position))

            last_agent_pos = self._get_grid().agent.position
            new_freq: int = history_frequency.get(last_agent_pos, 0) + 1
            history_frequency[last_agent_pos] = new_freq

            # fail safe
            if new_freq >= stuck_threshold:
                break

            it += 1
            self.key_frame()
