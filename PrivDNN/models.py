import enum
import torch
import ctypes

import torch.nn as nn
import torch.nn.functional as F


class WorkMode(enum.Enum):
    normal = 1
    split = 2
    recover = 3
    cipher = 4


class CppWorkMode(enum.Enum):
    separate = 1
    remove = 2
    full = 3


class MaskedLayer(nn.Module):
    def __init__(self, layer, layer_index):
        super(MaskedLayer, self).__init__()

        self.layer = layer
        self.layer_index = layer_index

        # remove
        self.neurons_to_remove = None
        self.remove_mask = None

        # seperate
        self.weight_backup = None
        self.seperate_weight = None
        self.last_layer_neurons_subset = None
        self.current_layer_neurons_subset = None

    def forward(self, input):
        output = None
        if not self.neurons_to_remove is None:
            # remove
            output = self.layer(input)
            if self.remove_mask is None:
                layer_shape = output.shape[1:]
                mask = torch.ones(layer_shape).long().cuda()
                for index in self.neurons_to_remove:
                    mask[index] = (
                        torch.zeros(layer_shape[1:]).long().cuda()
                        if len(layer_shape) > 1
                        else 0
                    )
                self.remove_mask = mask

            output = self.remove_mask * output
        elif not (
            self.last_layer_neurons_subset is None
            and self.current_layer_neurons_subset is None
        ):
            # seperate
            if self.seperate_weight is None:
                output_channel_count = self.layer.out_channels
                with torch.no_grad():
                    current_layer_seperate = [
                        x
                        for x in range(output_channel_count)
                        if x not in self.current_layer_neurons_subset
                    ]

                    for current_layer_neuron in current_layer_seperate:
                        for last_layer_neuron in self.last_layer_neurons_subset:
                            self.layer.weight[current_layer_neuron][
                                last_layer_neuron
                            ] = 0
                    self.seperate_weight = self.layer.weight

            output = self.layer(input)
        else:
            # normal
            output = self.layer(input)

        return output

    def set_neurons_to_remove(self, neurons_to_remove):
        if (
            self.neurons_to_remove is None
            or self.neurons_to_remove != neurons_to_remove
        ):
            self.neurons_to_remove = neurons_to_remove
            self.remove_mask = None

    def clear_neurons_to_remove(self):
        self.neurons_to_remove = None
        self.remove_mask = None

    def set_neurons_to_separate(
        self, last_layer_neurons_subset, current_layer_neurons_subset
    ):
        self.weight_backup = self.layer.weight
        if (
            self.last_layer_neurons_subset is None
            and self.current_layer_neurons_subset is None
        ) or (
            self.last_layer_neurons_subset != last_layer_neurons_subset
            and self.current_layer_neurons_subset != current_layer_neurons_subset
        ):
            self.last_layer_neurons_subset = last_layer_neurons_subset
            self.current_layer_neurons_subset = current_layer_neurons_subset
            self.seperate_weight = None

    def clear_neurons_to_separate(self):
        # unreliable!
        if not self.weight_backup is None:
            self.last_layer_neurons_subset = None
            self.current_layer_neurons_subset = None
            self.seperate_weight = None
            self.layer.weight = self.weight_backup
            self.weight_backup = None


class SplitNet(nn.Module):
    def __init__(self):
        super(SplitNet, self).__init__()

        self.work_mode = WorkMode.normal
        self.cpp_work_mode = CppWorkMode.separate
        self.selected_neurons = {}

    def get_layers_list(self, include_fc_layers=False):
        return []

    def set_layers_on_cuda(self):
        layers_list = self.get_layers_list(True)
        for layers in layers_list:
            if isinstance(layers, nn.Linear) or isinstance(layers, nn.BatchNorm2d):
                layers.cuda()
            else:
                for layer in layers:
                    layer.cuda()

    def copy_parameters_to_split_model(self):
        layers = self.get_layers_list()
        for layer in layers:
            for i in range(1, len(layer)):
                weight_shape = [1] + list(layer[0].layer.weight[i - 1].shape)
                layer[i].layer.weight = torch.nn.Parameter(
                    torch.reshape(layer[0].layer.weight[i - 1], weight_shape)
                )
                layer[i].layer.bias = torch.nn.Parameter(
                    torch.reshape(layer[0].layer.bias[i - 1], [1])
                )

    def _get_trained_data_pointer(self):
        layers = self.get_layers_list(include_fc_layers=True)
        trained_data_list = []
        for layer in layers:
            if isinstance(layer, list):
                trained_data_list.extend(torch.flatten(layer[0].layer.weight).tolist())
                trained_data_list.extend(torch.flatten(layer[0].layer.bias).tolist())
            else:
                # fc weight needs to be transposed, but we don't do that here
                trained_data_list.extend(torch.flatten(layer.weight).tolist())
                trained_data_list.extend(torch.flatten(layer.bias).tolist())

        trained_data_pointer = (ctypes.c_double * len(trained_data_list))(
            *trained_data_list
        )
        return trained_data_pointer

    def _conv(self, layers, input):
        if self.work_mode == WorkMode.normal:
            return layers[0](input)
        elif self.work_mode == WorkMode.split or self.work_mode == WorkMode.recover:
            outputs = []
            for i in range(1, len(layers)):
                outputs.append(layers[i](input))

            output = torch.cat(outputs, dim=1)
            return output
        else:
            print(self.work_mode)
            quit()

    def _activate(self, input, activates):
        if self.work_mode == WorkMode.normal or self.work_mode == WorkMode.split:
            return activates[0](input)
        elif self.work_mode == WorkMode.recover:
            return activates[1](input)
        else:
            print(self.work_mode)
            quit()


class SplitMNISTNet(SplitNet):
    def __init__(self):
        super().__init__()

        self.conv1_layers = [
            MaskedLayer(nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5), 1)
        ] + [
            MaskedLayer(nn.Conv2d(in_channels=1, out_channels=1, kernel_size=5), 1)
            for _ in range(6)
        ]

        self.conv2_layers = [
            MaskedLayer(nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5), 2)
        ] + [
            MaskedLayer(nn.Conv2d(in_channels=6, out_channels=1, kernel_size=5), 2)
            for _ in range(16)
        ]

        self.avg_pool_layer = nn.AvgPool2d(kernel_size=2)

        self.fc1_layer = nn.Linear(in_features=256, out_features=120)
        self.fc2_layer = nn.Linear(in_features=120, out_features=84)
        self.fc3_layer = nn.Linear(in_features=84, out_features=10)

        self.work_mode = WorkMode.normal
        self.cpp_work_mode = CppWorkMode.separate

    def forward(self, input):
        output = None
        torch.set_printoptions(
            precision=8,
            threshold=None,
            edgeitems=None,
            linewidth=None,
            profile="full",
        )

        if (
            self.work_mode == WorkMode.normal
            or self.work_mode == WorkMode.split
            or self.work_mode == WorkMode.recover
        ):
            conv1_output = self._conv(self.conv1_layers, input)
            avg_pool1_output = torch.square(self.avg_pool_layer(conv1_output))

            conv2_output = self._conv(self.conv2_layers, avg_pool1_output)
            avg_pool2_output = torch.square(self.avg_pool_layer(conv2_output))

            fc1_input = avg_pool2_output.reshape(avg_pool2_output.shape[0], -1)
            fc1_output = torch.square(self.fc1_layer(fc1_input))
            fc2_output = torch.square(self.fc2_layer(fc1_output))
            output = self.fc3_layer(fc2_output)
        elif self.work_mode == WorkMode.cipher:
            work_mode = int(  # 0 separate, 1 remove, 2 full
                0
                if self.cpp_work_mode == CppWorkMode.separate
                else 1
                if self.cpp_work_mode == CppWorkMode.remove
                else 2,
            )

            client_library = ctypes.CDLL("../seal/output/lib/libclient.so")
            server_library = ctypes.CDLL("../seal/output/lib/libserver.so")

            cpp_is_file_complete = server_library.is_file_complete
            cpp_is_file_complete.argtypes = [
                ctypes.c_char_p,
                ctypes.c_int,
            ]
            cpp_is_file_complete.restype = ctypes.c_bool

            if not cpp_is_file_complete(
                b"MNIST",
                work_mode,
            ):
                cpp_save_trained_data = server_library.save_trained_data
                cpp_save_trained_data.argtypes = [
                    ctypes.c_char_p,
                    ctypes.POINTER(ctypes.c_double),
                    ctypes.c_int,
                ]

                trained_data_pointer = self._get_trained_data_pointer()
                cpp_save_trained_data(
                    b"MNIST",
                    trained_data_pointer,
                    work_mode,
                )

            cpp_worker = client_library.worker
            cpp_worker.argtypes = [
                ctypes.c_char_p,
                ctypes.c_int,
                ctypes.POINTER(ctypes.c_double),
                ctypes.c_int,
            ]

            input_list = torch.flatten(input).tolist()
            input_pointer = (ctypes.c_double * len(input_list))(*input_list)
            cpp_worker(b"MNIST", input.shape[0], input_pointer, work_mode)

            cpp_get_result = server_library.get_result
            cpp_get_result.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int]
            cpp_get_result.restype = ctypes.POINTER(ctypes.c_double)

            if (
                self.cpp_work_mode == CppWorkMode.separate
                or self.cpp_work_mode == CppWorkMode.remove
            ):
                conv2_output = cpp_get_result(
                    b"MNIST",
                    input.shape[0],
                    work_mode,
                )

                conv2_output = [
                    conv2_output[i] for i in range(input.shape[0] * 16 * 8 * 8)
                ]
                conv2_output = torch.reshape(
                    torch.FloatTensor(conv2_output), [input.shape[0], 16, 8, 8]
                ).cuda()

                avgpool2_output = torch.square(self.avg_pool_layer(conv2_output))
                fc1_input = avgpool2_output.view(-1, 256)
                fc1_output = torch.square(self.fc1_layer(fc1_input))
                fc2_output = torch.square(self.fc2_layer(fc1_output))
                output = self.fc3_layer(fc2_output)
            else:
                output = cpp_get_result(
                    b"MNIST",
                    input.shape[0],
                    work_mode,
                )
                output = [output[i] for i in range(input.shape[0] * 10)]
                output = torch.reshape(
                    torch.FloatTensor(output), [input.shape[0], 10]
                ).cuda()
        else:
            raise Exception("SplitMNISTNet unknown work mode")

        return output

    def get_layers_list(self, include_fc_layers=False):
        layers_list = [
            self.conv1_layers,
            self.conv2_layers,
        ]

        if include_fc_layers:
            layers_list += [self.fc1_layer, self.fc2_layer, self.fc3_layer]

        return layers_list


class SplitEMNISTNet(SplitNet):
    def __init__(self):
        super().__init__()

        self.conv1_layers = [
            MaskedLayer(nn.Conv2d(in_channels=1, out_channels=10, kernel_size=5), 1)
        ] + [
            MaskedLayer(nn.Conv2d(in_channels=1, out_channels=1, kernel_size=5), 1)
            for _ in range(10)
        ]

        self.conv2_layers = [
            MaskedLayer(nn.Conv2d(in_channels=10, out_channels=20, kernel_size=5), 2)
        ] + [
            MaskedLayer(nn.Conv2d(in_channels=10, out_channels=1, kernel_size=5), 2)
            for _ in range(20)
        ]

        self.avg_pool_layer = nn.AvgPool2d(kernel_size=2)

        self.fc1_layer = nn.Linear(in_features=320, out_features=120)
        self.fc2_layer = nn.Linear(in_features=120, out_features=84)
        self.fc3_layer = nn.Linear(in_features=84, out_features=27)

        self.work_mode = WorkMode.normal
        self.cpp_work_mode = CppWorkMode.separate

    def forward(self, input):
        output = None
        torch.set_printoptions(
            precision=8,
            threshold=None,
            edgeitems=None,
            linewidth=None,
            profile="full",
        )

        if (
            self.work_mode == WorkMode.normal
            or self.work_mode == WorkMode.split
            or self.work_mode == WorkMode.recover
        ):
            conv1_output = self._conv(self.conv1_layers, input)
            avg_pool1_output = torch.square(self.avg_pool_layer(conv1_output))

            conv2_output = self._conv(self.conv2_layers, avg_pool1_output)
            avg_pool2_output = torch.square(self.avg_pool_layer(conv2_output))

            fc1_input = avg_pool2_output.reshape(avg_pool2_output.shape[0], -1)
            fc1_output = torch.square(self.fc1_layer(fc1_input))
            fc2_output = torch.square(self.fc2_layer(fc1_output))
            output = self.fc3_layer(fc2_output)
        elif self.work_mode == WorkMode.cipher:
            pass
        else:
            raise Exception("SplitMNISTNet unknown work mode")

        return output

    def get_layers_list(self, include_fc_layers=False):
        layers_list = [
            self.conv1_layers,
            self.conv2_layers,
        ]

        if include_fc_layers:
            layers_list += [self.fc1_layer, self.fc2_layer, self.fc3_layer]

        return layers_list


class SplitGTSRBNet(SplitNet):
    def __init__(self):
        super().__init__()

        self.conv1_layers = [
            MaskedLayer(
                nn.Conv2d(in_channels=3, out_channels=96, kernel_size=5, padding=1),
                1,
            )
        ] + [
            MaskedLayer(
                nn.Conv2d(in_channels=3, out_channels=1, kernel_size=5, padding=1),
                1,
            )
            for _ in range(96)
        ]
        self.batch_normal1_layer = nn.BatchNorm2d(96)

        self.conv2_layers = [
            MaskedLayer(
                nn.Conv2d(in_channels=96, out_channels=256, kernel_size=3, padding=1),
                2,
            )
        ] + [
            MaskedLayer(
                nn.Conv2d(in_channels=96, out_channels=1, kernel_size=3, padding=1),
                2,
            )
            for _ in range(256)
        ]
        self.batch_normal2_layer = nn.BatchNorm2d(256)

        self.conv3_layers = [
            MaskedLayer(
                nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, padding=1),
                3,
            )
        ] + [
            MaskedLayer(
                nn.Conv2d(in_channels=256, out_channels=1, kernel_size=3, padding=1),
                3,
            )
            for _ in range(384)
        ]

        self.conv4_layers = [
            MaskedLayer(
                nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, padding=1),
                4,
            )
        ] + [
            MaskedLayer(
                nn.Conv2d(in_channels=384, out_channels=1, kernel_size=3, padding=1),
                4,
            )
            for _ in range(384)
        ]

        self.conv5_layers = [
            MaskedLayer(
                nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, padding=1),
                5,
            )
        ] + [
            MaskedLayer(
                nn.Conv2d(in_channels=384, out_channels=1, kernel_size=3, padding=1),
                5,
            )
            for _ in range(256)
        ]

        self.max_pool_layer = nn.MaxPool2d(kernel_size=2)
        self.avg_pool_layer = nn.AvgPool2d(kernel_size=2)
        self.dropout = nn.Dropout(p=0.5)

        self.fc1_layer = nn.Linear(in_features=2304, out_features=512)
        self.fc2_layer = nn.Linear(in_features=512, out_features=128)
        self.fc3_layer = nn.Linear(in_features=128, out_features=43)

    def forward(self, input):
        output = None
        torch.set_printoptions(
            precision=8,
            threshold=None,
            edgeitems=None,
            linewidth=None,
            profile="full",
        )

        if (
            self.work_mode == WorkMode.normal
            or self.work_mode == WorkMode.split
            or self.work_mode == WorkMode.recover
        ):
            conv1_output = self._conv(self.conv1_layers, input)
            conv1_output = self.batch_normal1_layer(conv1_output)
            conv1_output = self._activate(conv1_output, [torch.square, F.relu])
            avg_pool1_output = self.avg_pool_layer(conv1_output)

            conv2_output = self._conv(self.conv2_layers, avg_pool1_output)
            conv2_output = self.batch_normal2_layer(conv2_output)
            conv2_output = F.relu(conv2_output)
            max_pool2_output = self.max_pool_layer(conv2_output)

            # conv3_output = self._conv(self.conv3_layers, max_pool2_output)
            conv3_output = self.conv3_layers[0](max_pool2_output)
            conv3_output = F.relu(conv3_output)

            # conv4_output = self._conv(self.conv4_layers, conv3_output)
            conv4_output = self.conv4_layers[0](conv3_output)
            conv4_output = F.relu(conv4_output)

            # conv5_output = self._conv(self.conv5_layers, conv4_output)
            conv5_output = self.conv5_layers[0](conv4_output)
            conv5_output = F.relu(conv5_output)
            max_pool5_output = self.max_pool_layer(conv5_output)

            fc1_input = max_pool5_output.reshape(max_pool5_output.shape[0], -1)
            fc1_output = self.dropout(F.relu(self.fc1_layer(fc1_input)))
            fc2_output = self.dropout(F.relu(self.fc2_layer(fc1_output)))
            output = self.fc3_layer(fc2_output)
        else:
            pass

        return output

    def get_layers_list(self, include_fc_layers=False):
        layers_list = [
            self.conv1_layers,
            self.conv2_layers,
            self.conv3_layers,
            self.conv4_layers,
            self.conv5_layers,
        ]

        if include_fc_layers:
            layers_list.extend(
                [
                    self.batch_normal1_layer,
                    self.batch_normal2_layer,
                    self.fc1_layer,
                    self.fc2_layer,
                    self.fc3_layer,
                ]
            )

        return layers_list


class SplitCIFAR10Net(SplitNet):
    def __init__(self):
        super().__init__()

        self.conv1_layers = [
            MaskedLayer(
                nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1), 1
            )
        ] + [
            MaskedLayer(
                nn.Conv2d(in_channels=3, out_channels=1, kernel_size=3, padding=1), 1
            )
            for _ in range(64)
        ]
        self.batch_normal1_layer = nn.BatchNorm2d(64)

        self.conv2_layers = [
            MaskedLayer(
                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1), 2
            )
        ] + [
            MaskedLayer(
                nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, padding=1), 2
            )
            for _ in range(64)
        ]
        self.batch_normal2_layer = nn.BatchNorm2d(64)

        self.conv3_layers = [
            MaskedLayer(
                nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
                3,
            )
        ] + [
            MaskedLayer(
                nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, padding=1), 3
            )
            for _ in range(128)
        ]
        self.batch_normal3_layer = nn.BatchNorm2d(128)

        self.conv4_layers = [
            MaskedLayer(
                nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
                4,
            )
        ] + [
            MaskedLayer(
                nn.Conv2d(in_channels=128, out_channels=1, kernel_size=3, padding=1), 4
            )
            for _ in range(128)
        ]
        self.batch_normal4_layer = nn.BatchNorm2d(128)

        self.conv5_layers = [
            MaskedLayer(
                nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
                5,
            )
        ] + [
            MaskedLayer(
                nn.Conv2d(in_channels=128, out_channels=1, kernel_size=3, padding=1), 5
            )
            for _ in range(256)
        ]
        self.batch_normal5_layer = nn.BatchNorm2d(256)

        self.conv6_layers = [
            MaskedLayer(
                nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
                6,
            )
        ] + [
            MaskedLayer(
                nn.Conv2d(in_channels=256, out_channels=1, kernel_size=3, padding=1), 6
            )
            for _ in range(256)
        ]
        self.batch_normal6_layer = nn.BatchNorm2d(256)

        self.conv7_layers = [
            MaskedLayer(
                nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
                7,
            )
        ] + [
            MaskedLayer(
                nn.Conv2d(in_channels=256, out_channels=1, kernel_size=3, padding=1), 7
            )
            for _ in range(256)
        ]
        self.batch_normal7_layer = nn.BatchNorm2d(256)

        self.conv8_layers = [
            MaskedLayer(
                nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
                8,
            )
        ] + [
            MaskedLayer(
                nn.Conv2d(in_channels=256, out_channels=1, kernel_size=3, padding=1), 8
            )
            for _ in range(512)
        ]
        self.batch_normal8_layer = nn.BatchNorm2d(512)

        self.conv9_layers = [
            MaskedLayer(
                nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
                9,
            )
        ] + [
            MaskedLayer(
                nn.Conv2d(in_channels=512, out_channels=1, kernel_size=3, padding=1), 9
            )
            for _ in range(512)
        ]
        self.batch_normal9_layer = nn.BatchNorm2d(512)

        self.conv10_layers = [
            MaskedLayer(
                nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
                10,
            )
        ] + [
            MaskedLayer(
                nn.Conv2d(in_channels=512, out_channels=1, kernel_size=3, padding=1), 10
            )
            for _ in range(512)
        ]
        self.batch_normal10_layer = nn.BatchNorm2d(512)

        self.conv11_layers = [
            MaskedLayer(
                nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
                11,
            )
        ] + [
            MaskedLayer(
                nn.Conv2d(in_channels=512, out_channels=1, kernel_size=3, padding=1), 11
            )
            for _ in range(512)
        ]
        self.batch_normal11_layer = nn.BatchNorm2d(512)

        self.conv12_layers = [
            MaskedLayer(
                nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
                12,
            )
        ] + [
            MaskedLayer(
                nn.Conv2d(in_channels=512, out_channels=1, kernel_size=3, padding=1), 12
            )
            for _ in range(512)
        ]
        self.batch_normal12_layer = nn.BatchNorm2d(512)

        self.conv13_layers = [
            MaskedLayer(
                nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
                13,
            )
        ] + [
            MaskedLayer(
                nn.Conv2d(in_channels=512, out_channels=1, kernel_size=3, padding=1), 13
            )
            for _ in range(512)
        ]
        self.batch_normal13_layer = nn.BatchNorm2d(512)

        self.max_pool_layer = nn.MaxPool2d(kernel_size=2)
        self.avg_pool_layer = nn.AvgPool2d(kernel_size=2)
        self.dropout = nn.Dropout(p=0.5)

        self.fc1_layer = nn.Linear(in_features=512, out_features=512)
        self.fc2_layer = nn.Linear(in_features=512, out_features=512)
        self.fc3_layer = nn.Linear(in_features=512, out_features=10)

    def forward(self, input):
        output = None
        torch.set_printoptions(
            precision=8,
            threshold=None,
            edgeitems=None,
            linewidth=None,
            profile="full",
        )

        if (
            self.work_mode == WorkMode.normal
            or self.work_mode == WorkMode.split
            or self.work_mode == WorkMode.recover
        ):
            conv1_output = self._conv(self.conv1_layers, input)

            bn1_output = self.batch_normal1_layer(conv1_output)
            bn1_output = self._activate(bn1_output, [torch.square, F.relu])

            conv2_output = self._conv(self.conv2_layers, bn1_output)
            bn2_output = self.batch_normal2_layer(conv2_output)
            bn2_output = F.relu(bn2_output)
            max_pool2_output = self.max_pool_layer(bn2_output)

            # conv3_output = self._conv(self.conv3_layers, max_pool2_output)
            conv3_output = self.conv3_layers[0](max_pool2_output)
            bn3_output = self.batch_normal3_layer(conv3_output)
            bn3_output = F.relu(bn3_output)

            # conv4_output = self._conv(self.conv4_layers, bn3_output)
            conv4_output = self.conv4_layers[0](bn3_output)
            bn4_output = self.batch_normal4_layer(conv4_output)
            bn4_output = F.relu(bn4_output)
            max_pool4_output = self.max_pool_layer(bn4_output)

            # conv5_output = self._conv(self.conv5_layers, max_pool4_output)
            conv5_output = self.conv5_layers[0](max_pool4_output)
            bn5_output = self.batch_normal5_layer(conv5_output)
            bn5_output = F.relu(bn5_output)

            # conv6_output = self._conv(self.conv6_layers, bn5_output)
            conv6_output = self.conv6_layers[0](bn5_output)
            bn6_output = self.batch_normal6_layer(conv6_output)
            bn6_output = F.relu(bn6_output)

            # conv7_output = self._conv(self.conv7_layers, bn6_output)
            conv7_output = self.conv7_layers[0](bn6_output)
            bn7_output = self.batch_normal7_layer(conv7_output)
            bn7_output = F.relu(bn7_output)
            max_pool7_output = self.max_pool_layer(bn7_output)

            # conv8_output = self._conv(self.conv8_layers, max_pool7_output)
            conv8_output = self.conv8_layers[0](max_pool7_output)
            bn8_output = self.batch_normal8_layer(conv8_output)
            bn8_output = F.relu(bn8_output)

            # conv9_output = self._conv(self.conv9_layers, bn8_output)
            conv9_output = self.conv9_layers[0](bn8_output)
            bn9_output = self.batch_normal9_layer(conv9_output)
            bn9_output = F.relu(bn9_output)

            # conv10_output = self._conv(self.conv10_layers, bn9_output)
            conv10_output = self.conv10_layers[0](bn9_output)
            bn10_output = self.batch_normal10_layer(conv10_output)
            bn10_output = F.relu(bn10_output)
            max_pool10_output = self.max_pool_layer(bn10_output)

            # conv11_output = self._conv(self.conv11_layers, max_pool10_output)
            conv11_output = self.conv11_layers[0](max_pool10_output)
            bn11_output = self.batch_normal11_layer(conv11_output)
            bn11_output = F.relu(bn11_output)

            # conv12_output = self._conv(self.conv12_layers, bn11_output)
            conv12_output = self.conv12_layers[0](bn11_output)
            bn12_output = self.batch_normal12_layer(conv12_output)
            bn12_output = F.relu(bn12_output)

            # conv13_output = self._conv(self.conv13_layers, bn12_output)
            conv13_output = self.conv13_layers[0](bn12_output)
            bn13_output = self.batch_normal13_layer(conv13_output)
            bn13_output = F.relu(bn13_output)
            max_pool13_output = self.max_pool_layer(bn13_output)

            fc1_input = max_pool13_output.reshape(max_pool13_output.shape[0], -1)
            fc1_output = self.dropout(F.relu(self.fc1_layer(fc1_input)))
            fc2_output = self.dropout(F.relu(self.fc2_layer(fc1_output)))
            output = self.fc3_layer(fc2_output)
        else:
            pass
        return output

    def get_layers_list(self, include_fc_layers=False):
        layers_list = [
            self.conv1_layers,
            self.conv2_layers,
            self.conv3_layers,
            self.conv4_layers,
            self.conv5_layers,
            self.conv6_layers,
            self.conv7_layers,
            self.conv8_layers,
            self.conv9_layers,
            self.conv10_layers,
            self.conv11_layers,
            self.conv12_layers,
            self.conv13_layers,
        ]

        if include_fc_layers:
            layers_list.extend(
                [
                    self.batch_normal1_layer,
                    self.batch_normal2_layer,
                    self.batch_normal3_layer,
                    self.batch_normal4_layer,
                    self.batch_normal5_layer,
                    self.batch_normal6_layer,
                    self.batch_normal7_layer,
                    self.batch_normal8_layer,
                    self.batch_normal9_layer,
                    self.batch_normal10_layer,
                    self.batch_normal11_layer,
                    self.batch_normal12_layer,
                    self.batch_normal13_layer,
                    self.fc1_layer,
                    self.fc2_layer,
                    self.fc3_layer,
                ]
            )

        return layers_list
