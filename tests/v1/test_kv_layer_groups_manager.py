# SPDX-License-Identifier: Apache-2.0
# Third Party
from lmcache_tests.v1.test_kv_layer_groups_manager import (
    TestKVLayerGroupsManager as UpstreamKVLayerGroupTests,
)
from lmcache.v1.kv_layer_groups import KVLayerGroupsManager
import torch


class TestKVLayerGroupsManager(UpstreamKVLayerGroupTests):
    pass


def test_build_kv_layer_groups_with_mla_tuple_kvcache():
    manager = KVLayerGroupsManager()
    kv_caches = {
        "layer0": (
            torch.empty(64, 16, 1, 512, dtype=torch.bfloat16),
            torch.empty(64, 16, 1, 64, dtype=torch.bfloat16),
        ),
        "layer1": (
            torch.empty(64, 16, 1, 512, dtype=torch.bfloat16),
            torch.empty(64, 16, 1, 64, dtype=torch.bfloat16),
        ),
    }

    manager.build_kv_layer_groups(kv_caches)

    assert manager.num_groups == 1
    group = manager.kv_layer_groups[0]
    assert group.shape == torch.Size([64, 16, 576])
    assert group.hidden_dim_size == 576


def test_build_kv_layer_groups_with_dsa_tuple_kvcache():
    manager = KVLayerGroupsManager()
    kv_caches = {
        "layer0": (
            torch.empty(64, 16, 1, 512, dtype=torch.bfloat16),
            torch.empty(64, 16, 1, 64, dtype=torch.bfloat16),
            torch.empty(64, 16, 1, 128, dtype=torch.bfloat16),
        ),
    }

    manager.build_kv_layer_groups(kv_caches)

    assert manager.num_groups == 1
    group = manager.kv_layer_groups[0]
    assert group.shape == torch.Size([64, 16, 704])
    assert group.hidden_dim_size == 704
