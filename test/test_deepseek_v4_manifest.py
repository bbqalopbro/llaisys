import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest


MODULE_PATH = (
    Path(__file__).parents[1]
    / "python"
    / "llaisys"
    / "models"
    / "deepseek_v4.py"
)
SPEC = importlib.util.spec_from_file_location("llaisys_deepseek_v4", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)

DeepSeekV4Config = MODULE.DeepSeekV4Config
DeepSeekV4ConfigError = MODULE.DeepSeekV4ConfigError
DeepSeekV4WeightManifest = MODULE.DeepSeekV4WeightManifest
parse_weight_key = MODULE.parse_weight_key


class DeepSeekV4ConfigTests(unittest.TestCase):
    def _config(self):
        return {
            "model_type": "deepseek_v4",
            "architectures": ["DeepseekV4ForCausalLM"],
            "hidden_size": 4096,
            "num_hidden_layers": 2,
            "num_attention_heads": 64,
            "num_key_value_heads": 1,
            "head_dim": 512,
            "q_lora_rank": 1024,
            "qk_rope_head_dim": 64,
            "o_groups": 8,
            "o_lora_rank": 1024,
            "sliding_window": 128,
            "compress_ratios": [0, 4, 0],
            "compress_rope_theta": 160000,
            "index_n_heads": 64,
            "index_head_dim": 128,
            "index_topk": 512,
            "vocab_size": 129280,
            "max_position_embeddings": 1048576,
            "torch_dtype": "bfloat16",
            "quantization_config": {
                "quant_method": "fp8",
                "fmt": "e4m3",
                "scale_fmt": "ue8m0",
                "weight_block_size": [128, 128],
            },
            "expert_dtype": "fp4",
            "moe_intermediate_size": 2048,
            "n_routed_experts": 8,
            "n_shared_experts": 1,
            "num_experts_per_tok": 2,
            "num_hash_layers": 1,
            "topk_method": "noaux_tc",
            "scoring_func": "sqrtsoftplus",
            "routed_scaling_factor": 1.5,
            "dspark_block_size": 5,
            "dspark_target_layer_ids": [0, 1],
            "dspark_markov_rank": 256,
            "num_nextn_predict_layers": 1,
        }

    def test_config_and_inference_aliases(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "inference").mkdir()
            (root / "config.json").write_text(json.dumps(self._config()))
            inference = {
                "dim": 4096,
                "n_layers": 2,
                "n_heads": 64,
                "n_routed_experts": 8,
                "n_shared_experts": 1,
                "n_activated_experts": 2,
                "q_lora_rank": 1024,
                "head_dim": 512,
                "rope_head_dim": 64,
                "window_size": 128,
                "n_mtp_layers": 1,
            }
            (root / "inference" / "config.json").write_text(json.dumps(inference))
            config = DeepSeekV4Config.from_directory(root)
            self.assertEqual(config.nope_head_dim, 448)
            self.assertEqual(config.main_compress_ratios, (0, 4))
            self.assertEqual(config.inference_mtp_stages, 1)

    def test_rejects_wrong_model_family(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = self._config()
            config["model_type"] = "qwen2"
            (root / "config.json").write_text(json.dumps(config))
            with self.assertRaises(DeepSeekV4ConfigError):
                DeepSeekV4Config.from_directory(root)


class DeepSeekV4WeightKeyTests(unittest.TestCase):
    def test_maps_attention_indexer(self):
        spec = parse_weight_key("layers.2.attn.indexer.compressor.wkv.weight")
        self.assertEqual(spec.scope, "layer")
        self.assertEqual(spec.block_index, 2)
        self.assertEqual(spec.role, "attention_indexer")

    def test_maps_routed_expert(self):
        spec = parse_weight_key("mtp.1.ffn.experts.255.w2.scale")
        self.assertEqual(spec.scope, "mtp")
        self.assertEqual(spec.block_index, 1)
        self.assertEqual(spec.role, "routed_expert")
        self.assertEqual(spec.expert_index, 255)
        self.assertEqual(spec.projection, "w2")
        self.assertEqual(spec.value_kind, "scale")

    def test_rejects_unknown_namespace(self):
        with self.assertRaises(DeepSeekV4ConfigError):
            parse_weight_key("layers.0.unowned.weight")


class PublishedCheckpointTests(unittest.TestCase):
    MODEL_DIR = Path("/home/lcpu/models/deepseek-ai/DeepSeek-V4-Flash-0731")

    @unittest.skipUnless(MODEL_DIR.is_dir(), "shared DeepSeek checkpoint unavailable")
    def test_real_checkpoint_contract_and_headers(self):
        config = DeepSeekV4Config.from_directory(self.MODEL_DIR)
        manifest = DeepSeekV4WeightManifest.from_directory(self.MODEL_DIR)
        manifest.validate_structure(config)
        metadata = manifest.read_tensor_metadata()

        self.assertEqual(config.num_hidden_layers, 43)
        self.assertEqual(config.num_routed_experts, 256)
        self.assertEqual(config.experts_per_token, 6)
        self.assertEqual(config.nope_head_dim, 448)
        self.assertEqual(config.inference_mtp_stages, 3)
        self.assertEqual(manifest.mtp_stages, (0, 1, 2))
        self.assertEqual(manifest.tensor_count, 72317)
        self.assertEqual(len(manifest.shard_names), 48)
        self.assertEqual(manifest.declared_total_size, 166878536440)
        self.assertEqual(metadata["embed.weight"].shape, (129280, 4096))
        self.assertEqual(metadata["layers.0.attn.wkv.weight"].shape, (512, 4096))
        self.assertEqual(metadata["layers.0.ffn.experts.0.w1.weight"].dtype, "I8")
        self.assertEqual(metadata["layers.0.ffn.experts.0.w1.scale"].dtype, "F8_E8M0")

    @unittest.skipUnless(MODEL_DIR.is_dir(), "shared DeepSeek checkpoint unavailable")
    def test_real_checkpoint_loads_small_slices_only(self):
        manifest = DeepSeekV4WeightManifest.from_directory(self.MODEL_DIR)
        q_norm = manifest.load_tensor_slice(
            "layers.0.attn.q_norm.weight", slice(0, 8)
        )
        token_routes = manifest.load_tensor_slice(
            "layers.0.ffn.gate.tid2eid", slice(0, 2)
        )
        fp8_weight = manifest.load_tensor_slice(
            "layers.0.attn.wkv.weight", (slice(0, 2), slice(0, 8))
        )
        self.assertEqual(tuple(q_norm.shape), (8,))
        self.assertEqual(str(q_norm.dtype), "torch.bfloat16")
        self.assertEqual(tuple(token_routes.shape), (2, 6))
        self.assertGreaterEqual(int(token_routes.min()), 0)
        self.assertLess(int(token_routes.max()), 256)
        self.assertEqual(tuple(fp8_weight.shape), (2, 8))
        self.assertEqual(str(fp8_weight.dtype), "torch.float8_e4m3fn")


if __name__ == "__main__":
    unittest.main(verbosity=2)
