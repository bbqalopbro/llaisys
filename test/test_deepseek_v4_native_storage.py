"""C++ payload ownership, DLPack transfer, stream and real model parity."""

import gc
import os
import threading
import unittest
from unittest.mock import patch

import torch

from llaisys import _C
from llaisys.models.deepseek_v4_model import PagedCachePool
from test_deepseek_v4_model import initialized_model


class NativePagedStorageTests(unittest.TestCase):
    def devices(self):
        return ["cpu", "cuda"] if os.environ.get("SLURM_JOB_ID") and torch.cuda.is_available() else ["cpu"]

    def allocate(self, device="cpu", ratios=(0, 4, 128), blocks=3, stream=0):
        return _C.V4PagedStorage(blocks, 128, 128, 128, list(ratios), device, 0, stream)

    def test_native_layout_and_zero_copy_aliases(self):
        for device in self.devices():
            storage = self.allocate(device)
            for layer, ratio in enumerate((0, 4, 128)):
                components = ["latent", "index"] if ratio == 4 else ["latent"]
                for component in components:
                    with self.subTest(device=device, layer=layer, component=component):
                        view = storage.view(layer, component)
                        first = torch.utils.dlpack.from_dlpack(view)
                        second = torch.utils.dlpack.from_dlpack(view)
                        slots = (128 + (128 // ratio if ratio else 0)) if component == "latent" else 32
                        self.assertEqual(tuple(first.shape), (1, 3 * slots, 128))
                        self.assertEqual(first.dtype, torch.bfloat16)
                        self.assertEqual(first.device.type, device)
                        self.assertTrue(first.is_contiguous())
                        self.assertEqual(first.data_ptr(), second.data_ptr())
                        first.fill_(layer + 1)
                        self.assertTrue(bool((second == layer + 1).all()))
                        entry = next(row for row in storage.report()["layers"][layer]["components"] if row["name"] == component)
                        self.assertEqual(first.data_ptr(), entry["data_ptr"])
                        self.assertEqual(entry["block_stride_bytes"], slots * 128 * 2)
                        del first, second, view
            self.assertEqual(storage.report()["payload_bytes"], 3 * (128 + 160 + 32 + 129) * 128 * 2)
            del storage

    def test_legacy_and_versioned_capsules_are_single_consumer_and_keep_memory_alive(self):
        for version in (None, (1, 0)):
            before = _C.v4_storage_counters()
            storage = self.allocate(ratios=(4,), blocks=1)
            view = storage.view(0)
            capsule = view.__dlpack__(max_version=version)
            tensor = torch.utils.dlpack.from_dlpack(capsule)
            tensor.fill_(7)
            with self.assertRaises(RuntimeError):
                torch.utils.dlpack.from_dlpack(capsule)
            sliced = tensor[:, 17:31]
            del tensor, storage, view, capsule
            gc.collect()
            self.assertGreater(_C.v4_storage_counters()["live_bytes"], before["live_bytes"])
            self.assertTrue(bool((sliced == 7).all()))
            del sliced
            gc.collect()
            self.assertEqual(_C.v4_storage_counters(), before)

    def test_unconsumed_capsules_release_their_owners(self):
        for version in (None, (1, 0)):
            before = _C.v4_storage_counters()
            storage = self.allocate(ratios=(0,), blocks=1)
            capsule = storage.view(0).__dlpack__(max_version=version)
            del storage
            self.assertGreater(_C.v4_storage_counters()["live_bytes"], before["live_bytes"])
            del capsule
            gc.collect()
            self.assertEqual(_C.v4_storage_counters(), before)

    def test_invalid_layout_is_rejected_before_any_allocation(self):
        before = _C.v4_storage_counters()
        invalid = ((0, 128, 128, 128, [0]), (1, 128, 0, 128, [0]),
                   (1, 127, 128, 128, [4]), (1, 128, 128, 128, [0, 8]),
                   (1, 128, 128, 128, []), (2**31, 128, 128, 128, [0]),
                   (1, 128, 2**62, 128, [0]))
        for args in invalid:
            with self.subTest(args=args), self.assertRaises((ValueError, OverflowError)):
                _C.V4PagedStorage(*args)
            self.assertEqual(_C.v4_storage_counters(), before)
        for device, index, stream in (("meta", 0, 0), ("cpu", 1, 0), ("cpu", 0, 1)):
            with self.assertRaises(ValueError):
                _C.V4PagedStorage(1, 128, 128, 128, [0], device, index, stream)
        self.assertEqual(_C.v4_storage_counters(), before)

    def test_export_rejects_copy_device_stream_and_invalid_components(self):
        storage = self.allocate()
        view = storage.view(0)
        for kwargs in ({"copy": True}, {"dl_device": (2, 0)}, {"stream": 1}):
            with self.assertRaises(BufferError):
                view.__dlpack__(**kwargs)
        with self.assertRaises(ValueError):
            view.__dlpack__(max_version=(-1, 0))
        with self.assertRaises(IndexError):
            storage.view(3)
        for layer, component in ((0, "index"), (1, "key"), (2, "index")):
            with self.assertRaises(ValueError):
                storage.view(layer, component)
        tensor = torch.utils.dlpack.from_dlpack(view.__dlpack__(copy=False, dl_device=(1, 0)))
        self.assertEqual(tensor.dtype, torch.bfloat16)

    def test_missing_native_support_requires_explicit_reference_selection(self):
        model = initialized_model()
        with patch.object(_C, "v4_native_storage_available", False):
            with self.assertRaisesRegex(RuntimeError, "rebuild pybind"):
                PagedCachePool(model, num_blocks=2)
            reference = PagedCachePool(model, num_blocks=2, storage_backend="torch")
            self.assertEqual(reference.report()["tensor_storage"], "torch-device-pool")
            self.assertFalse(reference.report()["fallback"])
        with self.assertRaisesRegex(ValueError, "explicit cpp or torch"):
            PagedCachePool(model, num_blocks=2, storage_backend="auto")

    def test_gpu_storage_is_external_to_torch_allocator_and_requires_owning_stream(self):
        if "cuda" not in self.devices():
            return  # CPU regressions do not claim CUDA stream validation.
        before = _C.v4_storage_counters()
        stream = torch.cuda.Stream()
        torch_bytes = torch.cuda.memory_allocated()
        with torch.cuda.stream(stream):
            storage = self.allocate("cuda", ratios=(4,), blocks=1, stream=stream.cuda_stream)
            self.assertEqual(torch.cuda.memory_allocated(), torch_bytes)
            view = storage.view(0)
            tensor = torch.utils.dlpack.from_dlpack(view)
            self.assertEqual(torch.cuda.memory_allocated(), torch_bytes)
            tensor.fill_(9)
        with self.assertRaises(BufferError):
            torch.utils.dlpack.from_dlpack(view)
        with torch.cuda.stream(stream):
            self.assertTrue(bool((tensor == 9).all()))
        holder = [tensor]
        del tensor, storage, view
        # The consumer may invoke its pure-C++ DLPack deleter on another thread.
        thread = threading.Thread(target=holder.clear)
        thread.start()
        thread.join(timeout=10)
        self.assertFalse(thread.is_alive())
        gc.collect()
        self.assertEqual(_C.v4_storage_counters(), before)

    def test_native_and_torch_storage_match_model_prefill_chunks_and_decode(self):
        previous_threads = torch.get_num_threads()
        torch.set_num_threads(2)
        try:
            for device in self.devices():
                torch.manual_seed(43)
                model = initialized_model().to(device)
                native = PagedCachePool(model, num_blocks=4)
                reference = PagedCachePool(model, num_blocks=4, storage_backend="torch")
                self.assertEqual(native.report()["tensor_storage"], "cpp-paged-cache-storage")
                self.assertEqual(native.report()["payload_bytes"], native.native_storage.report()["payload_bytes"])
                tokens = torch.randint(0, model.config.vocab_size, (1, 137), device=device)
                for sizes in ((137,), (65, 65, 7), (3, 2, 7, 115, 10)):
                    first, second = model.new_request(cache_pool=native), model.new_request(cache_pool=reference)
                    try:
                        start = 0
                        for count in sizes:
                            ids = tokens[:, start:start+count]
                            expected = model(ids, second)
                            actual = model(ids, first)
                            torch.testing.assert_close(actual.logits, expected.logits, atol=0, rtol=0)
                            start += count
                        for _ in range(3):
                            ids = expected.logits.argmax(-1).unsqueeze(1)
                            expected, actual = model(ids, second), model(ids, first)
                            torch.testing.assert_close(actual.logits, expected.logits, atol=0, rtol=0)
                    finally:
                        first.close()
                        second.close()
                self.assertEqual(native.blocks.num_free, 4)
                self.assertEqual(reference.blocks.num_free, 4)
        finally:
            torch.set_num_threads(previous_threads)


if __name__ == "__main__":
    unittest.main()
