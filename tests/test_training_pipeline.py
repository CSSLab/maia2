import io
import hashlib
import json
import queue
import random
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import chess.pgn
import numpy as np
import torch

from maia2 import train
from maia2.main import process_per_game, read_monthly_data_path
from maia2.utils import create_elo_dict, extract_clock_time, get_chunks, get_side_info
from maia2.utils import read_or_create_chunks


PGN_GAME = """[Event "Rated Rapid game"]
[Site "https://lichess.org/testgame"]
[Date "2026.07.15"]
[Round "-"]
[White "WhitePlayer"]
[Black "BlackPlayer"]
[Result "1-0"]
[WhiteElo "1500"]
[BlackElo "1500"]
[TimeControl "600+0"]

1. e4 { [%clk 0:10:00] } e5 { [%clk 0:09:59] } 1-0
"""


class TrainingPipelineTest(unittest.TestCase):
    def test_fresh_and_resumed_training_schedules_preserve_all_epochs(self):
        paths = [
            f"/data/lichess_db_standard_rated_2023-{month:02d}.pgn"
            for month in range(1, 4)
        ]
        fresh_cfg = SimpleNamespace(max_epochs=3, from_checkpoint=False)
        self.assertEqual(
            train._training_schedule(fresh_cfg, paths),
            [(0, paths), (1, paths), (2, paths)],
        )

        resumed_cfg = SimpleNamespace(
            max_epochs=3,
            from_checkpoint=True,
            checkpoint_epoch=2,
            checkpoint_year=2023,
            checkpoint_month=2,
        )
        self.assertEqual(
            train._training_schedule(resumed_cfg, paths),
            [(1, paths[2:]), (2, paths)],
        )

        resumed_cfg.checkpoint_month = 3
        self.assertEqual(
            train._training_schedule(resumed_cfg, paths),
            [(1, []), (2, paths)],
        )

    def test_invalid_resume_schedule_fails_closed(self):
        paths = ["/data/lichess_db_standard_rated_2023-01.pgn"]
        cfg = SimpleNamespace(
            max_epochs=3,
            from_checkpoint=True,
            checkpoint_epoch=0,
            checkpoint_year=2023,
            checkpoint_month=1,
        )
        with self.assertRaisesRegex(ValueError, "checkpoint_epoch"):
            train._training_schedule(cfg, paths)

        cfg.checkpoint_epoch = 1
        cfg.checkpoint_month = 2
        with self.assertRaisesRegex(ValueError, "original full start/end range"):
            train._training_schedule(cfg, paths)

    def test_month_paths_are_validated_and_historical_skip_is_configurable(self):
        cfg = SimpleNamespace(
            data_root="/data",
            start_year=2019,
            start_month=11,
            end_year=2020,
            end_month=1,
        )
        paths = read_monthly_data_path(cfg)
        self.assertEqual(
            [Path(path).name for path in paths],
            [
                "lichess_db_standard_rated_2019-11.pgn",
                "lichess_db_standard_rated_2020-01.pgn",
            ],
        )

        cfg.skip_months = []
        self.assertEqual(len(read_monthly_data_path(cfg)), 3)

        cfg.start_month = 13
        with self.assertRaisesRegex(ValueError, "between 1 and 12"):
            read_monthly_data_path(cfg)

        cfg.start_year = 2021
        cfg.start_month = 1
        with self.assertRaisesRegex(ValueError, "must not be after"):
            read_monthly_data_path(cfg)

    def test_empty_pgn_has_no_chunks(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, "empty.pgn")
            path.write_text("\n\n", encoding="utf-8")

            self.assertEqual(get_chunks(path, chunk_size=10), [])

    def test_chunk_cache_depends_on_chunk_and_source_size(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, "games.pgn")
            path.write_text(PGN_GAME + "\n" + PGN_GAME, encoding="utf-8")

            one_game_chunks = read_or_create_chunks(path, SimpleNamespace(chunk_size=1))
            two_game_chunks = read_or_create_chunks(path, SimpleNamespace(chunk_size=2))

            self.assertEqual(len(one_game_chunks), 2)
            self.assertEqual(len(two_game_chunks), 1)
            self.assertTrue(Path(directory, "games_chunks_1.json").is_file())
            self.assertTrue(Path(directory, "games_chunks_2.json").is_file())

    def test_tampered_chunk_cache_is_rebuilt_with_exact_source_coverage(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, "games.pgn")
            path.write_text(PGN_GAME + "\n" + PGN_GAME, encoding="utf-8")
            cfg = SimpleNamespace(chunk_size=1)
            expected = read_or_create_chunks(path, cfg)
            cache_path = Path(directory, "games_chunks_1.json")
            clean_payload = json.loads(cache_path.read_text(encoding="utf-8"))

            tampered_chunks = [
                [],
                [[1, expected[0][1]], list(expected[1])],
                [list(expected[0]), [expected[1][0] + 1, expected[1][1]]],
                [list(expected[0]), [expected[1][0], expected[1][1] - 1]],
            ]
            for chunks in tampered_chunks:
                with self.subTest(chunks=chunks):
                    payload = dict(clean_payload)
                    payload["chunks"] = chunks
                    cache_path.write_text(json.dumps(payload), encoding="utf-8")

                    rebuilt = read_or_create_chunks(path, cfg)

                    self.assertEqual(rebuilt, expected)
                    rebuilt_payload = json.loads(cache_path.read_text(encoding="utf-8"))
                    self.assertEqual(
                        [tuple(chunk) for chunk in rebuilt_payload["chunks"]],
                        expected,
                    )

            cache_path.write_text("[]", encoding="utf-8")
            with mock.patch("maia2.utils.get_chunks", wraps=get_chunks) as chunker:
                rebuilt = read_or_create_chunks(path, cfg)
            chunker.assert_called_once()
            self.assertEqual(rebuilt, expected)

    def test_clock_parser_handles_fractional_and_malformed_annotations(self):
        self.assertEqual(extract_clock_time("[%clk 1:02:03]"), 3723)
        self.assertEqual(extract_clock_time("[%clk 0:00:03.5]"), 3.5)
        self.assertEqual(extract_clock_time("[%clk invalid]"), 0.0)
        self.assertEqual(extract_clock_time(""), 0.0)

    def test_en_passant_labels_a_captured_pawn_in_side_info(self):
        board = chess.Board()
        for move in ("e2e4", "a7a6", "e4e5", "d7d5"):
            board.push_uci(move)
        all_moves = {
            move: index for index, move in enumerate(train.get_all_possible_moves())
        }

        _, side_info = get_side_info(board, "e5d6", all_moves)

        captured_pawn_index = 6
        self.assertEqual(side_info[captured_pawn_index].item(), 1.0)

    def test_malformed_clock_is_discarded_instead_of_crashing(self):
        pgn = PGN_GAME.replace("[%clk 0:10:00]", "[%clk invalid]")
        game = chess.pgn.read_game(io.StringIO(pgn))
        cfg = SimpleNamespace(first_n_moves=0, max_ply=300, clock_threshold=30)

        positions = process_per_game(game, 1500, 1500, 1, cfg)

        self.assertEqual(len(positions), 1)

    def test_worker_traceback_is_propagated_to_parent(self):
        result_queue = queue.Queue()
        with mock.patch(
            "maia2.train.preprocess_thread", side_effect=ValueError("bad PGN")
        ):
            train._preprocess_worker(
                result_queue,
                SimpleNamespace(),
                "bad.pgn",
                [(0, 1)],
                {},
            )

        worker = mock.Mock()
        result = result_queue.get_nowait()
        result_queue.put(result)
        with self.assertRaisesRegex(RuntimeError, "ValueError: bad PGN"):
            train._wait_for_preprocessing_result(result_queue, worker)
        worker.join.assert_called_once_with()

    def test_spawn_preprocessing_pipeline_returns_real_positions(self):
        with tempfile.TemporaryDirectory() as directory:
            pgn_path = Path(directory, "rapid.pgn")
            pgn_path.write_text(PGN_GAME, encoding="utf-8")
            chunks = get_chunks(pgn_path, chunk_size=1)
            cfg = SimpleNamespace(
                queue_length=1,
                multiprocessing_start_method="spawn",
                verbose=0,
                first_n_moves=0,
                max_ply=300,
                clock_threshold=30,
                max_games_per_elo_range=20,
            )

            batches = list(
                train._iter_preprocessed_batches(
                    cfg,
                    str(pgn_path),
                    [chunks],
                    create_elo_dict(),
                )
            )

            self.assertEqual(len(batches), 1)
            positions, games, chunk_count = batches[0]
            self.assertEqual(len(positions), 2)
            self.assertEqual(games, 1)
            self.assertEqual(chunk_count, 1)

    def test_silent_worker_exit_has_a_clear_error(self):
        result_queue = queue.Queue()
        worker = mock.Mock()
        worker.is_alive.return_value = False
        worker.exitcode = 17

        with (
            mock.patch.object(train, "_WORKER_POLL_SECONDS", 0.001),
            self.assertRaisesRegex(RuntimeError, "exit code 17"),
        ):
            train._wait_for_preprocessing_result(result_queue, worker)

        worker.join.assert_called_once_with()

    def test_optimizer_step_count_reads_actual_adamw_state(self):
        parameter = torch.nn.Parameter(torch.tensor(1.0))
        optimizer = torch.optim.AdamW([parameter])
        self.assertEqual(train._optimizer_step_count(optimizer), 0)

        parameter.grad = torch.tensor(1.0)
        optimizer.step()

        self.assertEqual(train._optimizer_step_count(optimizer), 1)

    def test_empty_month_fails_closed_before_checkpointing(self):
        with self.assertRaisesRegex(
            RuntimeError,
            r"positions=0, games=0, optimizer_steps=0.*no checkpoint",
        ):
            train._require_trained_month(
                "month.pgn",
                positions=0,
                games=0,
                optimizer_steps=0,
            )

        train._require_trained_month(
            "month.pgn",
            positions=10,
            games=1,
            optimizer_steps=2,
        )

    def test_run_keeps_empty_month_and_does_not_write_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            pgn_path = Path(directory, "lichess_db_standard_rated_2023-01.pgn")
            pgn_path.write_text("", encoding="utf-8")
            Path(str(pgn_path) + ".zst").write_bytes(b"archive")
            cfg = SimpleNamespace(
                seed=42,
                num_cpu_left=0,
                save_root=str(Path(directory, "saves")),
                lr=1e-4,
                batch_size=2,
                wd=1e-5,
                from_checkpoint=False,
                max_epochs=1,
            )
            tiny_model = torch.nn.Linear(1, 1)

            with (
                mock.patch("maia2.train.seed_everything"),
                mock.patch("maia2.train.get_all_possible_moves", return_value=[]),
                mock.patch("maia2.train.create_elo_dict", return_value={}),
                mock.patch("maia2.train.MAIA2Model", return_value=tiny_model),
                mock.patch(
                    "maia2.train.read_monthly_data_path",
                    return_value=[str(pgn_path)],
                ),
                mock.patch("maia2.train.decompress_zst"),
                mock.patch(
                    "maia2.train._validated_decompression_provenance",
                    return_value={
                        "archive": {
                            "name": f"{pgn_path.name}.zst",
                            "size": 7,
                            "sha256": hashlib.sha256(b"archive").hexdigest(),
                        },
                        "decompressed": {
                            "name": pgn_path.name,
                            "size": 0,
                            "sha256": hashlib.sha256(b"").hexdigest(),
                        },
                    },
                ),
                mock.patch("maia2.train.read_or_create_chunks", return_value=[]),
                mock.patch("maia2.train.torch.save") as save,
                self.assertRaisesRegex(RuntimeError, "no usable data"),
            ):
                train.run(cfg, device="cpu")

            self.assertTrue(pgn_path.exists())
            save.assert_not_called()

    def test_checkpoint_write_is_atomic_and_cleans_up_failures(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory, "checkpoint.pt")
            train._save_checkpoint_atomic({"value": torch.tensor([1, 2])}, destination)
            loaded = torch.load(destination, weights_only=True)
            torch.testing.assert_close(loaded["value"], torch.tensor([1, 2]))

            temporary_pattern = f".{destination.name}.*.tmp"
            self.assertEqual(list(Path(directory).glob(temporary_pattern)), [])

            with (
                mock.patch(
                    "maia2.train.torch.save", side_effect=RuntimeError("disk full")
                ),
                self.assertRaisesRegex(RuntimeError, "disk full"),
            ):
                train._save_checkpoint_atomic({"value": 1}, destination, overwrite=True)

            loaded = torch.load(destination, weights_only=True)
            torch.testing.assert_close(loaded["value"], torch.tensor([1, 2]))
            self.assertEqual(list(Path(directory).glob(temporary_pattern)), [])

            with self.assertRaisesRegex(FileExistsError, "Refusing to overwrite"):
                train._save_checkpoint_atomic({"value": 3}, destination)

            train._save_checkpoint_atomic(
                {"value": torch.tensor([3])}, destination, overwrite=True
            )
            loaded = torch.load(destination, weights_only=True)
            torch.testing.assert_close(loaded["value"], torch.tensor([3]))

    def test_rng_state_round_trips_through_weights_only_checkpoint(self):
        random.seed(7)
        np.random.seed(7)
        torch.manual_seed(7)
        state = train._capture_rng_state(torch.device("cpu"))
        expected = (random.random(), np.random.random(), torch.rand(1))

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, "rng.pt")
            torch.save({"rng_state": state}, path)
            loaded = torch.load(path, weights_only=True)

        random.seed(999)
        np.random.seed(999)
        torch.manual_seed(999)
        train._restore_rng_state(loaded["rng_state"])

        self.assertEqual(random.random(), expected[0])
        self.assertEqual(np.random.random(), expected[1])
        torch.testing.assert_close(torch.rand(1), expected[2])

    def test_cuda_rng_restore_tolerates_a_different_visible_device_count(self):
        state = {
            "python": random.getstate(),
            "numpy": train._capture_rng_state(torch.device("cpu"))["numpy"],
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": [
                torch.tensor([1], dtype=torch.uint8),
                torch.tensor([2], dtype=torch.uint8),
            ],
        }
        with (
            mock.patch("maia2.train.torch.cuda.is_available", return_value=True),
            mock.patch("maia2.train.torch.cuda.device_count", return_value=1),
            mock.patch("maia2.train.torch.cuda.set_rng_state") as set_rng_state,
            self.assertWarnsRegex(RuntimeWarning, "2 device.*1 are visible"),
        ):
            train._restore_rng_state(state)

        set_rng_state.assert_called_once()
        self.assertEqual(set_rng_state.call_args.kwargs["device"], 0)

    def test_training_metadata_records_config_and_source_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            pgn_path = Path(directory, "lichess_db_standard_rated_2023-01.pgn")
            pgn_path.write_bytes(b"pgn")
            Path(str(pgn_path) + ".zst").write_bytes(b"archive")
            archive_sha256 = hashlib.sha256(b"archive").hexdigest()
            cfg = SimpleNamespace(
                data_root=Path(directory),
                source_sha256=archive_sha256,
                source_sha256_by_path={Path("archive.pgn.zst"): archive_sha256},
            )

            metadata = train._training_metadata(
                cfg,
                str(pgn_path),
                epoch=2,
                optimizer_steps=17,
                source_sha256=archive_sha256,
            )

            self.assertEqual(metadata["format_version"], 3)
            self.assertEqual(metadata["optimizer_steps"], 17)
            self.assertEqual(metadata["config"]["data_root"], directory)
            self.assertEqual(metadata["source"]["archive_size"], 7)
            self.assertEqual(metadata["source"]["decompressed_size"], 3)
            self.assertEqual(metadata["source"]["archive_sha256"], archive_sha256)
            self.assertEqual(
                metadata["source"]["decompressed_sha256"],
                hashlib.sha256(b"pgn").hexdigest(),
            )
            self.assertEqual(
                metadata["critical_config_sha256"],
                train._run_manifest(cfg)["critical_config_sha256"],
            )
            self.assertEqual(
                metadata["config"]["source_sha256_by_path"],
                {"archive.pgn.zst": archive_sha256},
            )
            checkpoint_path = Path(directory, "metadata.pt")
            torch.save({"training_metadata": metadata}, checkpoint_path)
            loaded = torch.load(checkpoint_path, weights_only=True)
            self.assertEqual(loaded["training_metadata"], metadata)

            del cfg.source_sha256
            metadata = train._training_metadata(
                cfg,
                str(pgn_path),
                epoch=2,
                optimizer_steps=17,
            )
            self.assertEqual(
                metadata["source"]["archive_sha256"],
                hashlib.sha256(b"archive").hexdigest(),
            )

    def test_optional_source_hash_is_verified_before_training(self):
        with tempfile.TemporaryDirectory() as directory:
            pgn_path = Path(directory, "lichess_db_standard_rated_2023-01.pgn")
            archive_path = Path(str(pgn_path) + ".zst")
            contents = b"archive"
            archive_path.write_bytes(contents)
            expected = hashlib.sha256(contents).hexdigest()

            train._verify_expected_source_hash(
                SimpleNamespace(source_sha256=expected),
                str(pgn_path),
            )
            self.assertEqual(
                train._verify_expected_source_hash(SimpleNamespace(), str(pgn_path)),
                expected,
            )
            with self.assertRaisesRegex(RuntimeError, "SHA-256 mismatch"):
                train._verify_expected_source_hash(
                    SimpleNamespace(source_sha256="0" * 64),
                    str(pgn_path),
                )

    def test_source_hash_config_is_unambiguous_and_complete_for_all_months(self):
        paths = [
            "/data/lichess_db_standard_rated_2023-01.pgn",
            "/data/lichess_db_standard_rated_2023-02.pgn",
        ]
        january = "1" * 64
        february = "2" * 64

        with self.assertRaisesRegex(ValueError, "one digest only"):
            train._source_hash_expectations(
                SimpleNamespace(source_sha256=january), paths
            )
        with self.assertRaisesRegex(ValueError, "incomplete.*2023-02"):
            train._source_hash_expectations(
                SimpleNamespace(source_sha256={"2023-01": january}), paths
            )
        with self.assertRaisesRegex(ValueError, "unknown key"):
            train._source_hash_expectations(
                SimpleNamespace(
                    source_sha256={
                        "2023-01": january,
                        "2023-02": february,
                        "2023-03": "3" * 64,
                    }
                ),
                paths,
            )

        resolved = train._source_hash_expectations(
            SimpleNamespace(
                source_sha256={
                    "2023-01": january.upper(),
                    "lichess_db_standard_rated_2023-02.pgn.zst": february,
                }
            ),
            paths,
        )
        self.assertEqual(resolved[paths[0]], january)
        self.assertEqual(resolved[paths[1]], february)

        with self.assertRaisesRegex(ValueError, "64-character hexadecimal"):
            train._source_hash_expectations(
                SimpleNamespace(source_sha256={"2023-01": "not-a-hash"}),
                paths[:1],
            )

    def test_multi_month_scalar_source_hash_fails_before_model_construction(self):
        cfg = SimpleNamespace(
            seed=42,
            num_cpu_left=0,
            max_epochs=1,
            from_checkpoint=False,
            source_sha256="a" * 64,
        )
        paths = [
            "/data/lichess_db_standard_rated_2023-01.pgn",
            "/data/lichess_db_standard_rated_2023-02.pgn",
        ]
        with (
            mock.patch("maia2.train.seed_everything"),
            mock.patch("maia2.train.read_monthly_data_path", return_value=paths),
            mock.patch("maia2.train.MAIA2Model") as model,
            self.assertRaisesRegex(ValueError, "one digest only"),
        ):
            train.run(cfg, device="cpu")
        model.assert_not_called()

    def test_run_manifest_detects_config_and_legacy_directory_collisions(self):
        cfg = self._resume_config()
        with tempfile.TemporaryDirectory() as directory:
            save_root = Path(directory)
            manifest_path = train._ensure_run_manifest(save_root, cfg)
            original = manifest_path.read_text(encoding="utf-8")
            self.assertEqual(train._ensure_run_manifest(save_root, cfg), manifest_path)
            manifest = json.loads(original)
            self.assertEqual(
                manifest["critical_config"]["filters"]["policy"],
                "rated-rapid-bot-title-clock-v1",
            )

            incompatible = self._resume_config(dim_cnn=16)
            with self.assertRaisesRegex(RuntimeError, "Critical training"):
                train._ensure_run_manifest(save_root, incompatible)
            self.assertEqual(manifest_path.read_text(encoding="utf-8"), original)

            incompatible_data = self._resume_config(chunk_size=999)
            with self.assertRaisesRegex(RuntimeError, "Critical training"):
                train._ensure_run_manifest(save_root, incompatible_data)

            incompatible_source = self._resume_config(source_sha256="b" * 64)
            with self.assertRaisesRegex(RuntimeError, "Critical training"):
                train._ensure_run_manifest(save_root, incompatible_source)

        with tempfile.TemporaryDirectory() as directory:
            save_root = Path(directory)
            Path(save_root, "epoch_1_2023-01.pgn.pt").touch()
            with self.assertRaisesRegex(RuntimeError, "no run manifest"):
                train._ensure_run_manifest(save_root, cfg)
            with self.assertWarnsRegex(RuntimeWarning, "legacy resume"):
                train._ensure_run_manifest(save_root, cfg, allow_legacy_resume=True)

        with tempfile.TemporaryDirectory() as directory:
            save_root = Path(directory)
            Path(save_root, "epoch_1_2023-01.pgn.pt").touch()
            cfg.overwrite_checkpoints = True
            with self.assertRaisesRegex(RuntimeError, "no run manifest"):
                train._ensure_run_manifest(save_root, cfg)

    def test_checkpoint_schedule_collisions_fail_before_training(self):
        with tempfile.TemporaryDirectory() as directory:
            pgn_path = "/data/lichess_db_standard_rated_2023-01.pgn"
            destination = Path(directory, train._checkpoint_name(1, pgn_path))
            destination.touch()
            schedule = [(0, [pgn_path])]

            with self.assertRaisesRegex(FileExistsError, "would overwrite"):
                train._validate_checkpoint_destinations(directory, schedule)
            train._validate_checkpoint_destinations(
                directory, schedule, overwrite_checkpoints=True
            )

    def test_resume_checkpoint_is_cpu_staged_validated_and_device_normalized(self):
        cfg = self._resume_config()
        archive_sha256 = "a" * 64
        source_model = torch.nn.Linear(2, 1)
        source_optimizer = torch.optim.AdamW(
            source_model.parameters(), lr=cfg.lr, weight_decay=cfg.wd
        )
        source_model(torch.ones(1, 2)).sum().backward()
        source_optimizer.step()
        checkpoint = {
            "model_state_dict": source_model.state_dict(),
            "optimizer_state_dict": source_optimizer.state_dict(),
            "training_metadata": {
                "format_version": 3,
                "epoch": cfg.checkpoint_epoch,
                "critical_config_sha256": train._run_manifest(cfg)[
                    "critical_config_sha256"
                ],
                "config": vars(cfg).copy(),
                "source": {
                    "archive_name": "lichess_db_standard_rated_2023-01.pgn.zst",
                    "archive_sha256": archive_sha256,
                },
            },
        }

        target_model = torch.nn.Linear(2, 1)
        target_optimizer = torch.optim.AdamW(
            target_model.parameters(), lr=cfg.lr, weight_decay=cfg.wd
        )
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory, "checkpoint.pt")
            torch.save(checkpoint, checkpoint_path)
            original_torch_load = torch.load
            with mock.patch(
                "maia2.train.torch.load", wraps=original_torch_load
            ) as load:
                train._load_resume_checkpoint(
                    checkpoint_path,
                    target_model,
                    target_optimizer,
                    cfg,
                    expected_source_sha256=archive_sha256,
                )

        self.assertEqual(load.call_args.kwargs["map_location"], "cpu")
        self.assertTrue(load.call_args.kwargs["weights_only"])
        for key, value in source_model.state_dict().items():
            torch.testing.assert_close(target_model.state_dict()[key], value)
        for parameter, state in target_optimizer.state.items():
            self.assertEqual(state["step"].device, torch.device("cpu"))
            self.assertEqual(state["exp_avg"].device, parameter.device)
            self.assertEqual(state["exp_avg_sq"].device, parameter.device)

    def test_resume_rejects_optimizer_hyperparameters_that_disagree_with_config(self):
        cfg = self._resume_config()
        archive_sha256 = "a" * 64
        source_model = torch.nn.Linear(2, 1)
        source_optimizer = torch.optim.AdamW(
            source_model.parameters(), lr=9e-4, weight_decay=cfg.wd
        )
        source_model(torch.ones(1, 2)).sum().backward()
        source_optimizer.step()
        checkpoint = {
            "model_state_dict": source_model.state_dict(),
            "optimizer_state_dict": source_optimizer.state_dict(),
            "training_metadata": {
                "format_version": 3,
                "epoch": cfg.checkpoint_epoch,
                "critical_config_sha256": train._run_manifest(cfg)[
                    "critical_config_sha256"
                ],
                "config": vars(cfg).copy(),
                "source": {
                    "archive_name": "lichess_db_standard_rated_2023-01.pgn.zst",
                    "archive_sha256": archive_sha256,
                },
            },
        }

        target_model = torch.nn.Linear(2, 1)
        target_optimizer = torch.optim.AdamW(
            target_model.parameters(), lr=cfg.lr, weight_decay=cfg.wd
        )
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory, "checkpoint.pt")
            torch.save(checkpoint, checkpoint_path)
            with self.assertRaisesRegex(
                RuntimeError,
                "optimizer hyperparameters[\\s\\S]*param_group\\[0\\].lr",
            ):
                train._load_resume_checkpoint(
                    checkpoint_path,
                    target_model,
                    target_optimizer,
                    cfg,
                    expected_source_sha256=archive_sha256,
                )

    def test_optimizer_validation_locks_adamw_semantics(self):
        cfg = self._resume_config()
        model = torch.nn.Linear(2, 1)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=cfg.lr, weight_decay=cfg.wd
        )
        train._validate_optimizer_hyperparameters(optimizer, cfg)

        incompatible_values = {
            "betas": (0.8, 0.999),
            "eps": 1e-7,
            "amsgrad": True,
            "maximize": True,
            "capturable": True,
            "differentiable": True,
            "foreach": False,
            "fused": False,
            "decoupled_weight_decay": False,
        }
        parameter_group = optimizer.param_groups[0]
        for key, incompatible_value in incompatible_values.items():
            with self.subTest(key=key):
                compatible_value = parameter_group[key]
                parameter_group[key] = incompatible_value
                with self.assertRaisesRegex(RuntimeError, f"param_group\\[0\\].{key}"):
                    train._validate_optimizer_hyperparameters(optimizer, cfg)
                parameter_group[key] = compatible_value

    def test_resume_metadata_rejects_epoch_source_and_critical_config_mismatch(self):
        cfg = self._resume_config()
        source_sha256 = "a" * 64

        def checkpoint_for(config=None, epoch=None, archive=None, digest=None):
            return {
                "training_metadata": {
                    "format_version": 3,
                    "epoch": cfg.checkpoint_epoch if epoch is None else epoch,
                    "critical_config_sha256": train._run_manifest(cfg)[
                        "critical_config_sha256"
                    ],
                    "config": vars(cfg).copy() if config is None else config,
                    "source": {
                        "archive_name": (
                            "lichess_db_standard_rated_2023-01.pgn.zst"
                            if archive is None
                            else archive
                        ),
                        "archive_sha256": (source_sha256 if digest is None else digest),
                    },
                }
            }

        common = dict(
            cfg=cfg,
            expected_epoch=2,
            expected_archive_name="lichess_db_standard_rated_2023-01.pgn.zst",
            expected_source_sha256=source_sha256,
        )
        missing_format_version = checkpoint_for()
        del missing_format_version["training_metadata"]["format_version"]
        with self.assertRaisesRegex(RuntimeError, "format_version"):
            train._validate_checkpoint_metadata(missing_format_version, **common)

        unknown_format_version = checkpoint_for()
        unknown_format_version["training_metadata"]["format_version"] = 999
        with self.assertRaisesRegex(RuntimeError, "format_version"):
            train._validate_checkpoint_metadata(unknown_format_version, **common)

        missing_critical_hash = checkpoint_for()
        del missing_critical_hash["training_metadata"]["critical_config_sha256"]
        with self.assertRaisesRegex(RuntimeError, "configuration SHA-256"):
            train._validate_checkpoint_metadata(missing_critical_hash, **common)

        incompatible_critical_hash = checkpoint_for()
        incompatible_critical_hash["training_metadata"]["critical_config_sha256"] = (
            "b" * 64
        )
        with self.assertRaisesRegex(RuntimeError, "configuration SHA-256"):
            train._validate_checkpoint_metadata(incompatible_critical_hash, **common)

        with self.assertRaisesRegex(RuntimeError, "metadata epoch"):
            train._validate_checkpoint_metadata(checkpoint_for(epoch=1), **common)
        with self.assertRaisesRegex(RuntimeError, "source archive"):
            train._validate_checkpoint_metadata(
                checkpoint_for(archive="wrong.pgn.zst"), **common
            )
        with self.assertRaisesRegex(RuntimeError, "source SHA-256"):
            train._validate_checkpoint_metadata(
                checkpoint_for(digest="b" * 64), **common
            )

        for key, changed_value in (
            ("dim_cnn", 999),
            ("clock_threshold", 999),
            ("value_coefficient", 9.0),
            ("lr", 9.0),
        ):
            with self.subTest(key=key):
                saved_config = vars(cfg).copy()
                saved_config[key] = changed_value
                with self.assertRaisesRegex(RuntimeError, f"critical[\\s\\S]*{key}"):
                    train._validate_checkpoint_metadata(
                        checkpoint_for(config=saved_config), **common
                    )

        with self.assertWarnsRegex(RuntimeWarning, "legacy checkpoint"):
            train._validate_checkpoint_metadata({}, **common)

    @staticmethod
    def _resume_config(**overrides):
        values = dict(
            input_channels=18,
            dim_cnn=8,
            num_blocks_cnn=1,
            vit_length=2,
            dim_vit=32,
            num_blocks_vit=1,
            elo_dim=8,
            first_n_moves=10,
            last_n_moves=10,
            max_ply=300,
            clock_threshold=30,
            max_games_per_elo_range=20,
            side_info=True,
            side_info_coefficient=1.0,
            value=True,
            value_coefficient=1.0,
            lr=1e-4,
            wd=1e-5,
            batch_size=8192,
            seed=42,
            start_year=2023,
            start_month=1,
            end_year=2023,
            end_month=3,
            skip_months=[],
            chunk_size=20_000,
            num_cpu_left=16,
            checkpoint_epoch=2,
            checkpoint_year=2023,
            checkpoint_month=1,
        )
        values.update(overrides)
        return SimpleNamespace(**values)


if __name__ == "__main__":
    unittest.main()
