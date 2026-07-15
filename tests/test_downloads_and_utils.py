import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pyzstd

from maia2.utils import decompression_provenance_path, decompress_zst
from maia2.utils import download_google_drive_file, read_decompression_provenance
from maia2.utils import get_chunks, read_or_create_chunks


PGN_WHITE_WIN = """[Event "Rated Rapid game"]
[WhiteElo "1500"]
[BlackElo "1500"]
[Result "1-0"]

1. e4 e5 1-0

"""
PGN_BLACK_WIN = PGN_WHITE_WIN.replace('Result "1-0"', 'Result "0-1"').replace(
    "e5 1-0", "e5 0-1"
)


class DownloadAndCompressionTest(unittest.TestCase):
    def test_download_is_atomic_and_checksum_verified(self):
        contents = b"verified maia2 asset"
        expected_hash = hashlib.sha256(contents).hexdigest()

        def fake_download(url, output, quiet):
            Path(output).write_bytes(contents)
            return output

        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory, "asset.bin")
            with mock.patch(
                "maia2.utils._gdown_download", side_effect=fake_download
            ) as download:
                result = download_google_drive_file(
                    "https://example.invalid/asset",
                    destination,
                    sha256=expected_hash,
                )

            self.assertEqual(result, str(destination))
            self.assertEqual(destination.read_bytes(), contents)
            download.assert_called_once()
            self.assertEqual(list(Path(directory).glob("*.part")), [])

    def test_existing_valid_download_is_reused(self):
        contents = b"existing asset"
        expected_hash = hashlib.sha256(contents).hexdigest()

        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory, "asset.bin")
            destination.write_bytes(contents)
            with mock.patch("maia2.utils._gdown_download") as download:
                download_google_drive_file(
                    "https://example.invalid/asset",
                    destination,
                    sha256=expected_hash,
                )

            download.assert_not_called()

    def test_existing_invalid_download_is_replaced_only_after_verification(self):
        contents = b"verified replacement"
        expected_hash = hashlib.sha256(contents).hexdigest()

        def fake_download(url, output, quiet):
            Path(output).write_bytes(contents)
            return output

        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory, "asset.bin")
            destination.write_bytes(b"stale partial download")
            with mock.patch(
                "maia2.utils._gdown_download", side_effect=fake_download
            ) as download:
                download_google_drive_file(
                    "https://example.invalid/asset",
                    destination,
                    sha256=expected_hash,
                )

            self.assertEqual(destination.read_bytes(), contents)
            download.assert_called_once()
            self.assertEqual(list(Path(directory).glob("*.part")), [])

    def test_failed_replacement_preserves_an_existing_invalid_asset(self):
        stale_contents = b"stale partial download"

        def fake_download(url, output, quiet):
            Path(output).write_bytes(b"also invalid")
            return output

        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory, "asset.bin")
            destination.write_bytes(stale_contents)
            with (
                mock.patch("maia2.utils._gdown_download", side_effect=fake_download),
                self.assertRaisesRegex(RuntimeError, "SHA-256 mismatch"),
            ):
                download_google_drive_file(
                    "https://example.invalid/asset",
                    destination,
                    sha256="0" * 64,
                )

            self.assertEqual(destination.read_bytes(), stale_contents)
            self.assertEqual(list(Path(directory).glob("*.part")), [])

    def test_checksum_failure_does_not_install_download(self):
        def fake_download(url, output, quiet):
            Path(output).write_bytes(b"wrong contents")
            return output

        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory, "asset.bin")
            with (
                mock.patch("maia2.utils._gdown_download", side_effect=fake_download),
                self.assertRaisesRegex(RuntimeError, "SHA-256 mismatch"),
            ):
                download_google_drive_file(
                    "https://example.invalid/asset",
                    destination,
                    sha256="0" * 64,
                )

            self.assertFalse(destination.exists())
            self.assertEqual(list(Path(directory).glob("*.part")), [])

    def test_zstd_decompression_replaces_destination_atomically(self):
        contents = b"lichess training data\n" * 100
        with tempfile.TemporaryDirectory() as directory:
            compressed = Path(directory, "games.pgn.zst")
            destination = Path(directory, "games.pgn")
            compressed.write_bytes(pyzstd.compress(contents))
            destination.write_bytes(b"old contents")

            decompress_zst(compressed, destination)

            self.assertEqual(destination.read_bytes(), contents)
            self.assertEqual(list(Path(directory).glob("*.tmp")), [])
            provenance = read_decompression_provenance(destination)
            self.assertIsNotNone(provenance)
            self.assertEqual(
                provenance["archive"]["sha256"],
                hashlib.sha256(compressed.read_bytes()).hexdigest(),
            )
            self.assertEqual(provenance["archive"]["size"], compressed.stat().st_size)
            self.assertEqual(
                provenance["decompressed"]["sha256"],
                hashlib.sha256(contents).hexdigest(),
            )
            self.assertEqual(provenance["decompressed"]["size"], len(contents))
            self.assertEqual(
                decompression_provenance_path(destination),
                Path(f"{destination}.provenance.json"),
            )

    def test_zstd_decompression_can_explicitly_reuse_completed_output(self):
        with tempfile.TemporaryDirectory() as directory:
            compressed = Path(directory, "games.pgn.zst")
            destination = Path(directory, "games.pgn")
            compressed.write_bytes(pyzstd.compress(b"new contents"))
            self.assertTrue(decompress_zst(compressed, destination))

            created = decompress_zst(
                compressed,
                destination,
                reuse_existing=True,
            )

            self.assertFalse(created)
            self.assertEqual(destination.read_bytes(), b"new contents")

    def test_zstd_reuse_rebuilds_an_empty_or_unprovenanced_output(self):
        with tempfile.TemporaryDirectory() as directory:
            compressed = Path(directory, "games.pgn.zst")
            destination = Path(directory, "games.pgn")
            compressed.write_bytes(pyzstd.compress(b"new contents"))
            destination.write_bytes(b"")

            created = decompress_zst(compressed, destination, reuse_existing=True)

            self.assertTrue(created)
            self.assertEqual(destination.read_bytes(), b"new contents")
            self.assertIsNotNone(read_decompression_provenance(destination))

    def test_zstd_reuse_detects_same_size_decompressed_tampering(self):
        contents = b"original pgn"
        with tempfile.TemporaryDirectory() as directory:
            compressed = Path(directory, "games.pgn.zst")
            destination = Path(directory, "games.pgn")
            compressed.write_bytes(pyzstd.compress(contents))
            decompress_zst(compressed, destination)
            destination.write_bytes(b"tampered pgn")
            self.assertEqual(len(contents), destination.stat().st_size)

            created = decompress_zst(compressed, destination, reuse_existing=True)

            self.assertTrue(created)
            self.assertEqual(destination.read_bytes(), contents)

    def test_zstd_reuse_detects_archive_replacement(self):
        with tempfile.TemporaryDirectory() as directory:
            compressed = Path(directory, "games.pgn.zst")
            destination = Path(directory, "games.pgn")
            compressed.write_bytes(pyzstd.compress(b"first archive"))
            decompress_zst(compressed, destination)
            first_provenance = read_decompression_provenance(destination)

            compressed.write_bytes(pyzstd.compress(b"other archive"))
            created = decompress_zst(compressed, destination, reuse_existing=True)
            second_provenance = read_decompression_provenance(destination)

            self.assertTrue(created)
            self.assertEqual(destination.read_bytes(), b"other archive")
            self.assertNotEqual(
                first_provenance["archive"]["sha256"],
                second_provenance["archive"]["sha256"],
            )

    def test_zstd_reuse_rebuilds_when_sidecar_is_malformed(self):
        with tempfile.TemporaryDirectory() as directory:
            compressed = Path(directory, "games.pgn.zst")
            destination = Path(directory, "games.pgn")
            compressed.write_bytes(pyzstd.compress(b"verified output"))
            decompress_zst(compressed, destination)
            decompression_provenance_path(destination).write_text(
                "not JSON", encoding="utf-8"
            )

            created = decompress_zst(compressed, destination, reuse_existing=True)

            self.assertTrue(created)
            self.assertEqual(destination.read_bytes(), b"verified output")
            self.assertIsNotNone(read_decompression_provenance(destination))

    def test_chunk_cache_rejects_same_size_changed_pgn(self):
        self.assertEqual(len(PGN_WHITE_WIN), len(PGN_BLACK_WIN))
        with tempfile.TemporaryDirectory() as directory:
            pgn = Path(directory, "games.pgn")
            cfg = SimpleNamespace(chunk_size=1)
            pgn.write_text(PGN_WHITE_WIN, encoding="utf-8")
            read_or_create_chunks(pgn, cfg)
            cache = Path(directory, "games_chunks_1.json")
            first_payload = json.loads(cache.read_text(encoding="utf-8"))

            pgn.write_text(PGN_BLACK_WIN, encoding="utf-8")
            with mock.patch("maia2.utils.get_chunks", wraps=get_chunks) as chunker:
                read_or_create_chunks(pgn, cfg)
            second_payload = json.loads(cache.read_text(encoding="utf-8"))

            chunker.assert_called_once()
            self.assertEqual(
                first_payload["source_size"], second_payload["source_size"]
            )
            self.assertNotEqual(
                first_payload["source_fingerprint"],
                second_payload["source_fingerprint"],
            )

    def test_chunk_cache_accepts_caller_supplied_sha256_fingerprint(self):
        with tempfile.TemporaryDirectory() as directory:
            pgn = Path(directory, "games.pgn")
            pgn.write_text(PGN_WHITE_WIN, encoding="utf-8")
            fingerprint = hashlib.sha256(pgn.read_bytes()).hexdigest()

            with mock.patch(
                "maia2.utils.sha256_file",
                side_effect=AssertionError("PGN should not be hashed again"),
            ):
                read_or_create_chunks(
                    pgn,
                    SimpleNamespace(chunk_size=1),
                    source_fingerprint=fingerprint,
                )

            payload = json.loads(
                Path(directory, "games_chunks_1.json").read_text(encoding="utf-8")
            )
            self.assertEqual(payload["version"], 2)
            self.assertEqual(payload["source_fingerprint"], f"sha256:{fingerprint}")


if __name__ == "__main__":
    unittest.main()
