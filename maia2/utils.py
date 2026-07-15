import chess
import hashlib
import json
import os
import random
import re
import tempfile
from pathlib import Path

import numpy as np
import pyzstd
import torch
import time
import yaml


class Config:
    def __init__(self, config_dict):
        for key, value in config_dict.items():
            setattr(self, key, value)


def parse_args(cfg_file_path):
    with open(cfg_file_path, "r", encoding="utf-8") as f:
        cfg_dict = yaml.safe_load(f)

    cfg = Config(cfg_dict)

    return cfg


def seed_everything(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def delete_file(filename):
    if os.path.exists(filename):
        os.remove(filename)
        print(f"Data {filename} has been deleted.")
    else:
        print(f"The file '{filename}' does not exist.")


def readable_num(num):
    if num >= 1e9:  # if parameters are in the billions
        return f"{num / 1e9:.2f}B"
    elif num >= 1e6:  # if parameters are in the millions
        return f"{num / 1e6:.2f}M"
    elif num >= 1e3:  # if parameters are in the thousands
        return f"{num / 1e3:.2f}K"
    else:
        return str(num)


def readable_time(elapsed_time):
    hours, rem = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(rem, 60)

    if hours > 0:
        return f"{int(hours)}h {int(minutes)}m {seconds:.2f}s"
    elif minutes > 0:
        return f"{int(minutes)}m {seconds:.2f}s"
    else:
        return f"{seconds:.2f}s"


def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return readable_num(total_params)


def create_elo_dict():
    inteval = 100
    start = 1100
    end = 2000

    range_dict = {f"<{start}": 0}
    range_index = 1

    for lower_bound in range(start, end - 1, inteval):
        upper_bound = lower_bound + inteval
        range_dict[f"{lower_bound}-{upper_bound - 1}"] = range_index
        range_index += 1

    range_dict[f">={end}"] = range_index

    # print(range_dict, flush=True)

    return range_dict


def map_to_category(elo, elo_dict):
    inteval = 100
    start = 1100
    end = 2000

    if elo < start:
        return elo_dict[f"<{start}"]
    elif elo >= end:
        return elo_dict[f">={end}"]
    else:
        for lower_bound in range(start, end - 1, inteval):
            upper_bound = lower_bound + inteval
            if lower_bound <= elo < upper_bound:
                return elo_dict[f"{lower_bound}-{upper_bound - 1}"]


def get_side_info(board, move_uci, all_moves_dict):
    move = chess.Move.from_uci(move_uci)

    moving_piece = board.piece_at(move.from_square)
    captured_piece = board.piece_at(move.to_square)
    if board.is_en_passant(move):
        captured_piece = chess.Piece(chess.PAWN, not board.turn)

    from_square_encoded = torch.zeros(64)
    from_square_encoded[move.from_square] = 1

    to_square_encoded = torch.zeros(64)
    to_square_encoded[move.to_square] = 1

    if move_uci == "e1g1":
        rook_move = chess.Move.from_uci("h1f1")
        from_square_encoded[rook_move.from_square] = 1
        to_square_encoded[rook_move.to_square] = 1

    if move_uci == "e1c1":
        rook_move = chess.Move.from_uci("a1d1")
        from_square_encoded[rook_move.from_square] = 1
        to_square_encoded[rook_move.to_square] = 1

    board.push(move)
    is_check = board.is_check()
    board.pop()

    # Order: Pawn, Knight, Bishop, Rook, Queen, King
    side_info = torch.zeros(6 + 6 + 1)
    side_info[moving_piece.piece_type - 1] = 1
    if move_uci in ["e1g1", "e1c1"]:
        side_info[3] = 1
    if captured_piece:
        side_info[6 + captured_piece.piece_type - 1] = 1
    if is_check:
        side_info[-1] = 1

    legal_moves = torch.zeros(len(all_moves_dict))
    legal_moves_idx = torch.tensor(
        [all_moves_dict[move.uci()] for move in board.legal_moves]
    )
    legal_moves[legal_moves_idx] = 1

    side_info = torch.cat(
        [side_info, from_square_encoded, to_square_encoded, legal_moves], dim=0
    )

    return legal_moves, side_info


def extract_clock_time(comment):
    """Return a Lichess clock annotation in seconds.

    Lichess clock comments normally use ``[%clk H:MM:SS]`` and may contain
    fractional seconds. Malformed or absent annotations return ``0.0`` so a
    position fails the training clock threshold instead of crashing the
    preprocessing worker.
    """

    match = re.search(r"\[%clk\s+(\d+):(\d+):(\d+(?:\.\d+)?)\]", comment or "")
    if match:
        hours, minutes = map(int, match.groups()[:2])
        seconds = float(match.group(3))
        return hours * 3600 + minutes * 60 + seconds
    return 0.0


def _validate_chunks(chunks, source_size, allow_empty=False):
    if not chunks:
        if allow_empty:
            return []
        raise ValueError("A non-empty PGN source cannot have an empty chunk cache.")

    validated = []
    previous_end = 0
    for chunk in chunks:
        if not isinstance(chunk, (list, tuple)) or len(chunk) != 2:
            raise ValueError("Invalid PGN chunk cache entry.")
        start, end = chunk
        if not isinstance(start, int) or not isinstance(end, int):
            raise ValueError("PGN chunk offsets must be integers.")
        if start != previous_end or end <= start or end > source_size:
            raise ValueError(
                "PGN chunk offsets must provide exact contiguous source coverage."
            )
        validated.append((start, end))
        previous_end = end
    if previous_end != source_size:
        raise ValueError("PGN chunk cache does not cover the complete source file.")
    return validated


def _is_whitespace_only(file_path):
    with Path(file_path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            if block.strip():
                return False
    return True


def _normalize_sha256_fingerprint(source_fingerprint):
    if not isinstance(source_fingerprint, str):
        raise TypeError("source_fingerprint must be a SHA-256 string.")

    digest = source_fingerprint.removeprefix("sha256:").lower()
    if re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        raise ValueError(
            "source_fingerprint must contain a 64-character SHA-256 digest."
        )
    return f"sha256:{digest}"


def read_or_create_chunks(pgn_path, cfg, source_fingerprint=None):
    """Read cached PGN offsets, or create a validated JSON cache.

    The chunk size, source size, and a strong source fingerprint are part of
    the cache identity. By default, the PGN is hashed with SHA-256. A caller
    that already verified the PGN may provide its SHA-256 digest through
    ``source_fingerprint`` to avoid hashing a large file again. JSON avoids
    executing arbitrary pickle payloads from a shared data directory.
    """

    pgn_path = Path(pgn_path)
    initial_stat = pgn_path.stat()
    source_size = initial_stat.st_size
    if source_fingerprint is None:
        source_fingerprint, hashed_size = _file_sha256_and_size(pgn_path)
        if hashed_size != source_size or not _same_file_identity(
            initial_stat, pgn_path.stat()
        ):
            raise RuntimeError(f"PGN source changed while hashing: {pgn_path}")
    source_fingerprint = _normalize_sha256_fingerprint(source_fingerprint)
    cache_file = pgn_path.with_name(f"{pgn_path.stem}_chunks_{cfg.chunk_size}.json")

    if cache_file.exists():
        print(f"Loading cached chunks from {cache_file}")
        try:
            with cache_file.open("r", encoding="utf-8") as cache:
                payload = json.load(cache)
            if not isinstance(payload, dict):
                raise ValueError("PGN chunk cache must be a JSON object.")
            if payload.get("version") != 2:
                raise ValueError("Unsupported PGN chunk cache version.")
            if payload.get("chunk_size") != cfg.chunk_size:
                raise ValueError("PGN chunk cache uses a different chunk size.")
            if payload.get("source_size") != source_size:
                raise ValueError("PGN chunk cache source size has changed.")
            if payload.get("source_fingerprint") != source_fingerprint:
                raise ValueError("PGN chunk cache source fingerprint has changed.")
            cached_chunks = payload.get("chunks", [])
            validated_chunks = _validate_chunks(
                cached_chunks,
                source_size,
                allow_empty=not cached_chunks and _is_whitespace_only(pgn_path),
            )
            if not _same_file_identity(initial_stat, pgn_path.stat()):
                raise RuntimeError(
                    f"PGN source changed while validating its cache: {pgn_path}"
                )
            return validated_chunks
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
            print(
                f"Ignoring invalid chunk cache {cache_file}: {error}",
                flush=True,
            )

    print(f"Cache not found. Creating chunks for {pgn_path}")
    start_time = time.time()
    pgn_chunks = get_chunks(pgn_path, cfg.chunk_size)
    final_stat = pgn_path.stat()
    if not _same_file_identity(initial_stat, final_stat):
        raise RuntimeError(f"PGN source changed while it was being chunked: {pgn_path}")
    print(
        f"Chunking took {readable_time(time.time() - start_time)}",
        flush=True,
    )

    payload = {
        "version": 2,
        "chunk_size": cfg.chunk_size,
        "source_size": source_size,
        "source_fingerprint": source_fingerprint,
        "chunks": pgn_chunks,
    }
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    temporary_cache = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=cache_file.parent,
            prefix=f".{cache_file.name}.",
            suffix=".tmp",
            delete=False,
        ) as cache:
            temporary_cache = Path(cache.name)
            json.dump(payload, cache)
            cache.flush()
            os.fsync(cache.fileno())
        os.replace(temporary_cache, cache_file)
    except BaseException:
        if temporary_cache is not None:
            temporary_cache.unlink(missing_ok=True)
        raise

    return pgn_chunks


def board_to_tensor(board):
    piece_types = [
        chess.PAWN,
        chess.KNIGHT,
        chess.BISHOP,
        chess.ROOK,
        chess.QUEEN,
        chess.KING,
    ]
    num_piece_channels = 12  # 6 piece types * 2 colors
    additional_channels = (
        6  # 1 for player's turn, 4 for castling rights, 1 for en passant
    )
    tensor = torch.zeros(
        (num_piece_channels + additional_channels, 8, 8), dtype=torch.float32
    )

    # Precompute indices for each piece type
    piece_indices = {piece: i for i, piece in enumerate(piece_types)}

    # Fill tensor for each piece type
    for piece_type in piece_types:
        for color in [True, False]:  # True is White, False is Black
            piece_map = board.pieces(piece_type, color)
            index = piece_indices[piece_type] + (0 if color else 6)
            for square in piece_map:
                row, col = divmod(square, 8)
                tensor[index, row, col] = 1.0

    # Player's turn channel (White = 1, Black = 0)
    turn_channel = num_piece_channels
    if board.turn == chess.WHITE:
        tensor[turn_channel, :, :] = 1.0

    # Castling rights channels
    castling_rights = [
        board.has_kingside_castling_rights(chess.WHITE),
        board.has_queenside_castling_rights(chess.WHITE),
        board.has_kingside_castling_rights(chess.BLACK),
        board.has_queenside_castling_rights(chess.BLACK),
    ]
    for i, has_right in enumerate(castling_rights):
        if has_right:
            tensor[num_piece_channels + 1 + i, :, :] = 1.0

    # En passant target channel
    ep_channel = num_piece_channels + 5
    if board.ep_square is not None:
        row, col = divmod(board.ep_square, 8)
        tensor[ep_channel, row, col] = 1.0

    return tensor


def generate_pawn_promotions():
    # Define the promotion rows for both colors and the promotion pieces
    # promotion_rows = {'white': '7', 'black': '2'}
    promotion_rows = {"white": "7"}
    promotion_pieces = ["q", "r", "b", "n"]
    promotions = []

    # Iterate over each color
    for color, row in promotion_rows.items():
        # Target rows for promotion (8 for white, 1 for black)
        target_row = "8" if color == "white" else "1"

        # Each file from 'a' to 'h'
        for file in "abcdefgh":
            # Direct move to promotion
            for piece in promotion_pieces:
                promotions.append(f"{file}{row}{file}{target_row}{piece}")

            # Capturing moves to the left and right (if not on the edges of the board)
            if file != "a":
                left_file = chr(ord(file) - 1)  # File to the left
                for piece in promotion_pieces:
                    promotions.append(f"{file}{row}{left_file}{target_row}{piece}")

            if file != "h":
                right_file = chr(ord(file) + 1)  # File to the right
                for piece in promotion_pieces:
                    promotions.append(f"{file}{row}{right_file}{target_row}{piece}")

    return promotions


def mirror_square(square):
    file = square[0]
    rank = str(9 - int(square[1]))

    return file + rank


def mirror_move(move_uci):
    # Check if the move is a promotion (length of UCI string will be more than 4)
    is_promotion = len(move_uci) > 4

    # Extract the start and end squares, and the promotion piece if applicable
    start_square = move_uci[:2]
    end_square = move_uci[2:4]
    promotion_piece = move_uci[4:] if is_promotion else ""

    # Mirror the start and end squares
    mirrored_start = mirror_square(start_square)
    mirrored_end = mirror_square(end_square)

    # Return the mirrored move, including the promotion piece if applicable
    return mirrored_start + mirrored_end + promotion_piece


def get_chunks(pgn_path, chunk_size):
    if chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer.")

    chunks = []
    with open(pgn_path, "r", encoding="utf-8") as pgn_file:
        while True:
            start_pos = pgn_file.tell()
            game_count = 0
            saw_content = False
            while game_count < chunk_size:
                line = pgn_file.readline()
                if not line:
                    break
                saw_content = saw_content or bool(line.strip())
                stripped_line = line.rstrip()
                if stripped_line.endswith("1-0") or stripped_line.endswith("0-1"):
                    game_count += 1
                elif stripped_line.endswith("1/2-1/2"):
                    game_count += 1
                elif stripped_line.endswith("*"):
                    game_count += 1

            if game_count == 0 and not saw_content:
                if chunks:
                    chunks[-1] = (chunks[-1][0], pgn_file.tell())
                break
            if game_count == 0:
                raise ValueError(
                    f"No complete PGN games were found after byte {start_pos} "
                    f"in {pgn_path}."
                )

            line = pgn_file.readline()
            if line not in ["\n", ""]:
                raise ValueError(
                    f"Expected a blank line between PGN games in {pgn_path}."
                )
            end_pos = pgn_file.tell()
            chunks.append((start_pos, end_pos))
            if not line:
                break

    return chunks


def decompression_provenance_path(decompressed_path):
    """Return the JSON sidecar path for an atomically decompressed file."""

    decompressed_path = Path(decompressed_path)
    return decompressed_path.with_name(f"{decompressed_path.name}.provenance.json")


def _valid_provenance_payload(payload):
    if not isinstance(payload, dict) or payload.get("version") != 1:
        return False
    for section in ("archive", "decompressed"):
        identity = payload.get(section)
        if not isinstance(identity, dict):
            return False
        if not isinstance(identity.get("size"), int) or identity["size"] < 0:
            return False
        digest = identity.get("sha256")
        if (
            not isinstance(digest, str)
            or re.fullmatch(r"[0-9a-f]{64}", digest.lower()) is None
        ):
            return False
    return True


def read_decompression_provenance(decompressed_path):
    """Read a valid decompression sidecar, returning ``None`` if unavailable."""

    provenance_path = decompression_provenance_path(decompressed_path)
    try:
        with provenance_path.open("r", encoding="utf-8") as provenance_file:
            payload = json.load(provenance_file)
    except (OSError, json.JSONDecodeError):
        return None
    return payload if _valid_provenance_payload(payload) else None


def _write_json_atomically(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            json.dump(payload, temporary_file, sort_keys=True)
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        os.replace(temporary_path, path)
    except BaseException:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise


def _file_sha256_and_size(file_path):
    file_path = Path(file_path)
    initial_stat = file_path.stat()
    digest = sha256_file(file_path)
    final_stat = file_path.stat()
    if (
        final_stat.st_size != initial_stat.st_size
        or final_stat.st_mtime_ns != initial_stat.st_mtime_ns
        or final_stat.st_ino != initial_stat.st_ino
    ):
        raise RuntimeError(f"File changed while it was being hashed: {file_path}")
    return digest, final_stat.st_size


def _same_file_identity(first_stat, second_stat):
    return (
        second_stat.st_size == first_stat.st_size
        and second_stat.st_mtime_ns == first_stat.st_mtime_ns
        and second_stat.st_ino == first_stat.st_ino
    )


def decompress_zst(file_path, decompressed_path, reuse_existing=False):
    """Atomically decompress zstd data and record cryptographic provenance.

    ``reuse_existing`` returns ``False`` only when the current archive's size
    and SHA-256 match the sidecar and a freshly computed SHA-256 verifies the
    decompressed output. Missing, stale, or malformed provenance safely causes
    a new atomic decompression. A newly installed output returns ``True``.
    """

    file_path = Path(file_path)
    decompressed_path = Path(decompressed_path)
    if not file_path.is_file():
        raise FileNotFoundError(f"Compressed training data not found: {file_path}")
    archive_stat = file_path.stat()
    archive_sha256, archive_size = _file_sha256_and_size(file_path)
    hashed_archive_stat = file_path.stat()
    if not _same_file_identity(archive_stat, hashed_archive_stat):
        raise RuntimeError(f"Compressed source changed while hashing: {file_path}")
    archive_stat = hashed_archive_stat

    if reuse_existing and decompressed_path.is_file():
        provenance = read_decompression_provenance(decompressed_path)
        if provenance is not None:
            expected_archive = provenance["archive"]
            expected_decompressed = provenance["decompressed"]
            if (
                expected_archive["size"] == archive_size
                and expected_archive["sha256"].lower() == archive_sha256
                and expected_decompressed["size"] == decompressed_path.stat().st_size
            ):
                decompressed_sha256, decompressed_size = _file_sha256_and_size(
                    decompressed_path
                )
                if (
                    expected_decompressed["size"] == decompressed_size
                    and expected_decompressed["sha256"].lower() == decompressed_sha256
                ):
                    if not _same_file_identity(archive_stat, file_path.stat()):
                        raise RuntimeError(
                            f"Compressed source changed during validation: {file_path}"
                        )
                    return False

        print(
            f"Existing decompressed file lacks matching verified provenance; "
            f"rebuilding {decompressed_path}.",
            flush=True,
        )

    decompressed_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "wb",
        dir=decompressed_path.parent,
        prefix=f".{decompressed_path.name}.",
        suffix=".tmp",
        delete=False,
    ) as temporary_file:
        temporary_path = Path(temporary_file.name)

    try:
        decompressed_digest = hashlib.sha256()
        decompressed_size = 0
        with pyzstd.ZstdFile(file_path, "rb") as compressed_file:
            with temporary_path.open("wb") as decompressed_file:
                for block in iter(lambda: compressed_file.read(1024 * 1024), b""):
                    decompressed_file.write(block)
                    decompressed_digest.update(block)
                    decompressed_size += len(block)
                if decompressed_size == 0:
                    raise RuntimeError(
                        f"Refusing to install empty decompressed data: {file_path}"
                    )
                decompressed_file.flush()
                os.fsync(decompressed_file.fileno())
        final_archive_stat = file_path.stat()
        if not _same_file_identity(archive_stat, final_archive_stat):
            raise RuntimeError(
                f"Compressed source changed while it was being decompressed: {file_path}"
            )
        os.replace(temporary_path, decompressed_path)
        provenance = {
            "version": 1,
            "archive": {
                "name": file_path.name,
                "size": archive_size,
                "sha256": archive_sha256,
            },
            "decompressed": {
                "name": decompressed_path.name,
                "size": decompressed_size,
                "sha256": decompressed_digest.hexdigest(),
            },
        }
        _write_json_atomically(
            decompression_provenance_path(decompressed_path), provenance
        )
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise
    return True


def sha256_file(file_path):
    """Return a streaming SHA-256 digest for a local file."""

    digest = hashlib.sha256()
    with Path(file_path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _gdown_download(url, output_path, quiet):
    import gdown

    return gdown.download(url, output_path, quiet=quiet)


def download_google_drive_file(url, output_path, sha256=None, quiet=False):
    """Download a Google Drive file atomically and optionally verify SHA-256.

    Existing files are verified as well. A failed or interrupted download never
    replaces a valid destination file.
    """

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    def validate(path):
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"Downloaded file is missing or empty: {path}")
        if sha256 is not None:
            actual_sha256 = sha256_file(path)
            if actual_sha256.lower() != sha256.lower():
                raise RuntimeError(
                    f"SHA-256 mismatch for {path}: expected {sha256}, "
                    f"got {actual_sha256}."
                )

    if output_path.exists():
        try:
            validate(output_path)
            return str(output_path)
        except RuntimeError as error:
            print(
                f"Existing asset failed validation; downloading a verified "
                f"replacement for {output_path}: {error}",
                flush=True,
            )

    with tempfile.NamedTemporaryFile(
        "wb",
        dir=output_path.parent,
        prefix=f".{output_path.name}.",
        suffix=".part",
        delete=False,
    ) as temporary_file:
        temporary_path = Path(temporary_file.name)

    try:
        downloaded_path = _gdown_download(
            url,
            str(temporary_path),
            quiet,
        )
        if downloaded_path is None:
            raise RuntimeError(f"Google Drive download failed for {url}")
        validate(temporary_path)
        os.replace(temporary_path, output_path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise

    return str(output_path)


def get_all_possible_moves():
    all_moves = []

    for rank in range(8):
        for file in range(8):
            square = chess.square(file, rank)

            board = chess.Board(None)
            board.set_piece_at(square, chess.Piece(chess.QUEEN, chess.WHITE))
            legal_moves = list(board.legal_moves)
            all_moves.extend(legal_moves)

            board = chess.Board(None)
            board.set_piece_at(square, chess.Piece(chess.KNIGHT, chess.WHITE))
            legal_moves = list(board.legal_moves)
            all_moves.extend(legal_moves)

    all_moves = [all_moves[i].uci() for i in range(len(all_moves))]

    pawn_promotions = generate_pawn_promotions()

    return all_moves + pawn_promotions


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]
