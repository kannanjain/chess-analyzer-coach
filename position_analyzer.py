#!/usr/bin/env python3

import argparse
import os
import sys
import chess
import chess.engine


def get_stockfish_path():
    """Find stockfish binary."""
    if os.environ.get("STOCKFISH_PATH"):
        return os.environ["STOCKFISH_PATH"]
    local = "./stockfish/stockfish-macos-m1-apple-silicon"
    if os.path.exists(local):
        return local
    return "stockfish"


def pv_to_san(board, pv):
    """Convert a list of moves to standard algebraic notation."""
    san_moves = []
    temp = board.copy()
    for move in pv:
        try:
            san_moves.append(temp.san(move))
            temp.push(move)
        except Exception:
            break
    return san_moves


def format_eval(score):
    """Format engine score from white's perspective into readable text."""
    if score is None:
        return "unknown"

    white_score = score.white()

    if white_score.is_mate(): #true if mate flase otherwise
        mate_in = white_score.mate() #number of moves until mate, positive if white mates, negative if black mates
        if mate_in > 0:
            return f"White mates in {mate_in}"
        else:
            return f"Black mates in {-mate_in}"

    cp = white_score.score()
    if cp is None:
        return "unknown"
    pawns = cp / 100.0
    if pawns > 0:
        return f"+{pawns:.2f} (White is better)"
    elif pawns < 0:
        return f"{pawns:.2f} (Black is better)"
    return "0.00 (equal)"


PIECE_VALUES = {
    chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
    chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0,
}

PIECE_NAMES = {
    chess.PAWN: "pawn", chess.KNIGHT: "knight", chess.BISHOP: "bishop",
    chess.ROOK: "rook", chess.QUEEN: "queen", chess.KING: "king",
}


def detect_double_attack(board, move):
    """Detect forks and double attacks.

    Cases:
    1. Knight attacks 2+ pieces where the realized gain after the opponent saves
       the best target is >= 1.5 pawns → 'fork'
    2. Any other piece in the same situation → 'double attack'
    3. Piece attacks 1 valuable target AND a favorable exchange exists on the
       landing square (cheapest enemy recapture costs more than our piece, and
       we have a defender to recapture back) → 'double attack'

    For cases 1 & 2, "realized gain" is computed as follows:
    - For each attacked enemy piece, run a one-ply SEE (static exchange evaluation):
        undefended piece  → gain = full piece value
        defended piece    → gain = piece_value - our_piece_value
    - The opponent saves the most profitable target for us, so we capture the
      next best. That second-best gain is the realized gain.
    - King is always treated as a forcing threat (gain = 999) since check must
      be answered.
    """
    moving_piece = board.piece_at(move.from_square)
    if not moving_piece:
        return None

    attacker_value = PIECE_VALUES.get(moving_piece.piece_type, 0)
    attacker_name = PIECE_NAMES[moving_piece.piece_type]
    is_knight = moving_piece.piece_type == chess.KNIGHT

    temp = board.copy()
    temp.push(move)

    is_check = temp.is_check()

    # Find enemy pieces attacked by the moved piece from its new square
    attacked = []
    for sq in temp.attacks(move.to_square):
        target = temp.piece_at(sq)
        if target and target.color != moving_piece.color:
            attacked.append((sq, target))

    # Include king as a target if this is a direct check (not discovered)
    if is_check:
        king_sq = temp.king(not moving_piece.color)
        if king_sq is not None and move.to_square in temp.attackers(moving_piece.color, king_sq):
            king_piece = temp.piece_at(king_sq)
            if king_piece and (king_sq, king_piece) not in attacked:
                attacked.append((king_sq, king_piece))

    def capture_gain(target_sq, target_piece):
        """One-ply SEE: net material gain of capturing target_piece at target_sq."""
        if target_piece.piece_type == chess.KING:
            return 999  # Check is always forcing — opponent must respond
        target_val = PIECE_VALUES.get(target_piece.piece_type, 0)
        defenders = list(temp.attackers(not moving_piece.color, target_sq))
        if not defenders:
            return target_val          # Undefended: we take it for free
        return target_val - attacker_value  # Defended: one-ply exchange

    # Compute gain for every attacked piece, sorted best-first
    gains = sorted(
        [(sq, p, capture_gain(sq, p)) for sq, p in attacked],
        key=lambda x: x[2],
        reverse=True,
    )

    # Cases 1 & 2: 2+ attacked pieces with significant realized gain.
    # The opponent saves their most valuable threatened piece, so we capture
    # the next best. If that second-best gain >= 1.5 pawns, it's a real fork.
    SIGNIFICANT = 1.5
    if len(gains) >= 2 and gains[1][2] >= SIGNIFICANT:
        names = [PIECE_NAMES[p.piece_type] for _, p, _ in gains[:2]]
        label = "fork" if is_knight else "double attack"
        return f"{attacker_name} {label}s {' and '.join(names)}"

    # Case 3: 1 valuable target + favorable exchange on the landing square.
    # The opponent's cheapest recapture costs them more than our piece is worth,
    # AND we have a defender ready to recapture — so they can't safely take us,
    # creating two threats at once: win their piece or profit from the exchange.
    valuable_targets = [
        (sq, p) for sq, p in attacked
        if p.piece_type == chess.KING or PIECE_VALUES.get(p.piece_type, 0) > attacker_value
    ]
    if len(valuable_targets) == 1:
        enemy_attackers = [
            sq for sq in temp.attackers(not moving_piece.color, move.to_square)
            if temp.piece_at(sq)
        ]
        if enemy_attackers:
            min_enemy_val = min(
                PIECE_VALUES.get(temp.piece_at(sq).piece_type, 0)
                for sq in enemy_attackers
            )
            our_defenders = [
                sq for sq in temp.attackers(moving_piece.color, move.to_square)
                if sq != move.to_square
            ]
            if min_enemy_val > attacker_value and our_defenders:
                target_name = PIECE_NAMES[valuable_targets[0][1].piece_type]
                return (f"double attack: {attacker_name} threatens {target_name}, "
                        f"recapturing it loses material")

    return None


def detect_pin(board, move):
    """Detect pin-related tactics using is_pinned.

    Three cases:
    1. The move creates a new absolute pin on an opponent's piece — our slider
       lands on a ray that now runs: our piece → enemy piece → enemy king.
    2. The move attacks a piece that is already absolutely pinned and therefore
       cannot flee (it would expose the king).
    3. The move lands on a square that looks defended by an enemy piece, but
       that defender is pinned and cannot actually capture (ghost defender).
    """
    moving_piece = board.piece_at(move.from_square)
    if not moving_piece:
        return None

    opp_color = not moving_piece.color
    results = []

    # Case 3: ghost defender — evaluated on the board BEFORE the move so we
    # can see what nominally defends the destination square right now.
    enemy_attackers = list(board.attackers(opp_color, move.to_square))
    if enemy_attackers and all(board.is_pinned(opp_color, sq) for sq in enemy_attackers):
        defender_names = [PIECE_NAMES[board.piece_at(sq).piece_type] for sq in enemy_attackers]
        results.append(
            f"safe square: only defended by pinned {' and '.join(defender_names)}"
        )

    temp = board.copy()
    temp.push(move)

    # Case 1: creates a new pin — compare pin status before and after.
    newly_pinned = []
    for sq in chess.SQUARES:
        piece = temp.piece_at(sq)
        if not (piece and piece.color == opp_color):
            continue
        orig = board.piece_at(sq)
        was_pinned = bool(orig and orig.color == opp_color and board.is_pinned(opp_color, sq))
        if not was_pinned and temp.is_pinned(opp_color, sq):
            newly_pinned.append(PIECE_NAMES[piece.piece_type])
    if newly_pinned:
        results.append(f"pins opponent's {' and '.join(newly_pinned)}")

    # Case 2: attacks a piece that is pinned and cannot run away.
    attacked_pinned = []
    for sq in temp.attacks(move.to_square):
        target = temp.piece_at(sq)
        if target and target.color == opp_color and temp.is_pinned(opp_color, sq):
            attacked_pinned.append(PIECE_NAMES[target.piece_type])
    if attacked_pinned:
        results.append(f"attacks pinned {' and '.join(attacked_pinned)} (cannot flee)")

    return "; ".join(results) if results else None


def detect_discovered_attack(board, move):
    """Detect a discovered check where the moved piece attacks a valuable target.

    A discovered attack happens when you move a piece out of the way, revealing
    a check from a piece behind it (e.g., rook, bishop, queen).  The opponent
    must deal with the check and cannot save the piece attacked by the moved
    piece, as long as that target is worth more than the moved piece.
    """
    moving_piece = board.piece_at(move.from_square)
    if not moving_piece:
        return None

    temp = board.copy()
    temp.push(move)

    # Must be check for discovered attack to be forcing
    if not temp.is_check():
        return None

    # Find which piece is giving check — if it's NOT the moved piece, it's discovered
    enemy_king_sq = temp.king(not moving_piece.color)
    checkers = temp.attackers(moving_piece.color, enemy_king_sq)

    discovered_checker = None
    for sq in checkers:
        if sq != move.to_square:
            discovered_checker = temp.piece_at(sq)
            break

    if not discovered_checker:
        return None  # Direct check, not discovered

    # Check what the moved piece attacks that's worth more than itself
    moved_value = PIECE_VALUES.get(moving_piece.piece_type, 0)
    best_target = None
    best_target_value = 0
    for sq in temp.attacks(move.to_square):
        target = temp.piece_at(sq)
        if target and target.color != moving_piece.color:
            target_value = PIECE_VALUES.get(target.piece_type, 0)
            if target_value > moved_value and target_value > best_target_value:
                best_target = target
                best_target_value = target_value

    checker_name = PIECE_NAMES[discovered_checker.piece_type]
    mover_name = PIECE_NAMES[moving_piece.piece_type]
    if best_target:
        target_name = PIECE_NAMES[best_target.piece_type]
        return (f"discovered check from {checker_name}, "
                f"{mover_name} attacks {target_name}")
    return f"discovered check from {checker_name}"


def detect_double_check(board, move):
    """Detect a double check — the king is attacked by two pieces simultaneously.

    A double check always involves a discovered check: moving a piece reveals
    an attack from a piece behind it while the moved piece also gives check.
    It is the most forcing check because the king must move; it cannot block
    or capture both checkers at once.
    """
    moving_piece = board.piece_at(move.from_square)
    if not moving_piece:
        return None

    temp = board.copy()
    temp.push(move)

    if not temp.is_check():
        return None

    checkers = list(temp.checkers())
    if len(checkers) < 2:
        return None

    checker_names = [
        PIECE_NAMES[temp.piece_at(sq).piece_type]
        for sq in checkers
        if temp.piece_at(sq)
    ]
    return f"double check ({' and '.join(checker_names)} both give check)"


def setup_survives_response(temp, setup_move, our_color):
    """Return True only if a setup double attack still fires after the opponent moves
    any piece it threatens.

    temp        — board after our actual move (opponent to move)
    setup_move  — the candidate setup move (for our_color, found via null-move scan)

    Logic: find every enemy piece the setup_move would attack from its destination.
    For each, check every opponent legal move that takes that piece off its current
    square.  If any such escape makes detect_double_attack return None, the setup is
    too easy to avoid and we suppress it.
    """
    opp_color = not our_color

    # Replay the null-move to get a board where setup_move is legal
    our_board = temp.copy()
    our_board.push(chess.Move.null())
    if setup_move not in our_board.legal_moves:
        return False

    # Find pieces the setup threatens from its landing square
    after_setup = our_board.copy()
    after_setup.push(setup_move)
    threatened_squares = {
        sq for sq in after_setup.attacks(setup_move.to_square)
        if (p := after_setup.piece_at(sq))
        and p.color == opp_color
        and PIECE_VALUES.get(p.piece_type, 0) > 0
    }

    if not threatened_squares:
        return True

    # For each threatened piece, check if the opponent can move it to escape
    for opp_move in temp.legal_moves:
        if opp_move.from_square not in threatened_squares:
            continue
        temp_after_R = temp.copy()
        temp_after_R.push(opp_move)
        if setup_move not in temp_after_R.legal_moves:
            return False  # Setup move itself is now illegal (path blocked)
        if not detect_double_attack(temp_after_R, setup_move):
            return False  # Escaped — setup no longer fires

    return True  # No escape found; setup is real


def can_opponent_save_piece(board_after_move, target_sq, our_color, threshold=1.5):
    """Return True if opponent has at least one legal response that saves piece at target_sq.

    A piece is saved if after the opponent's move our net SEE gain on it (at its
    original or new square) drops below threshold, or we can no longer attack it.
    """
    target = board_after_move.piece_at(target_sq)
    if not target:
        return True

    opp_color = not our_color
    target_val = PIECE_VALUES.get(target.piece_type, 0)

    for opp_move in board_after_move.legal_moves:
        temp2 = board_after_move.copy()
        temp2.push(opp_move)

        if opp_move.from_square == target_sq:
            # Piece moved — check if the new square is actually safe
            new_sq = opp_move.to_square
            if not temp2.piece_at(new_sq):
                continue
            attackers_new = [s for s in temp2.attackers(our_color, new_sq) if temp2.piece_at(s)]
            if not attackers_new:
                return True  # Moved to a safe square
            min_att = min(PIECE_VALUES.get(temp2.piece_at(s).piece_type, 0) for s in attackers_new)
            recaps_new = [s for s in temp2.attackers(opp_color, new_sq) if temp2.piece_at(s)]
            net_new = target_val if not recaps_new else target_val - min_att
            if net_new < threshold:
                return True  # New square is sufficiently defended
        else:
            # Piece stayed — check if attack on original square is now unprofitable
            if not temp2.piece_at(target_sq):
                continue
            attackers_after = [s for s in temp2.attackers(our_color, target_sq) if temp2.piece_at(s)]
            if not attackers_after:
                return True  # Attack removed (blocker interposed or attacker captured)
            min_att_val = min(PIECE_VALUES.get(temp2.piece_at(s).piece_type, 0) for s in attackers_after)
            recaps = [s for s in temp2.attackers(opp_color, target_sq) if temp2.piece_at(s)]
            net = target_val if not recaps else target_val - min_att_val
            if net < threshold:
                return True  # Defended sufficiently

    return False  # No legal response saves the piece


def detect_created_threats(board, move):
    """Detect new material threats and tactical setups created by this move.

    Material threats: enemy pieces we can capture next turn for net SEE gain >= 1.5
    that we could NOT already capture before this move, AND that the opponent has
    no legal response to save.

    Tactical setups: double attacks / forks we can execute on our next turn,
    found by null-moving back to our turn after the move is played.

    Returns a list of descriptions (may be empty).
    """
    moving_piece = board.piece_at(move.from_square)
    if not moving_piece:
        return []

    our_color = moving_piece.color

    def winning_captures(b):
        """Return {square: (net_gain, target_piece)} for enemy pieces we can win with net >= 1.5."""
        found = {}
        for sq in chess.SQUARES:
            target = b.piece_at(sq)
            if not (target and target.color != our_color and target.piece_type != chess.KING):
                continue
            attackers = [s for s in b.attackers(our_color, sq) if b.piece_at(s)]
            if not attackers:
                continue
            min_att_val = min(PIECE_VALUES.get(b.piece_at(s).piece_type, 0) for s in attackers)
            target_val = PIECE_VALUES.get(target.piece_type, 0)
            recaps = [s for s in b.attackers(not our_color, sq) if b.piece_at(s)]
            net = target_val if not recaps else target_val - min_att_val
            if net >= 1.0:
                found[sq] = (net, target)
        return found

    pre = winning_captures(board)

    temp = board.copy()
    temp.push(move)

    results = []

    # New material threats (didn't exist before this move, and opponent cannot save the piece)
    for sq, (gain, target) in winning_captures(temp).items():
        if sq not in pre and not can_opponent_save_piece(temp, sq, our_color):
            results.append(
                f"threatens {PIECE_NAMES[target.piece_type]} on "
                f"{chess.square_name(sq)} (+{gain:.1f})"
            )

    # Tactical setups: null-move to get back to our turn, scan for double attacks.
    # Only report setups that are NEW — not ones that already existed before this move.
    pre_setups = set()
    if not board.is_check():
        pre_null = board.copy()
        pre_null.push(chess.Move.null())
        for pre_move in pre_null.legal_moves:
            da = detect_double_attack(pre_null, pre_move)
            if da:
                pre_setups.add(da)

    if not temp.is_check():
        # If the piece we just moved is immediately capturable, any setup starting
        # from that square is moot — the opponent would take it before we can follow up.
        piece_is_capturable = bool(list(temp.attackers(not our_color, move.to_square)))

        temp2 = temp.copy()
        temp2.push(chess.Move.null())
        seen = set()
        for next_move in temp2.legal_moves:
            if piece_is_capturable and next_move.from_square == move.to_square:
                continue
            da = detect_double_attack(temp2, next_move)
            if da and da not in seen and da not in pre_setups:
                if not setup_survives_response(temp, next_move, our_color):
                    continue
                seen.add(da)
                results.append(f"sets up {da}")

    return results


def detect_defended_threats(board, move):
    """Detect opponent threats that existed before this move but not after.

    Before the move: null-move to opponent's turn, find their most forcing
    candidates (high-SEE captures + checking moves), run tactical detectors.
    After the move: opponent's turn directly, same scan.
    Threats in before but not after → "defends against opponent's [threat]".

    Returns [] if currently in check (null move illegal).
    """
    if board.is_check():
        return []

    moving_piece = board.piece_at(move.from_square)
    if not moving_piece:
        return []

    def opponent_threats(b):
        """Set of threat strings for the side to move in b."""
        threats = set()
        candidates = []

        for m in b.legal_moves:
            priority = 0

            if b.is_capture(m):
                target = b.piece_at(m.to_square)
                mover = b.piece_at(m.from_square)
                if target and mover:
                    tv = PIECE_VALUES.get(target.piece_type, 0)
                    mv = PIECE_VALUES.get(mover.piece_type, 0)
                    recaps = [s for s in b.attackers(not b.turn, m.to_square) if b.piece_at(s)]
                    net = tv if not recaps else tv - mv
                    if net >= 1.5:
                        priority = int(net * 10)
                        threats.add(
                            f"free {PIECE_NAMES[target.piece_type]} capture "
                            f"on {chess.square_name(m.to_square)}"
                        )

            tmp = b.copy()
            tmp.push(m)
            if tmp.is_check():
                priority = max(priority, 50)

            if priority > 0:
                candidates.append((priority, m))

        candidates.sort(key=lambda x: x[0], reverse=True)
        for _, m in candidates[:5]:
            da = detect_double_attack(b, m)
            if da:
                threats.add(da)
            disc = detect_discovered_attack(b, m)
            if disc:
                threats.add(disc)
            dc = detect_double_check(b, m)
            if dc:
                threats.add(dc)

        return threats

    pre_board = board.copy()
    pre_board.push(chess.Move.null())
    pre_threats = opponent_threats(pre_board)

    post_board = board.copy()
    post_board.push(move)
    post_threats = opponent_threats(post_board)

    neutralized = pre_threats - post_threats
    return [f"defends against opponent's {t}" for t in sorted(neutralized)]


def detect_check_defense(board, move):
    """Detect check-related defensive purpose of a move.

    Case 1: we are currently in check → move resolves it → "saves king from check".
    Case 2: not in check, but opponent had a checking move available before this
            move and no longer does after → "prevents potential check from opponent".
    """
    if board.is_check():
        return "saves king from check"

    def opponent_can_check(b):
        for m in b.legal_moves:
            tmp = b.copy()
            tmp.push(m)
            if tmp.is_check():
                return True
        return False

    pre_board = board.copy()
    pre_board.push(chess.Move.null())
    if not opponent_can_check(pre_board):
        return None

    post_board = board.copy()
    post_board.push(move)
    if not opponent_can_check(post_board):
        return "prevents potential check from opponent"

    return None


def describe_move(board, move):
    """Describe what a move does: captures, checks, castling, etc."""
    details = []

    # Capture
    if board.is_capture(move):
        if board.is_en_passant(move):
            details.append("captures pawn en passant")
        else:
            captured = board.piece_at(move.to_square)
            if captured:
                details.append(f"captures {PIECE_NAMES[captured.piece_type]}")

    # Promotion
    if move.promotion:
        details.append(f"promotes to {PIECE_NAMES[move.promotion]}")

    # Castling
    if board.is_castling(move):
        side = "kingside" if chess.square_file(move.to_square) == 6 else "queenside"
        details.append(f"castles {side}")

    # Check / checkmate after the move
    temp = board.copy()
    temp.push(move)
    if temp.is_checkmate():
        details.append("delivers checkmate")
    elif temp.is_check():
        double_check = detect_double_check(board, move)
        if double_check:
            details.append(double_check)
        else:
            details.append("gives check")

    # Saving an attacked piece (only report if an attacker is worth less than the piece)
    moving_piece = board.piece_at(move.from_square)
    if moving_piece and board.is_attacked_by(not board.turn, move.from_square):
        piece_value = PIECE_VALUES.get(moving_piece.piece_type, 0)
        attackers = board.attackers(not board.turn, move.from_square)
        min_attacker_value = min(
            (PIECE_VALUES.get(board.piece_at(sq).piece_type, 0)
             for sq in attackers if board.piece_at(sq)),
            default=0,
        )
        if min_attacker_value < piece_value:
            details.append(f"saves attacked {PIECE_NAMES[moving_piece.piece_type]}")

    # Fork / double attack detection
    fork = detect_double_attack(board, move)
    if fork:
        details.append(fork)

    # Discovered attack detection
    discovered = detect_discovered_attack(board, move)
    if discovered:
        details.append(discovered)

    # Pin detection
    pin = detect_pin(board, move)
    if pin:
        details.append(pin)

    # Piece development
    development = detect_development(board, move)
    if development:
        details.append(development)

    # Created threats (new material wins or tactical setups on the next turn)
    details.extend(detect_created_threats(board, move))

    # Opponent threats neutralized by this move
    details.extend(detect_defended_threats(board, move))

    # Check defense (responding to actual check, or preventing a potential one)
    check_defense = detect_check_defense(board, move)
    if check_defense:
        details.append(check_defense)

    return details


def detect_development(board, move):
    """Detect if a minor piece or pawn is being developed from its starting rank."""
    moving_piece = board.piece_at(move.from_square)
    if not moving_piece:
        return None
    start_rank = 0 if moving_piece.color == chess.WHITE else 7
    if (moving_piece.piece_type in (chess.KNIGHT, chess.BISHOP)
            and chess.square_rank(move.from_square) == start_rank):
        return f"develops {PIECE_NAMES[moving_piece.piece_type]}"
    pawn_start_rank = 1 if moving_piece.color == chess.WHITE else 6
    if (moving_piece.piece_type == chess.PAWN
            and chess.square_rank(move.from_square) == pawn_start_rank
            and chess.square_file(move.from_square) == chess.square_file(move.to_square)):
        return "develops pawn"
    return None


def detect_trap(board, pv, score):
    """Detect if the PV involves a material trap.

    Walks the full PV (dynamic window) and flags when the side-to-move gains
    2+ pawns of material at any point in the line. Guards against positions
    that are already clearly winning or are mating attacks.
    """
    if len(pv) < 2:
        return None

    # Guard: skip mate sequences — material is irrelevant there
    if score:
        white_score = score.white()
        if white_score.is_mate():
            return None
        cp = white_score.score()
        if cp is not None:
            # Convert to side-to-move's perspective
            trapper_cp = cp if board.turn == chess.WHITE else -cp
            # Already clearly winning — material gain is expected, not a trap
            if trapper_cp >= 150:
                return None

    trapper_color = board.turn

    def material_balance(b):
        total = 0
        for sq in chess.SQUARES:
            p = b.piece_at(sq)
            if p:
                v = PIECE_VALUES.get(p.piece_type, 0)
                total += v if p.color == trapper_color else -v
        return total

    initial = material_balance(board)
    temp = board.copy()
    max_gain = 0.0
    trap_ply = None  # First ply where gain crosses 2.0

    for i, move in enumerate(pv):
        temp.push(move)
        gain = material_balance(temp) - initial
        if gain > max_gain:
            max_gain = gain
        if gain >= 2.0 and trap_ply is None:
            trap_ply = i

    if max_gain < 2.0:
        return None

    # Reconstruct board just before the trap fires to identify pieces
    temp2 = board.copy()
    for move in pv[:trap_ply]:
        temp2.push(move)

    winning_move = pv[trap_ply]
    captured = temp2.piece_at(winning_move.to_square)
    captured_name = PIECE_NAMES.get(captured.piece_type, "piece") if captured else "piece"
    full_move_num = (trap_ply // 2) + 1

    msg = f"trap springs on move {full_move_num}: wins {captured_name} (+{max_gain:.1f})"

    # The bait: the opponent's move just before the trap fires
    if trap_ply >= 1:
        temp3 = board.copy()
        for move in pv[:trap_ply - 1]:
            temp3.push(move)
        try:
            bait_san = temp3.san(pv[trap_ply - 1])
            msg += f", opponent's {bait_san} walks into it"
        except Exception:
            pass

    return msg


def analyze_position(fen, depth, stockfish_path):
    """Run Stockfish analysis and return structured results as a dict.

    Returns a dict with keys: fen, board, bestMove, bestMoveUci, evaluation,
    details, trap, pvSan, mateInfo, turnToMove, error.
    Raises ValueError for invalid FEN, RuntimeError if Stockfish fails.
    """
    board = chess.Board(fen)

    if board.is_game_over():
        return {
            "fen": fen,
            "board": board,
            "bestMove": None,
            "bestMoveUci": None,
            "evaluation": "Game over",
            "details": [],
            "trap": None,
            "pvSan": [],
            "mateInfo": None,
            "turnToMove": "White" if board.turn else "Black",
        }

    try:
        engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
    except Exception as e:
        raise RuntimeError(f"Failed to start Stockfish: {e}")

    try:
        result = engine.analyse(board, chess.engine.Limit(depth=depth))
    finally:
        engine.quit()

    pv = result.get("pv", [])
    score = result.get("score")

    if not pv:
        return {
            "fen": fen,
            "board": board,
            "bestMove": None,
            "bestMoveUci": None,
            "evaluation": "No legal moves",
            "details": [],
            "trap": None,
            "pvSan": [],
            "mateInfo": None,
            "turnToMove": "White" if board.turn else "Black",
        }

    best_move = pv[0]
    mate_info = None
    if score:
        white_score = score.white()
        if white_score.is_mate():
            mate_in = white_score.mate()
            abs_mate = abs(mate_in)
            side = "White" if mate_in > 0 else "Black"
            sequence = pv_to_san(board, pv[:abs_mate * 2])
            mate_info = {
                "side": side,
                "movesUntil": abs_mate,
                "sequence": " ".join(sequence),
            }

    return {
        "fen": fen,
        "board": board,
        "bestMove": board.san(best_move),
        "bestMoveUci": best_move.uci(),
        "evaluation": format_eval(score),
        "details": describe_move(board, best_move),
       # "trap": detect_trap(board, pv, score),
        "trap": None,  # Disable trap detection for now to avoid false positives
        "pvSan": pv_to_san(board, pv),
        "mateInfo": mate_info,
        "turnToMove": "White" if board.turn else "Black",
    }


def analyze(fen, depth, stockfish_path):
    """CLI wrapper: analyze a position and print results."""
    try:
        data = analyze_position(fen, depth, stockfish_path)
    except ValueError as e:
        print(f"Invalid FEN: {e}")
        sys.exit(1)
    except RuntimeError as e:
        print(e)
        sys.exit(1)

    if not data["bestMove"]:
        print(data["evaluation"])
        return

    print(f"Position: {data['fen']}")
    print(f"Best move: {data['bestMove']}")
    print(f"Evaluation: {data['evaluation']}")

    if data["details"]:
        print(f"Details: {', '.join(data['details'])}")

    if data["trap"]:
        print()
        print(f"*** TRAP DETECTED: {data['trap']} ***")

    if data["mateInfo"]:
        m = data["mateInfo"]
        abs_mate = m["movesUntil"]
        if abs_mate <= 3:
            print()
            print(f"*** FORCED CHECKMATE in {abs_mate} move{'s' if abs_mate != 1 else ''}! ***")
            print(f"{m['side']} delivers checkmate.")
            print(f"Sequence: {m['sequence']}")
        else:
            print()
            print(f"Checkmate in {abs_mate} moves ({m['side']} wins).")
            if m["sequence"]:
                print(f"Sequence: {m['sequence']}")


def main():
    parser = argparse.ArgumentParser(description="Analyze a chess position for best move and forced mates")
    parser.add_argument("--fen", default=chess.STARTING_FEN, help="FEN string (default: starting position)")
    parser.add_argument("--depth", type=int, default=18, help="Search depth (default: 10)")
    parser.add_argument("--stockfish", default=None, help="Path to Stockfish binary")
    args = parser.parse_args()

    stockfish_path = args.stockfish or get_stockfish_path()
    analyze(args.fen, args.depth, stockfish_path)


if __name__ == "__main__":
    main()
