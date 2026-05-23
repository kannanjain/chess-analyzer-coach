#!/usr/bin/env python3
"""Web UI for chess position analyzer."""

import os
import json as json_module
import chess
import chess.engine
import boto3
from dotenv import load_dotenv
from flask import Flask, render_template, request, jsonify

# Ensure working directory is the project root so relative Stockfish path resolves.
os.chdir(os.path.dirname(os.path.abspath(__file__)))
load_dotenv()

from position_analyzer import get_stockfish_path, analyze_position

app = Flask(__name__)

STOCKFISH_PATH = get_stockfish_path()
BEDROCK_CLIENT = boto3.client(
    "bedrock-runtime",
    region_name="us-west-2",
)
HINT_MODEL = "anthropic.claude-3-5-haiku-20241022-v1:0"


@app.route("/")
def index():
    """Serve the main page."""
    return render_template("index.html")


@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    """Analyze a chess position and return best move with explanation."""
    data = request.get_json(force=True)
    fen = data.get("fen", chess.STARTING_FEN)
    depth = data.get("depth", 18)
    depth = max(1, min(depth, 24))

    try:
        result = analyze_position(fen, depth, STOCKFISH_PATH)
    except ValueError as e:
        return jsonify({"error": f"Invalid FEN: {e}"}), 400
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 500

    return jsonify({
        "fen": result["fen"],
        "bestMove": result["bestMove"],
        "bestMoveUci": result["bestMoveUci"],
        "evaluation": result["evaluation"],
        "details": result["details"],
        "turnToMove": result["turnToMove"],
        "pvSan": result["pvSan"][:8],
        "mateInfo": result["mateInfo"],
    })


@app.route("/api/coach/move", methods=["POST"])
def api_coach_move():
    """Make the coach's move at a given rating level."""
    data = request.get_json(force=True)
    fen = data.get("fen", chess.STARTING_FEN)
    rating = int(data.get("rating", 1200))
    rating = max(500, min(rating, 2000))

    try:
        board = chess.Board(fen)
    except ValueError as e:
        return jsonify({"error": f"Invalid FEN: {e}"}), 400

    if board.is_game_over():
        return jsonify({"gameOver": True, "result": board.result(), "fen": fen})

    # UCI_Elo maps directly to ELO; minimum supported by Stockfish is 1320.
    elo = max(1320, min(3190, rating))

    try:
        engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
    except Exception as e:
        return jsonify({"error": f"Failed to start Stockfish: {e}"}), 500

    try:
        engine.configure({"UCI_LimitStrength": True, "UCI_Elo": elo})
        result = engine.play(board, chess.engine.Limit(time=0.1))
    finally:
        engine.quit()

    move = result.move
    move_san = board.san(move)
    board.push(move)

    return jsonify({
        "move": move_san,
        "moveUci": move.uci(),
        "fen": board.fen(),
        "gameOver": board.is_game_over(),
        "result": board.result() if board.is_game_over() else None,
    })


@app.route("/api/hint", methods=["POST"])
def api_hint():
    """Return a coach hint — vague nudge or direct move explanation."""
    data = request.get_json(force=True)
    fen = data.get("fen", chess.STARTING_FEN)
    direct = data.get("direct", False)

    try:
        analysis = analyze_position(fen, 15, STOCKFISH_PATH)
    except (ValueError, RuntimeError) as e:
        return jsonify({"error": str(e)}), 400

    best_move = analysis["bestMove"]
    details = analysis["details"]
    trap = analysis.get("trap")
    mate_info = analysis.get("mateInfo")

    if not best_move:
        return jsonify({"hint": "The game is over!", "bestMove": None})

    # Collect all detected facts from the analyzer.
    facts = list(details)
    if trap:
        facts.append(trap)
    if mate_info:
        facts.append(
            f"checkmate in {mate_info['movesUntil']} move(s) for {mate_info['side']}"
        )

    facts_block = "\n".join(f"- {f}" for f in facts) if facts else "- good move for position"

    print(f"\n[HINT] best={best_move} direct={direct}\n{facts_block}\n")

    if direct:
        prompt = (
            f"You are a chess coach. Write 2-3 grammatically correct, clear, and precise sentences explaining the best move to a student.\n"
            f"The best move is {best_move}.\n\n"
            f"FACTS (these are the only things this move does — do not add anything else):\n"
            f"{facts_block}\n\n"
            f"Strict rules:\n"
            f"- Describe ONLY what is listed in the facts above. Do not infer, expand, or add chess ideas.\n"
            f"- Do not add adjectives or qualifiers (like 'undefended', 'hanging', 'free') unless that exact word appears in the facts.\n"
            f"- Do not use chess terms (fork, pin, skewer, double attack, tempo, pressure, initiative, etc.) "
            f"unless that exact word appears in the facts above.\n"
            f"- Do not say the move 'creates pressure', 'improves position', or any vague strategic claim "
            f"unless a fact explicitly states it.\n"
            f"- Do not mention engines, computers, or analysis.\n"
            f"- Do not start with a greeting (no 'Hey', 'Hi', 'Great', etc.). Get straight to the point.\n"
            f"- Do not use informal or quirky language (no 'tasty', 'juicy', 'nice', 'love this', etc.).\n"
            f"- Each sentence must be complete, grammatically correct, and unambiguous.\n"
            f"- If the facts list is short, write fewer sentences — do not pad with invented ideas."
        )
    else:
        prompt = (
            f"You are a chess coach giving a student a hint. Do NOT name the move or any square.\n\n"
            f"FACTS about the best move (hint using ONLY these — do not add anything else):\n"
            f"{facts_block}\n\n"
            f"Strict rules:\n"
            f"- Base your hint ONLY on the facts above. Do not infer or add chess ideas.\n"
            f"- Do not add adjectives or qualifiers (like 'undefended', 'hanging', 'free') unless that exact word appears in the facts.\n"
            f"- Do not use chess terms (fork, pin, double attack, tempo, etc.) unless that exact word "
            f"appears in the facts above.\n"
            f"- Do not name any square or the specific move.\n"
            f"- Do not mention engines, computers, or analysis.\n"
            f"- Do not start with a greeting (no 'Hey', 'Hi', 'Great', etc.). Get straight to the point.\n"
            f"- Do not use informal or quirky language (no 'tasty', 'juicy', 'nice', 'love this', etc.).\n"
            f"- If a fact says 'develops [piece]', simply say 'develop a piece' — do not describe the piece or its movement pattern.\n"
            f"- Each sentence must be complete, grammatically correct, and unambiguous.\n"
            f"- Write 1-2 sentences nudging the student toward the idea. Nothing more."
        )

    body = json_module.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 1500,
        "temperature": 0.5,
        "messages": [{"role": "user", "content": prompt}]
    })

    try:
        response = BEDROCK_CLIENT.invoke_model(modelId=HINT_MODEL, body=body)
        result = json_module.loads(response["body"].read())
        hint_text = result["content"][0]["text"]
    except Exception as e:
        return jsonify({"error": f"LLM error: {e}"}), 500

    return jsonify({
        "hint": hint_text,
        "bestMove": best_move if direct else None,
    })


if __name__ == "__main__":
    app.run(debug=True, port=5001)
