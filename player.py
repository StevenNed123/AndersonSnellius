import chess
import torch
from typing import Optional
from transformers import AutoTokenizer, AutoModelForCausalLM
from chess_tournament.players import Player


class TransformerPlayer(Player):

    def __init__(self, name: str = "AndersonSnellius"):
        super().__init__(name)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model_path = "SteveNed123/AndersonSnellius-Chess"

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.PIECE_VALUES = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 1000}


    def calculate_material(self, board):
        score = 0
        for piece in board.piece_map().values():
            value = self.PIECE_VALUES[piece.piece_type]
            if piece.color == board.turn:
                score += value
            else:
                score -= value
        return score


    def is_endgame(self, board):
        pieces = board.piece_map().values()
        non_pawn_non_king = [p for p in pieces if p.piece_type not in (chess.PAWN, chess.KING)]
        return len(non_pawn_non_king) <= 2


    def get_move(self, fen: str) -> Optional[str]:
      board = chess.Board(fen)
      moves = [m.uci() for m in self.get_priority_moves(board)]
      if not moves:
          return None

      best_move = None
      best_score = -float("inf")

      for move in moves:
          move_score = self.score_move(fen, move)
          if move_score > best_score:
              best_score = move_score
              best_move = move

      return best_move


    def score_move(self, fen, move):
      prompt = f"FEN: {fen}\nMove:"
      text = prompt + " " + move
      inputs = self.tokenizer(text, return_tensors="pt").to(self.device)

      with torch.no_grad():
          outputs = self.model(**inputs, labels=inputs["input_ids"])
          loss = outputs.loss.item()

      return -loss


    def leaves_piece_hanging(self, board, move):
        board.push(move)
        square = move.to_square
        piece = board.piece_at(square)
        opponent = board.turn
        us = not board.turn

        attackers = board.attackers(opponent, square)
        defenders = board.attackers(us, square)

        if piece is None:
          board.pop()
          return False

        piece_value = self.PIECE_VALUES[piece.piece_type]

        attacker_value = 0
        for sq in attackers:
            p = board.piece_at(sq)
            attacker_value += self.PIECE_VALUES[p.piece_type]

        defender_value = piece_value
        for sq in defenders:
            p = board.piece_at(sq)
            defender_value += self.PIECE_VALUES[p.piece_type]

        board.pop()

        return attacker_value > defender_value

    def is_checkmate(self, board):
        for move in board.legal_moves:
            board.push(move)

            if board.is_checkmate():
                board.pop()
                return True
            board.pop()
        return False

    def is_stalemate(self, board, move):
      board.push(move)
      result = board.is_stalemate()
      board.pop()
      return result

    def is_threefold(self, board, move):
        board.push(move)
        result = board.is_repetition(3)
        board.pop()
        return result

    def get_priority_moves(self, board):

        legal_moves = list(board.legal_moves)
        safe_moves = []
        good_captures = []
        pawn_pushes = []
        checks = []
        promotions = []

        for move in legal_moves:
            if move == self.is_checkmate(board):
              return [move]
            moving_piece = board.piece_at(move.from_square)
            if move.promotion:
                promotions.append(move)
            if moving_piece.piece_type == chess.PAWN:
                pawn_pushes.append(move)
            if board.is_capture(move):
                captured_piece = board.piece_at(move.to_square)
                if captured_piece is None:
                    continue
                if self.PIECE_VALUES[captured_piece.piece_type] > self.PIECE_VALUES[moving_piece.piece_type] or board.attackers(not board.turn, move.to_square) == 0:
                    good_captures.append(move)
            elif not self.leaves_piece_hanging(board, move) and not self.is_stalemate(board, move) and not self.is_threefold(board, move):
                if board.gives_check(move):
                    checks.append(move)
                else:
                    safe_moves.append(move)

        if len(promotions):
          return promotions
        elif len(good_captures):
          return good_captures
        elif self.is_endgame(board) and len(pawn_pushes):
          return pawn_pushes
        elif len(checks):
          return checks
        elif len(safe_moves):
          return safe_moves
        else:
          return legal_moves