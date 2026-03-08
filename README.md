# Anderson Snellius the chess bot

Anderson Snellius is a chess bot based on the GPT2 decoder model. It has been fine-tuned on 500.000 positions from the lichess open database to improve its chess capabilities. 

The bot assumes the input format is in FEN notation and predicts its move in UCI notation.

## Tuning the Model

Snellius has been tuned during multiple training session (because of google colab GPU rate limit). The Tuning setup is fairly standard. The optimizer used is AadamW with a learning rate of 5*10^-5 and a batch size of 8. Epochs were originally set to 5 but it could barely finish 1 before google colab's timeout. Total training time is estimated to be about 10-12 hours.

During the training, Snellius is tasked with predicting the token(move) that completes the sequence(FEN position). The loss is calculated using cross entropy and gradiants are updated via backpropagation of error.

Data Source: https://database.lichess.org/. 

The Data has been:

- All Filtered on 1800+ elo
- 400.000 positions filtered on Moves 5-40
- 100.000 positions filtered on Moves 0-50
- Moves are repeated once to boost output signal

## Simple Heuristics
Because Snesllius has a bit of trouble at times, it uses a heuristic to stack moves into different priorites. It first checks if a move from the top layer is available then it goes to the second, then the third and so on. About 95% of positions uses the Safe Moves stack (which is usually filled with moves) so almost all decisions are still made by Snelllius. If there are for example, multiple Checks available Snellius would still pick between one of the Checks.

- Mate in one
- Promotions
- Good Captures (Captures that win material)
- Checks
- Safe Moves (Moves that do not blunder)
- The best legal Move

Training data is a bit sparse especially near the endgame(Defined as: less then 3 non-pawn, non-king pieces), because of this Snellius struggles a bit with closing out its games and needs some guidance. (If I had more time to train I would remove this heuristic and tune it extra on end-game/checkmate data.)

- Prioritze pawn moves; above Checks but bellow Good Captures

After the correct priority stack has been selected, Snellius is prompted to avaluate a move given the position and returns a Log-Likehood. After avaluating all moves at that stack it will output the one it thinks is the most likely.

## Running the bot

The bot uses the instructer's package. In Google Colab the install is:

```
git clone https://github.com/bylinina/chess_exam.git
cd chess-exam
pip install -e .
```
An instance of Snellius can be created by running:
```
AndersonSnellius = TransformerPlayer()
```
A game between the RandomPlayer from the instructor package and Snellius:
```
random1 = RandomPlayer("Random-1")
AndersonSnellius = TransformerPlayer()
game = Game(AndersonSnellius, random1, max_half_moves=300)
game.play(log_moves=True, log_to_file='log.csv')
```
