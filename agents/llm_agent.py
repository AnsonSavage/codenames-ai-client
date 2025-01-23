from base.assoc import Assoc
from base.constants import Team
from base.spymaster import BaseSpymaster

import ollama
from pydantic import BaseModel, Field

# Define the schema using Pydantic
class CodenamesLLMHint(BaseModel):
    # train_of_thought: str = Field(..., description="The train of thought that led to the hint.")
    hint: str = Field(..., description="A single word serving as the hint.")
    num_words: int = Field(..., description="The number of words the hint is intended for.")
    intended_words: list[str] = Field(..., description="The list of words the hint is targeting.")
    explanation: str = Field(..., description="Explanation of how the hint relates to each intended word.")



# class LLMAssoc(Assoc):
#     def __init__(self, model, debug=False):
#         super().__init__()
#         self.model = model
#         self.debug = debug

#     def getAssocs(self, pos, neg, topn) -> list[tuple[str, float]]:
#         pass

#     def preprocess(self, w):
#         return w


class LLMSypmaster(BaseSpymaster):
    def __init__(self, assoc=None, debug=False, max_words=3):
        super().__init__(None) # assoc is required by the parent class, but the LLM doesn't use it.
        self.debug = debug
        self.max_words = max_words

    def _is_valid_output(self, hint_data: dict) -> tuple[bool, str]:
        """Validate the model's output against the required rules and provide reasons if invalid."""
        try:
            # Check for required keys
            required_keys = {"hint", "num_words", "intended_words", "explanation"}
            if not required_keys.issubset(hint_data.keys()):
                missing = required_keys - hint_data.keys()
                return False, f"Missing keys: {', '.join(missing)}"

            # Validate 'hint' is a single English word
            hint = hint_data["hint"]
            if not isinstance(hint, str):
                return False, "'hint' is not a string."
            if ' ' in hint:
                return False, "'hint' contains spaces."
            if not hint.isalpha():
                return False, "'hint' contains non-alphabetic characters."
            if hint.lower() in self.board_words:
                return False, "'hint' is a word on the board."
            
            if len(hint > 3):
                # Also check to see if any word on the board contains the hint
                if any(hint in word for word in self.board_words):
                    return False, "'hint' is a substring of a word on the board."
                # Also check to see if the hint contains any word on the board
                if any(word in hint for word in self.board_words):
                    return False, "'hint' contains a word on the board."

            # Validate 'num_words' is a positive integer
            num_words = hint_data["num_words"]
            if not isinstance(num_words, int):
                return False, "'num_words' is not an integer."
            if num_words <= 0:
                return False, "'num_words' is not positive."

            # Validate 'intended_words' is a list of team words
            intended_words = hint_data["intended_words"]
            if not isinstance(intended_words, list):
                return False, "'intended_words' is not a list."
            if not all(isinstance(word, str) for word in intended_words):
                return False, "Not all items in 'intended_words' are strings."
            if len(intended_words) != num_words:
                return False, f"Length of 'intended_words' ({len(intended_words)}) does not match 'num_words' ({num_words})."
            if num_words > self.max_words:
                return False, f"'num_words' exceeds the maximum allowed ({self.max_words})."
            if not set(intended_words).issubset(set(self.board_words)):
                invalid_words = set(intended_words) - set(self.board_words)
                return False, f"'intended_words' contains invalid words: {', '.join(invalid_words)}"

            # Additional validations can be added here if necessary

            return True, "Valid output."
        except Exception as e:
            return False, f"Exception during validation: {str(e)}"

    def makeClue(self, board, team: Team) -> tuple[tuple[str, int], tuple[str]]:
        # Step 1: Extract all words from the game board into a set
        # This will be used later to ensure our clue isn't one of the board words
        self.board_words = set(
            [item for sublist in list(board.values()) for item in sublist]
        )

        # Step 2: Identify our team's words and the opponent's words based on team color
        # 'U' represents blue team, 'R' represents red team
        my_words = board["U" if team == Team.BLUE else "R"]
        opponent_words = board["R" if team == Team.BLUE else "U"]

        # Step 3: Create negative word list (words we want to avoid)
        # Combines opponent's words with neutral ('N') and assassin ('A') words
        # These are words our clue should NOT be similar to
        assassin_word = board["A"][0]
        neutral_words = board["N"]

        max_attempts = 5
        attempt = 0
        while attempt < max_attempts:
            # Craft the prompt
            prompt = f"""
            You are playing the game Codenames as the spymaster. Your team's words are: {', '.join(my_words)}.
            The opposing team's words and neutral words are: {', '.join(opponent_words + neutral_words)}.
            The assassin word is: {assassin_word} (you should avoid giving any hints that might lead the guesser to guess the assassin at all costs).
            Provide a hint that relates to a maximum of {self.max_words} of your team's words.
            Respond in the following JSON format:
            {{
            "hint": "A single word serving as the hint.",
            "num_words": Number of words the hint is intended for,
            "intended_words": ["A sublist of our team's words that the hint is targeting"],
            "explanation": "Explanation of how the hint relates to each intended word. Think step by step. Ensure that the hint wouldn't lead the user to guess any of the words of the other team, especially if the other team has a word that is in a similar category."
            }}

            Remember that the hint word must be:
                - A single English word
                - Not a derivative of a word on currently on the board
                - Not a proper noun
                - Not an acronym
            
            Your guesser is an American college student. If deciding how many words to get the guesser to guess, you should lean towards fewer intended words with more clear connections to the hint.
            """

            # Generate a response using the specified model and schema
            response = ollama.chat(
                messages=[
                    {'role': 'user', 'content': prompt}
                ],
                model='llama3:latest',
                format=CodenamesLLMHint.model_json_schema(),
            )

            # Parse and validate the response
            try:
                hint_data = CodenamesLLMHint.model_validate_json(response.message.content)
            except Exception:
                if self.debug:
                    print("Invalid JSON format received from model.")
                attempt += 1
                continue

            is_valid, reason = self._is_valid_output(hint_data.dict())
            if is_valid:
                # Utilize the valid response
                # print(f"Train of Thought: {hint_data.train_of_thought}")
                print(f"Hint: {hint_data.hint}")
                print(f"Number of Words: {hint_data.num_words}")
                print(f"Intended Words: {', '.join(hint_data.intended_words)}")
                print(f"Explanation: {hint_data.explanation}")
                return (hint_data.hint, hint_data.num_words), tuple(hint_data.intended_words)
            else:
                if self.debug:
                    print(f"Invalid hint data received: {reason}. Retrying...")
                attempt += 1

        # Fallback in case of repeated invalid outputs
        print("Failed to generate a valid hint after multiple attempts.")
        return ("None", 1), ()