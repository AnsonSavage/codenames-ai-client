from base.assoc import Assoc
from base.constants import Team
from base.spymaster import BaseSpymaster

import ollama
from pydantic import BaseModel, Field
import abc
from openai import OpenAI

# Define the schema using Pydantic
class CodenamesLLMHint(BaseModel):
    plan: str = Field(..., description="Your plan for how you will get your team to guess the words and which words you think you might try targetting. This field is a sentence.")
    hint: str = Field(..., description="A single word serving as the hint.")
    num_words: int = Field(..., description="The number of words the hint is intended for.")
    intended_words: list[str] = Field(..., description="The list of words the hint is targeting.")
    explanation: str = Field(..., description="Explanation of how the hint relates to each intended word.")

class LLMSpymaster(BaseSpymaster, metaclass=abc.ABCMeta):
    def __init__(self, debug=False, max_words=3):
        super().__init__(None)  # Parent requires assoc, but LLM doesn't use it
        self.debug = debug
        self.max_words = max_words
        self.board_words = set()
        self.team_words = []

    @abc.abstractmethod
    def get_llm_response(self, prompt: str) -> str:
        """Abstract method to be implemented by subclasses with specific LLM calls."""
        pass

    def _is_valid_output(self, hint_data: dict) -> tuple[bool, str]:
        """Validate the model's output against the required rules and provide reasons if invalid."""
        if self.debug:
            print(f"Validating hint data: {hint_data}")
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
            
            if len(hint) > 3:
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
        self.team_words = my_words  # used in the validator

        if self.debug:
            print(f"My words: {my_words}")
            print(f"Opponent words: {opponent_words}")
            print(f"Neutral words: {neutral_words}")
            print(f"Assassin word: {assassin_word}")

        max_attempts = 5
        attempt = 0
        while attempt < max_attempts:
            prompt = f"""
            You are playing the game Codenames as the spymaster. Your team's words are: {', '.join(my_words)}.
            The opposing team's words and neutral words are: {', '.join(opponent_words + neutral_words)}.
            The assassin word is: {assassin_word} (you should avoid giving any hints that might lead the guesser to guess the assassin at all costs).
            Provide a hint that relates to a maximum of {self.max_words} of your team's words.
            Respond in the following JSON format:
            {{
            "plan": "Your plan for how you will get your team to guess the words and which words you think you might try targetting. This field is a paragraph. Talk about how the word you choose will be similar to a subset of your team's words, and different from the neutral, opponent, and especially the assassin word.",
            "hint": "A single word serving as the hint.",
            "num_words": Number of words the hint is intended for,
            "intended_words": ["A sublist of our team's words that the hint is targeting"],
            "explanation": "Explanation of how the hint relates to each intended word. Think step by step. Also explain how the word is different from each of the negative words." 
            }}

            Remember that the hint word must be:
                - A single English word
                - Not a derivative of a word on currently on the board
                - Not a proper noun
                - Not an acronym
            
            Your guesser is an American college student. If deciding how many words to get the guesser to guess, you should lean towards fewer intended words with more clear connections to the hint.
            """
            response_content = self.get_llm_response(prompt)

            if self.debug:
                print(f"Response from model: {response_content}")
            try:
                hint_data = CodenamesLLMHint.model_validate_json(response_content)
                if self.debug:
                    print("Hint data validated successfully.")
            except Exception:
                if self.debug:
                    print("Invalid JSON format received from model.")
                attempt += 1
                continue

            is_valid, reason = self._is_valid_output(hint_data.dict())
            if is_valid:
                if self.debug:
                    print(f"Plan: {hint_data.plan}")
                    print(f"Hint: {hint_data.hint}")
                    print(f"Number of Words: {hint_data.num_words}")
                    print(f"Intended Words: {', '.join(hint_data.intended_words)}")
                    print(f"Explanation: {hint_data.explanation}")
                return (hint_data.hint, hint_data.num_words), tuple(hint_data.intended_words)
            else:
                if self.debug:
                    print(f"Invalid hint data: {reason}. Retrying...")
                attempt += 1

        print("Failed to generate a valid hint after multiple attempts.")
        return ("None", 1), ()

class OllamaSpymaster(LLMSpymaster):
    """Subclass that calls Ollama to get a response."""
    def __init__(self, debug=False, max_words=3):
        super().__init__(debug=debug, max_words=max_words)

    def get_llm_response(self, prompt: str) -> str:
        response = ollama.chat(
            messages=[{'role': 'user', 'content': prompt}],
            model='llama3:latest',
            format=CodenamesLLMHint.model_json_schema(),
        )
        return response.message.content

class OpenAISpymaster(LLMSpymaster):
    """Subclass that calls OpenAI to get a response."""
    def __init__(self, openai_api_key: str, model: str = 'gpt-4o', debug=False, max_words=3):
        super().__init__(debug=debug, max_words=max_words)
        self.openai_api_key = openai_api_key
        self.model = model
        self.client = OpenAI()  # Initialize the OpenAI client

    def get_llm_response(self, prompt: str) -> CodenamesLLMHint:
        # Use the OpenAI client to parse structured output
        completion = self.client.beta.chat.completions.parse(
            model=self.model,
            messages=[
                {"role": "system", "content": "Provide a hint for the game Codenames as per the schema."},
                {"role": "user", "content": prompt},
            ],
            response_format=CodenamesLLMHint,
        )

        # Extract the parsed output
        print(completion.choices[0].message.content)
        # import sys
        # sys.exit(0)
        return completion.choices[0].message.content