from dotenv import load_dotenv
load_dotenv()

import sys, os
import warnings
warnings.filterwarnings("ignore")

ANSI_RESET="\033[0m"        # Reset all formatting
ANSI_BOLD="\033[1m"         # Bold text
ANSI_YELLOW="\033[33m"      # Yellow text

sys.stdout.write(ANSI_BOLD + ANSI_YELLOW)
print("Finetuning and Inferencing LLM:")
print("\n")

import pyfiglet
LLM_art = pyfiglet.figlet_format("VIT",  font="slant", justify="center")
# print(LLM_art)

sys.stdout.write(ANSI_RESET)

from src.utils.logger import logger
from src import App


def main():
    app = App()
    app.run()


if __name__ == "__main__":
    main()