
from colorama import Fore

# diffusers.utils.eval_utils

def print_color(s, *args, c='r'):
    if c == 'r':
        # print(Fore.RED + s + Fore.RESET)
        print(Fore.RED, end='')
        print(s, *args, Fore.RESET)
    elif c == 'b':
        # print(Fore.BLUE + s + Fore.RESET)
        print(Fore.BLUE, end='')
        print(s, *args, Fore.RESET)
    elif c == 'y':
        # print(Fore.YELLOW + s + Fore.RESET)
        print(Fore.YELLOW, end='')
        print(s, *args, Fore.RESET)
    else:
        # print(Fore.CYAN + s + Fore.RESET)
        print(Fore.CYAN, end='')
        print(s, *args, Fore.RESET)