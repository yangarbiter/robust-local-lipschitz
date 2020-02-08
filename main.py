
from lolip.variables import auto_var
from utils import setup_experiments


def main():
    setup_experiments(auto_var)
    auto_var.parse_argparse()

if __name__ == '__main__':
    main()
