from .devtools import dev_cli


def main(argv=None, *, prog_name="pyrite-dev"):
    return dev_cli.main(argv, prog_name=prog_name)


if __name__ == "__main__":
    main()
