"""Class for managing WindNinja configurations."""

import re

class _Option:
    def __init__(self, name="", value=None):
        self.name = name
        self.value = value
    

    def __str__(self):
        return str(self.__class__) + ": " + str(self.__dict__)
    

    def __formatted_value(self):
        return str(self.value)


    def to_str(self):
        return "{0} = {1}".format(self.name, self.__formatted_value())


class WindNinjaConfig:
    def __init__(self, filename=None):
        self.options = []

        if filename:
            self.read(filename)
    

    def __str__(self):
        return str(self.__class__) + ": " + str(self.__dict__)
    

    def set_option(self, name, value):
        for opt in self.options:
            if opt.name == name:
                opt.value = value
                return
        self.options.append(_Option(name, value))
    

    def __stripComment(self, line):
        return line.split("#")[0]
    

    def read(self, filename):
        self.options = []
        with open(filename, "r") as file:
            for line in file:
                split = re.split(" += +", self.__stripComment(line).strip())
                if not line.strip():
                    continue
                elif line[0] == "#":
                    continue
                else:
                    opt = _Option(name=split[0], value=split[1])
                    self.options.append(opt)


    def write(self, filename):
        with open(filename, "w") as file:
            for opt in self.options:
                file.write(opt.to_str() + "\n")


def main():
    pass


if __name__ == "__main__":
    main()

