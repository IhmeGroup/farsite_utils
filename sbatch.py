"""Class for managing SLURM batch configurations"""

import os
import datetime as dt
from string import Template


class _DeltaTemplate(Template):
    delimiter = "%"


def strfdelta(tdelta, fmt):
    d = {"D": tdelta.days}
    d["H"], rem = divmod(tdelta.seconds, 3600)
    d["M"], d["S"] = divmod(rem, 60)
    t = _DeltaTemplate(fmt)
    return t.substitute(**d)


class _Option:
    def __init__(self, flag="", value=None):
        self.flag = flag
        self.value = value
    

    def __str__(self):
        return str(self.__class__) + ": " + str(self.__dict__)
    

    def __formatted_value(self):
        if isinstance(self.value, dt.timedelta):
            return strfdelta(self.value, "%D-%H:%M:%S")
        else:
            return str(self.value)
    

    def set_value_from_str(self, string, value_type=str):
        if value_type == dt.timedelta:
            if "-" in string:
                split = string.split("-")
                days = split[0]
                time = split[1]
            else:
                days = "0"
                time = string
            
            [hours, minutes, seconds] = time.split(":")
            
            self.value = dt.timedelta(
                days=int(days),
                hours=int(hours),
                minutes=int(minutes),
                seconds=int(seconds))
        elif value_type == int:
            self.value = int(string)
        elif value_type == str:
            self.value = string


    def to_str(self):
        return "#SBATCH {0} {1}".format(self.flag, self.__formatted_value())


class SBatch:
    def __init__(self, filename=None):
        self.shell_path = "#!/bin/bash"
        self.options = [
            _Option("-J", "job"),
            _Option("-o", "%x.%j.out"),
            _Option("-N", 1),
            _Option("-n", 1),
            _Option("-t", dt.timedelta(hours=2))]
        self.echoline = ""
        self.exec = []
        self.runfile_name_local = ""

        if filename:
            self.read(filename)
    

    def __str__(self):
        return str(self.__class__) + ": " + str(self.__dict__)
    

    def set_option(self, flag, value):
        for opt in self.options:
            if opt.flag == flag:
                opt.value = value
                return
        self.options.append(_Option(flag, value))
    

    def read(self, filename):
        self.options = []
        with open(filename, "r") as file:
            for line in file:
                split = line.strip().split(" ")
                if not line.strip():
                    continue
                elif line[0:2] == "#!":
                    self.shell_path = line.strip()
                elif split[0] == "#SBATCH":
                    if split[1] in ("-N, -n"):
                        value_type = int
                    elif split[1] == "-t":
                        value_type = dt.timedelta
                    else:
                        value_type = str
                    opt = _Option(flag=split[1])
                    opt.set_value_from_str(split[2], value_type)
                    self.options.append(opt)
                elif split[0] == "echo":
                    self.echoline = " ".join(split[1:])
                elif "TestFARSITE" in split[0]:
                    self.exec = split[0]
                    runfile_name = split[1]
                    runfile_name_split = runfile_name.split(os.path.sep)
                    if runfile_name_split[0] == ".":
                        self.runfile_name_local = os.path.join(*runfile_name_split[1:])
                    else:
                        self.runfile_name_local = runfile_name


    def write(self, filename):
        with open(filename, "w") as file:
            file.write(self.shell_path + "\n")
            file.write("\n")
            for opt in self.options:
                file.write(opt.to_str() + "\n")
            file.write("\n")
            file.write("echo " + self.echoline + "\n")
            file.write("\n")
            file.write(self.exec + " " + self.runfile_name_local)


def main():
    pass


if __name__ == "__main__":
    main()
