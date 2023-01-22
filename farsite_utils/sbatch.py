"""Class for managing SLURM batch configurations."""

import os
import datetime as dt
import re


def strdelta(tdelta):
    days = tdelta.days
    hours, rem = divmod(tdelta.seconds, 3600)
    minutes, seconds = divmod(rem, 60)
    return "{0:d}-{1:02d}:{2:02d}:{3:02d}".format(days, hours, minutes, seconds)


class _Option:
    def __init__(self, flag="", value=None):
        self.flag = flag
        self.value = value
    

    def __str__(self):
        return str(self.__class__) + ": " + str(self.__dict__)
    

    def __formatted_value(self):
        if isinstance(self.value, dt.timedelta):
            return strdelta(self.value)
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
            _Option("-t", dt.timedelta(hours=2)),
            _Option("--partition", "pdebug")]
        self.echoline = ""
        self.exec = []
        self.exec_windninja = []
        self.runfile_name_local = ""
        self.windninja_config_file_local = ""
        self.setup_lines = []

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
                split = re.split(" |=", line.strip())
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
                elif ("TestFARSITE" in split[0]) or ("WindNinja_cli" in split[0]):
                    self.exec = split[0]
                    runfile_name = split[1]
                    runfile_name_split = runfile_name.split(os.path.sep)
                    if runfile_name_split[0] == ".":
                        self.runfile_name_local = os.path.join(*runfile_name_split[1:])
                    else:
                        self.runfile_name_local = runfile_name
                elif len(line) > 0 and not line[0] == "#":
                    self.setup_lines.append(line.strip())


    def write(self, filename):
        with open(filename, "w") as file:
            file.write(self.shell_path + "\n")
            file.write("\n")
            for opt in self.options:
                file.write(opt.to_str() + "\n")
            file.write("\n")
            file.write("echo " + self.echoline + "\n")
            file.write("\n")
            for line in self.setup_lines:
                file.write(line + "\n")
            file.write(self.exec + " " + self.runfile_name_local)


def main():
    pass


if __name__ == "__main__":
    main()
