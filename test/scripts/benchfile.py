import os
import time
import re
import subprocess
from pathlib import Path


def create_test(nprocs: str, dendro_root: str):

    newfullfile = []

    with open(dendro_root + "test_doctests/partition_weighted.cpp", "r") as fval:

        fullfile = list(fval.readlines())
        prefix = ""
        linenumber = 0

        for idx, line in enumerate(fullfile):
            if re.match("MPI_TEST_CASE", line):

                linenumber = idx

                strmatch = re.search(r'\"([\w\s-]+)\"', line)
                strspan = strmatch.span()
                idxval1 = strspan[1]

                commamatch = re.search(",", line[idxval1:])
                commaspan = commamatch.span()
                idxval2 = commaspan[1]

                prefix = line[:(idxval1 + idxval2)]

        suffix = ")\n"
        newlistval = prefix + nprocs + suffix

        newfullfile = fullfile[:linenumber] + \
            [newlistval] + fullfile[linenumber + 1:]

    with open(dendro_root + "test_doctests/partition_weighted.cpp", "w") as fval:

        fval.writelines(newfullfile)


def run_tests(dendro_root):

    compile_cmd = "make -j 4"
    run_cmd_prefix = "mpirun -np "
    build_dir = dendro_root + "build/"
    run_cmd_mid = " --oversubscribe " + build_dir + \
        "test_doctests/doctest_partition_weighted"

    for nprocs in range(32, 513):

        strval = str(nprocs)
        create_test(strval, dendro_root)

        run_cmd = run_cmd_prefix + strval + run_cmd_mid

        outfile = build_dir + "test_doctests/" + "out_" + str(nprocs) + ".txt"

        compile_cmd_list = compile_cmd.split(" ")
        subprocess.run( compile_cmd_list, cwd=( build_dir + "test_doctests" ) )
        time.sleep(1)

        print(run_cmd)

        runlist = run_cmd.split(" ")
        subprocess.run(runlist, stdout=open(outfile, 'w'), stderr=open(outfile, 'w'), cwd=( build_dir + "test_doctests" ) )


if __name__ == "__main__":

    home_dir = Path.home()
    dendro_root = home_dir.as_posix() + "/Documents/Dendro-KT/"
    run_tests(dendro_root)
