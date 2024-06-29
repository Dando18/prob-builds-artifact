#!/usr/bin/env spack-python

""" Generate all Spack specs to build for experiments. Yields a list of all
    versions for each package.
"""
# std imports
from argparse import ArgumentParser

# tpl imports
import spack
import spack.repo
import yaml


def get_args():
    parser = ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-p", "--package", type=str, nargs="+", help="Package to generate specs for.")
    group.add_argument("--package-group", type=str, choices=["e4s", "ecp-proxy-apps"], help="Package group to generate specs for.")
    parser.add_argument("--no-enumerate-root-versions", action="store_true", help="Don't enumerate all versions for the root package. Default: False")
    parser.add_argument("--enumerate-dependency-versions", action="store_true", help="Enumerate all versions for each dependency. Default: False")
    parser.add_argument("-f", "--output-format", type=str, default="specs", choices=["specs", "yaml"], help="Format to output specs in. Default: specs")
    parser.add_argument("-o", "--output", type=str, help="Path to output file. Default: stdout.")
    return parser.parse_args()


PACKAGE_GROUPS = {
    "ecp-proxy-apps": ["amg", "cosmoflow-benchmark", "examinimd", "miniamr", "miniqmc", "mlperf-deepcam", "sw4lite", 
                    "xsbench", "candle-benchmarks", "cradl", "laghos", "minife", "minitri", "nekbone", "swfft", "comd", 
                    "ember", "macsio", "minigan", "minivite", "picsarlite", "thornado-mini"],
    "e4s": ["abyss","adios","amrex","ascent","axom","bolt","cabana","caliper","chai","conduit","darshan-runtime",
            "hypre","hpx","heffte","hdf5","gmp","globalarrays","fortrilinos","faodel","kokkos","kokkos-kernels",
            "libnrm","mercury","metall","mfem","mpark-variant","ninja","omega-h","openmpi","openpmd-api","papyrus",
            "parallel-netcdf","petsc","phist","plasma","pumi","py-libensemble","py-petsc4py","qt","qthreads","qwt",
            "raja","rempi","scr","slate","slepc","stc","sundials","superlu-dist","swig","sz","tasmanian","trilinos",
            "turbine","umap","umpire","unifyfs","upcxx","variorum","veloc","zfp"],
}

def main():
    args = get_args()

    # expand package list
    packages = args.package if args.package else PACKAGE_GROUPS[args.package_group]

    # get the spack package
    specs = spack.cmd.parse_specs(packages)
    if len(specs) == 0:
        raise ValueError(f"spack.cmd.parse_specs({packages}) returned {len(specs)} specs. Expected > 0.")

    # get all versions for the root package
    all_versions = { k.name: [] for k in specs }
    if not args.no_enumerate_root_versions:
        for spec in specs:
            all_versions[spec.name] = spack.repo.path.get_pkg_class(spec.name).versions

    # get all versions from the dependencies
    if args.enumerate_dependency_versions:
        raise NotImplementedError("Enumerating dependency versions is not yet implemented.")

    # generate specs
    final_specs = []
    if not args.no_enumerate_root_versions:
        for spec_name, versions in all_versions.items():
            for version in versions:
                final_specs.append(f"{spec_name}@{version}")
    else:
        final_specs.extend(spec.name for spec in specs)

    # output specs
    spack_yml = {"spack": {"specs": final_specs, "concretizer": {"unify": False}}}
    if args.output_format == "specs":
        if args.output is None:
            print("\n".join(final_specs))
        else:
            with open(args.output, "w") as f:
                f.write("\n".join(final_specs))
    elif args.output_format == "yaml":
        if args.output is None:
            yaml.dump(spack_yml, sys.stdout)
        else:
            with open(args.output, "w") as f:
                yaml.dump(spack_yml, f)


if __name__ == "__main__":
    main()