""" Generate Clingo/Plingo atoms for a set of version-pair build probabilities.
"""
# std imports
from argparse import ArgumentParser
import sys

# tpl imports
from alive_progress import alive_it
import pandas as pd
#from scipy.special import softmax

""" Format of atoms for clingo method. """
CLINGO_FMT_STR = 'will_fail("{parent}", "{parent_version}", "{child}", "{child_version}", {fail_weight}).\n'

""" Format of atoms for plingo+conflict method. """
PLINGO_CONFLICT_FMT_STR = """condition({idx0}, "conflict trigger {parent}@{parent_version}").
condition_requirement({idx0}, "node", "{parent}").
condition_requirement({idx0}, "node_version_satisfies", "{parent}", "{parent_version}").
condition({idx1}, "conflict constraint {child}@{child_version}").
condition_requirement({idx1}, "node", "{child}").
condition_requirement({idx1}, "node_version_satisfies", "{child}", "{child_version}").
conflict("{parent}", {idx0}, {idx1}, "{parent}@{parent_version} conflicts with {child}@{child_version}") :- &weight("{fail_weight}").
version_satisfies("{parent}", "{parent_version}", "{parent_version}").
version_satisfies("{child}", "{child_version}", "{child_version}").{line_end}"""

""" Format of atoms for plingo+error and plingo+constraint method. """
PLINGO_ERROR_FMT_STR = 'will_fail("{parent}", "{parent_version}", "{child}", "{child_version}") :- &weight("{fail_weight}").\n'


def get_args():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("-d", "--data", type=str, required=True, help="Path to csv data set containing version-pair build expected improvement.")
    parser.add_argument("-o", "--output", type=str, help="Path to output file. Default: stdout.")
    parser.add_argument("-n", "--normalize", nargs="?", type=str, const="per-pair", 
        choices=["per-pair", "all", "per-parent", "per-child", "per-parent-version", "per-child-version"], 
        help="Normalize expected improvement with softmax function by this group. Default: per-pair")
    parser.add_argument("--normalization-method", type=str, default="maxabs", choices=["softmax", "maxabs", "zscore"], help="Method to use for normalizing expected improvement. Default: maxabs")
    parser.add_argument("-m", "--method", type=str, default="clingo", choices=["clingo", "plingo+conflict", "plingo+error", "plingo+constraint"], help="Method to use for generating atoms. Default: clingo")
    parser.add_argument("-e", "--extrapolator", type=str, help="Path to sklearn extrapolator. If provided, use this extrapolator to label missing data. Default: None")
    parser.add_argument("--allow-partial-pairs", action="store_true", help="Only use the set of versions for each package in the data set. " + 
                        "Don't check Spack for an exhaustive set of versions. Default: False")
    parser.add_argument("-r", "--round", type=int, const=2, nargs="?", help="Round the expected improvement to this many decimal places. Default: 2")
    parser.add_argument("--idx-offset", type=int, default=20_000, help="What idx to start generating conflict atoms at for plingo+conflict method. Default: 20_000")
    parser.add_argument("--no-pretty-print", action="store_true", help="Don't include comments and line spacing.")
    return parser.parse_args()

""" Helper groupings for --normalize parameter """
NORMALIZATION_GROUPS = {
    "per-pair": ["parent", "child"],
    "all": ["parent", "child", "parent version", "child version"],
    "per-parent": ["parent"],
    "per-child": ["child"],
    "per-parent-version": ["parent", "parent version"],
    "per-child-version": ["child", "child version"]
}



def write_atoms(
    data: pd.DataFrame, 
    fmt_str: str, 
    writer, 
    idx_offset: int = 20_000, 
    pretty_print: bool = True, 
    progress: bool = True
):
    """ Write atoms to writer using fmt_str.

    Args:
        data: DataFrame containing version-pair build expected improvement.
        fmt_str: Format string for writing atoms.
        writer: File-like object to write atoms to.
        idx_offset: What idx to start generating conflict atoms at for plingo+conflict method.
        pretty_print: Whether to include comments and line spacing.
        progress: Whether to show progress bar.
    """
    
    # print out the atoms grouped by (parent, child)
    groups = data.groupby(["parent", "child"])
    groups = alive_it(groups, title="Writing Parent/Child Groups") if progress else groups
    global_idx = 0
    for (parent, child), group in groups:
        if pretty_print:
            writer.write(f"% {parent} - {child} version pair build weights.\n")
        for _, row in group.iterrows():
            writer.write(fmt_str.format(
                parent=parent, 
                parent_version=row["parent version"], 
                child=child, 
                child_version=row["child version"], 
                fail_weight=row["expected improvement"],
                idx0=idx_offset + global_idx,
                idx1=idx_offset + global_idx + 1,
                line_end="\n" if pretty_print else ""
            ))
            global_idx += 2
        
        if pretty_print:
            writer.write("\n")


def main():
    args = get_args()

    # raise errors for currently incomplete features
    if args.extrapolator is not None:
        # currently values are already extrapolated in the data set, no support for doing it here tho
        raise NotImplementedError("Extrapolation is not yet implemented. Omit --extrapolator flag.")

    if not args.allow_partial_pairs:
        raise NotImplementedError("Checking spack for exhaustive set of versions is not yet implemented. Add --allow-partial-pairs flag.")
    
    # if True, then generate probabilities of failure; otherwise, generate probabilities of success
    GEN_FAIL_PROBS = args.method in ["clingo", "plingo+conflict", "plingo+error", "plingo+constraint"]

    # read in data set
    data = pd.read_csv(args.data)

    # data assertions
    REQUIRED_COLUMNS = ["parent", "parent version", "child", "child version", "expected improvement"]
    assert all([c in data.columns for c in REQUIRED_COLUMNS]), f"Data set must contain columns {REQUIRED_COLUMNS}."
    assert data["expected improvement"].min() >= 0, "Expected improvement must be non-negative."
    assert len(data) > 0, "Data set must contain at least one row."

    # temporary for now: only use the root version pairs. todo -- handle this case better
    data = data[data.root == data.parent]

    # normalize expected improvement
    if args.normalize is not None:
        if args.normalization_method == "softmax":
            norm_func = softmax
        elif args.normalization_method == "maxabs":
            # data is already in [0, inf) so no need to take absolute value
            norm_func = lambda x: x / x.max()
        elif args.normalization_method == "zscore":
            norm_func = lambda x: (x - x.mean()) / x.std()
        else:
            raise ValueError("Normalization method {} is not supported.".format(args.normalization_method))

        norm_group = NORMALIZATION_GROUPS[args.normalize]
        data["expected improvement"] = data.groupby(norm_group)["expected improvement"].transform(norm_func)
        
        if GEN_FAIL_PROBS:
            # invert probabilities; currently they are probability of success, but we want probability of failure
            data["expected improvement"] = 1.0 - data["expected improvement"]
    elif not GEN_FAIL_PROBS:
        # no normalization and we're generating success values, so invert weights
        # the difference here is that expected improvement is in [0, inf) so we have to invert by subtracting from the max
        data["expected improvement"] = data.groupby(["parent", "child"])["expected improvement"].transform(lambda x: x.max() - x)

    # round expected improvement
    if args.round is not None:
        data["expected improvement"] = data["expected improvement"].round(args.round)

    # needs ordering; the clingo based methods dont work with floats, so we need to replace it with an integer rank
    if args.method in ["clingo"]:
        # most likely to build will be 1, least likely will be n
        # this is desirable since we'll minimize this weight in clingo
        data["expected improvement"] = data.groupby(["parent", "child"])["expected improvement"].rank(method="dense", ascending=True).astype(int)

    # get the format string
    if args.method == "clingo":
        fmt_str = CLINGO_FMT_STR
    elif args.method == "plingo+conflict":
        fmt_str = PLINGO_CONFLICT_FMT_STR
    elif args.method == "plingo+error":
        fmt_str = PLINGO_ERROR_FMT_STR
    elif args.method == "plingo+constraint":
        fmt_str = PLINGO_ERROR_FMT_STR
    else:
        raise NotImplementedError("Method {} is not yet implemented.".format(args.method))
    
    # generate output
    with open(args.output, "w") if args.output else nullcontext(sys.stdout) as f:
        write_atoms(data, fmt_str, f, progress=args.output is not None, idx_offset=args.idx_offset, 
            pretty_print=not args.no_pretty_print)


if __name__ == "__main__":
    main()