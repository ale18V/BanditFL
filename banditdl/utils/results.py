"""Result-file and plotting helpers."""

import os
import pathlib

import numpy
import pandas

from banditdl.core.analysis import study


def store_result(fd, *entries):
    fd.write(os.linesep + ("\t").join(str(entry) for entry in entries))
    fd.flush()


def make_result_file(fd, *fields):
    fd.write("# " + ("\t").join(str(field) for field in fields))
    fd.flush()


def compute_avg_err_op(name, seeds, result_directory, location, *colops, avgs="", errs="-err"):
    result_directory = pathlib.Path(result_directory)
    datas = tuple(
        study.select(
            study.Session(result_directory / f"{name}-{seed}", location),
            *(col for col, _ in colops),
        )
        for seed in seeds
    )

    def make_df_ro(col, op):
        subds = tuple(study.select(data, col).dropna() for data in datas)
        df = pandas.DataFrame(index=subds[0].index)
        ro = None
        for cn in subds[0]:
            avgn = cn + avgs
            errn = cn + errs
            numds = numpy.stack(tuple(subd[cn].to_numpy() for subd in subds))
            df[avgn] = numds.mean(axis=0)
            df[errn] = numds.std(axis=0)
            if op is not None:
                if ro is not None:
                    raise RuntimeError(
                        f"column selector {col!r} selected more than one column while a reduction operation was requested"
                    )
                ro = tuple(getattr(subd[cn], op)().item() for subd in subds)
        return df, ro

    dfs = []
    for col, op in colops:
        df, _ = make_df_ro(col, op)
        dfs.append(df)
    return dfs
