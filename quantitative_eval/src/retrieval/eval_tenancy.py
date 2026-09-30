from ranx import compare, Qrels, Run
import argparse
import sys
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('topic_set', choices=['known', 'inferred', 'overall'])
    parser.add_argument('qrel')
    parser.add_argument('runs', nargs='+', default=[])
    args = parser.parse_args()

    alpha = 0.01

    qrels_df = pd.read_csv(
        args.qrel,
        sep='\t',
        names=['q_id', '0', 'doc_id', 'score'],
        header=None
    )

    if args.topic_set == "known":
        qrels_df = qrels_df[
            qrels_df['q_id'].str.startswith('K')
        ]

    elif args.topic_set == "inferred":
        qrels_df = qrels_df[
            qrels_df['q_id'].str.startswith('I')
        ]

    # For "overall", keep all 20 questions.

    runs = [
        Run.from_file(run, kind="trec")
        for run in args.runs
    ]

    qrels = Qrels.from_df(
        qrels_df,
        q_id_col="q_id",
        doc_id_col="doc_id",
        score_col="score"
    )

    report = compare(
        qrels=qrels,
        runs=runs,
        metrics=["ndcg@1", "ndcg@3", "ndcg@5"],
        max_p=alpha,
        make_comparable=True,
        stat_test="tukey",
        rounding_digits=4,
    )

    print("{} Topics".format(args.topic_set))
    print(report)
    print(report.to_latex())


if __name__ == "__main__":
    sys.exit(main())
