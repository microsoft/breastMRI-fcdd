# Copyright (c) 2021 liznerski (original FCDD work)
# Copyright (c) 2025 Microsoft Corporation (FCDD for breast cancer detection)
# Licensed under the MIT License.

from fcdd.datasets.image_folder import extract_custom_classes
from fcdd.runners.bases import ClassesRunner
from fcdd.runners.argparse_configs import DefaultConfig
import fcdd.datasets


class CustomConfig(DefaultConfig):
    def __call__(self, parser):
        parser = super().__call__(parser)
        parser.add_argument(
            "--it",
            type=int,
            default=5,
            help="Number of runs per class with different random seeds.",
        )
        parser.add_argument(
            "--cls-restrictions",
            type=int,
            nargs="+",
            default=None,
            help="Run only training sessions for some of the classes being nominal.",
        )
        # parser.add_argument(
        #     "--ground-truth-maps",
        #     "-gtms",
        #     action="store_true",
        #     help="Activates utilization of binary ground-truth maps for training and/or testing. "
        #     "This requires additional dataset folders `data/custom/train_maps/classX/...` "
        #     "and/or `data/custom/test_maps/classX/...` where the corresponding maps are placed. "
        #     "The ground-truth maps need to be binary; i.e., need to be in {0, 255}^{1 x h x w}, "
        #     "where 255 marks anomalous regions. "
        #     "For more details see :class:`fcdd.datasets.image_folder_gtms.ADImageFolderDatasetGTM`.",
        # )
        parser.add_argument(
            "--gpu",
            type=str,
            default=str(0),
            help="Defines GPU in which to run model, for setups with multiple GPUs.",
        )
        parser.set_defaults(
            ground_truth_maps=False,
            dataset="ref",
            net="FCDD_REF_CNN224_VGG_F",
            supervise_mode="other",
            # 'fcddrefs' assumes that the second sample (reference) is always normal
            objective="fcddrefs",
        )

        return parser


if __name__ == "__main__":
    runner = ClassesRunner(CustomConfig())
    runner.args.logdir += "_ref_"
    fcdd.datasets.CUSTOM_CLASSES = [
        "scans"
    ]  # just for logging purposes, mandatory though
    fcdd.datasets.image_folder_refs.ADImageRefDataset.gtm = False
    del runner.args.ground_truth_maps

    runner.run()
    print()
