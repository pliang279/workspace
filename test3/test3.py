#!/usr/bin/env python
import argparse
import json
import os
from typing import Any, Iterable, Mapping, Tuple

import numpy as np
import sklearn
from sklearn import svm
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from config import CONFIG_BY_KEY, Config
from data_loader import DataHelper, DataLoader

RESULT_FILE = "output/{}.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-key", default="", choices=list(CONFIG_BY_KEY))
    return parser.parse_args()


def svm_train(config: Config, train_input: np.ndarray, train_output: np.ndarray) -> sklearn.base.BaseEstimator:
    clf = make_pipeline(
        StandardScaler() if config.svm_scale else FunctionTransformer(lambda x: x, validate=False),
        svm.SVC(C=config.svm_c, gamma="scale", kernel="rbf")
    )
    return clf.fit(train_input, np.argmax(train_output, axis=1))


def svm_test(clf: sklearn.base.BaseEstimator, test_input: np.ndarray,
             test_output: np.ndarray) -> Tuple[Mapping[str, Any], str]:
    probas = clf.predict(test_input)  # noqa
    y_pred = probas
    y_true = np.argmax(test_output, axis=1)

    result_string = classification_report(y_true, y_pred, digits=3)
    # print(confusion_matrix(y_true, y_pred))
    # print(result_string)
    return classification_report(y_true, y_pred, output_dict=True, digits=3), result_string


def train_io(config: Config, data: DataLoader, train_index: Iterable[int],
             test_index: Iterable[int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train_input, train_output = data.get_split(train_index)
    test_input, test_output = data.get_split(test_index)

    datahelper = DataHelper(train_input, train_output, test_input, test_output, config, data)

    trainT_input = np.empty((len(train_input), 0))
    testT_input = np.empty((len(test_input), 0))
    trainV_input = np.empty((len(train_input), 0))
    testV_input = np.empty((len(test_input), 0))
    trainA_input = np.empty((len(train_input), 0))
    testA_input = np.empty((len(test_input), 0))

    #use bert
    trainT_input = np.concatenate([trainT_input, datahelper.get_target_bert_feature(mode="train")], axis=1)
    testT_input = np.concatenate([testT_input, datahelper.get_target_bert_feature(mode="test")], axis=1)

    #use video
    trainV_input = np.concatenate([trainV_input, datahelper.get_target_video_pool(mode="train")], axis=1)
    testV_input = np.concatenate([testV_input, datahelper.get_target_video_pool(mode="test")], axis=1)

    #use audio
    trainA_input = np.concatenate([trainA_input, datahelper.get_target_audio_pool(mode="train")], axis=1)
    testA_input = np.concatenate([testA_input, datahelper.get_target_audio_pool(mode="test")], axis=1)

    train_output = datahelper.one_hot_output(mode="train", size=config.num_classes)
    test_output = datahelper.one_hot_output(mode="test", size=config.num_classes)

    return trainT_input, trainV_input, trainA_input, train_output, testT_input, testV_input, testA_input, test_output

def train_speaker_dependent(config: Config, data: DataLoader, model_name: str) -> None:
    resultsA = []
    resultsT = []
    resultsV = []
    for fold, (train_index, test_index) in enumerate(data.get_stratified_k_fold()):
        config.fold = fold + 1
        print("Present Fold:", config.fold)

        trainT_input, trainV_input, trainA_input, train_output, testT_input, testV_input, testA_input, test_output = train_io(config=config, data=data, train_index=train_index,
                                                                      test_index=test_index)

        clf = svm_train(config=config, train_input=trainA_input, train_output=train_output)
        result_dict, result_str = svm_test(clf, testA_input, test_output)
        resultsA.append(result_dict)

        clf = svm_train(config=config, train_input=trainT_input, train_output=train_output)
        result_dict, result_str = svm_test(clf, testT_input, test_output)
        resultsT.append(result_dict)

        clf = svm_train(config=config, train_input=trainV_input, train_output=train_output)
        result_dict, result_str = svm_test(clf, testV_input, test_output)
        resultsV.append(result_dict)

    if not os.path.exists(os.path.dirname(RESULT_FILE)):
        os.makedirs(os.path.dirname(RESULT_FILE))
    
    results = [resultsA, resultsT, resultsV]
    print("results A")
    print(results[0])
    print("results T")
    print(results[1])
    print("results V")
    print(results[2])

    with open(RESULT_FILE.format(model_name), "w") as file:
        json.dump(results, file)


def print_result(model_name: str) -> None:
    with open(RESULT_FILE.format(model_name)) as file:
        results = json.load(file)

    weighted_precision = []
    weighted_recall = []
    weighted_f_scores = []

    print("#" * 20)
    for fold, result in enumerate(results):
        weighted_f_scores.append(result["weighted avg"]["f1-score"])
        weighted_precision.append(result["weighted avg"]["precision"])
        weighted_recall.append(result["weighted avg"]["recall"])

        print(f"Fold {fold + 1}:")
        print(f"Weighted Precision: {result['weighted avg']['precision']}  "
              f"Weighted Recall: {result['weighted avg']['recall']}  "
              f"Weighted F score: {result['weighted avg']['f1-score']}")
    print("#" * 20)
    print("Avg :")
    print(f"Weighted Precision: {np.mean(weighted_precision):.3f}  "
          f"Weighted Recall: {np.mean(weighted_recall):.3f}  "
          f"Weighted F score: {np.mean(weighted_f_scores):.3f}")


def main() -> None:
    args = parse_args()
    print("Args:", args)

    config = CONFIG_BY_KEY[args.config_key]

    data = DataLoader(config)
    for _ in range(config.runs):
        train_speaker_dependent(config=config, data=data, model_name=config.model)
        # print_result(model_name=config.model)

if __name__ == "__main__":
    main()


