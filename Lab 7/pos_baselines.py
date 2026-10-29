"""Reusable Hindi POS baseline models from Assignment 7."""

from collections import Counter
from math import log
from pathlib import Path

import numpy as np
import pandas as pd

BOS = "<BOS>"
EOS = "<EOS>"


def load_tagged_sentences(path):
    """Load UTF-8 token<TAB>tag data with blank-line sentence boundaries."""
    path = Path(path)
    sentences = []
    sentence = []
    with path.open(encoding="utf-8") as data_file:
        for line_number, line in enumerate(data_file, start=1):
            line = line.rstrip("\r\n")
            if not line.strip():
                if sentence:
                    sentences.append(sentence)
                    sentence = []
                continue
            fields = line.rsplit("\t", 1)
            if len(fields) != 2 or not all(fields):
                raise ValueError(f"Invalid token/tag row in {path.name}:{line_number}")
            word, tag = fields
            sentence.append((word, tag))
    if sentence:
        sentences.append(sentence)
    return sentences


def _naive_bayes_features(words, index, previous_tag):
    word = words[index]
    previous_word = words[index - 1] if index else BOS
    next_word = words[index + 1] if index + 1 < len(words) else EOS
    length_feature = "MORE" if len(word) > 4 else "LESS"
    return (length_feature, word, previous_word, next_word, previous_tag)


def train_naive_bayes(sentences, label_set, alpha=1.0):
    tag_counts = Counter()
    feature_counts = {tag: [Counter() for _ in range(5)] for tag in label_set}
    feature_values = [set() for _ in range(5)]
    total_tokens = 0

    for sentence in sentences:
        words = [word for word, _ in sentence]
        previous_tag = BOS
        for index, (_, tag) in enumerate(sentence):
            values = _naive_bayes_features(words, index, previous_tag)
            tag_counts[tag] += 1
            total_tokens += 1
            for feature_index, value in enumerate(values):
                feature_counts[tag][feature_index][value] += 1
                feature_values[feature_index].add(value)
            previous_tag = tag

    return {
        "tags": list(label_set),
        "tag_counts": tag_counts,
        "feature_counts": feature_counts,
        "feature_sizes": [len(values) + 1 for values in feature_values],
        "total_tokens": total_tokens,
        "alpha": alpha,
    }


def predict_naive_bayes(words, model):
    predictions = []
    for index in range(len(words)):
        previous_tag = predictions[-1] if predictions else BOS
        values = _naive_bayes_features(words, index, previous_tag)
        best_tag = None
        best_score = float("-inf")
        for tag in model["tags"]:
            tag_count = model["tag_counts"][tag]
            alpha = model["alpha"]
            score = log(
                (tag_count + alpha)
                / (model["total_tokens"] + alpha * len(model["tags"]))
            )
            for feature_index, value in enumerate(values):
                denominator = tag_count + alpha * model["feature_sizes"][feature_index]
                count = model["feature_counts"][tag][feature_index][value]
                score += log((count + alpha) / denominator)
            if score > best_score:
                best_tag = tag
                best_score = score
        predictions.append(best_tag)
    return predictions


def train_hmm(sentences, label_set, alpha=0.1):
    tag_counts = Counter()
    emission_counts = {tag: Counter() for tag in label_set}
    transition_counts = Counter()
    start_counts = Counter()
    end_counts = Counter()

    for sentence in sentences:
        previous_tag = BOS
        for word, tag in sentence:
            tag_counts[tag] += 1
            emission_counts[tag][word] += 1
            if previous_tag == BOS:
                start_counts[tag] += 1
            else:
                transition_counts[(previous_tag, tag)] += 1
            previous_tag = tag
        end_counts[previous_tag] += 1

    tag_index = {tag: index for index, tag in enumerate(label_set)}
    tag_total = len(label_set)
    sentence_count = len(sentences)
    vocabulary_size = len(
        {word for counts in emission_counts.values() for word in counts}
    )
    transitions = np.empty((tag_total, tag_total))
    for previous_tag in label_set:
        denominator = tag_counts[previous_tag] + alpha * (tag_total + 1)
        for next_tag in label_set:
            transitions[tag_index[previous_tag], tag_index[next_tag]] = log(
                (transition_counts[(previous_tag, next_tag)] + alpha) / denominator
            )

    start = np.array(
        [
            log((start_counts[tag] + alpha) / (sentence_count + alpha * tag_total))
            for tag in label_set
        ]
    )
    end = np.array(
        [
            log(
                (end_counts[tag] + alpha)
                / (tag_counts[tag] + alpha * (tag_total + 1))
            )
            for tag in label_set
        ]
    )
    return {
        "tags": list(label_set),
        "tag_to_index": tag_index,
        "tag_counts": tag_counts,
        "emission_counts": emission_counts,
        "vocabulary_size": vocabulary_size,
        "transitions": transitions,
        "start": start,
        "end": end,
        "alpha": alpha,
    }


def _emission_log_probabilities(words, model):
    rows = []
    for word in words:
        row = []
        for tag in model["tags"]:
            denominator = model["tag_counts"][tag] + model["alpha"] * (
                model["vocabulary_size"] + 1
            )
            count = model["emission_counts"][tag][word]
            row.append(log((count + model["alpha"]) / denominator))
        rows.append(row)
    return np.asarray(rows)


def forward_viterbi(words, model):
    if not words:
        return []
    emissions = _emission_log_probabilities(words, model)
    scores = model["start"] + emissions[0]
    backpointers = []
    for position in range(1, len(words)):
        candidates = scores[:, None] + model["transitions"]
        previous_tags = np.argmax(candidates, axis=0)
        scores = candidates[previous_tags, np.arange(len(model["tags"]))] + emissions[
            position
        ]
        backpointers.append(previous_tags)
    last_tag = int(np.argmax(scores + model["end"]))
    path = [last_tag]
    for previous_tags in reversed(backpointers):
        path.append(int(previous_tags[path[-1]]))
    path.reverse()
    return [model["tags"][index] for index in path]


def evaluate_by_tag(sentences, predictions, label_set):
    if len(sentences) != len(predictions):
        raise ValueError("Prediction sentence count does not match gold data")

    true_counts = Counter()
    predicted_counts = Counter()
    correct_counts = Counter()
    total_correct = 0
    total_tokens = 0

    for sentence, predicted_tags in zip(sentences, predictions):
        if len(sentence) != len(predicted_tags):
            raise ValueError("Prediction length does not match sentence length")
        for (_, gold_tag), predicted_tag in zip(sentence, predicted_tags):
            true_counts[gold_tag] += 1
            predicted_counts[predicted_tag] += 1
            if gold_tag == predicted_tag:
                correct_counts[gold_tag] += 1
                total_correct += 1
            total_tokens += 1

    rows = []
    for tag in label_set:
        precision = (
            correct_counts[tag] / predicted_counts[tag] if predicted_counts[tag] else 0.0
        )
        recall = correct_counts[tag] / true_counts[tag] if true_counts[tag] else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        rows.append(
            {
                "tag": tag,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": true_counts[tag],
            }
        )
    accuracy = total_correct / total_tokens if total_tokens else 0.0
    return pd.DataFrame(rows), accuracy
