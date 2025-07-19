# Experiment 1: The Impact of Synthetic Data Diversity on Model Collapse

This directory contains the code and notebooks for Experiment 1, which investigates the impact of synthetic data diversity on model collapse.

## Research Question

**RQ1**: Is model collapse more severe for models fine-tuned on their own output than models fine-tuned on synthetic data from diverse sources?

## Directory Structure

-   `data_generation/`: Contains scripts used to generate the outputs from the various fine-tuned models. These outputs form the basis for the analysis.
-   `metric_calculation/`: Includes scripts and notebooks for calculating the metrics used to measure model collapse:
    -   **Perplexity**: To measure distributional drift and model degradation.
    -   **Lexical Diversity**: Assessed using Self-BLEU and Heaps' Law.
    -   **Semantic Diversity**: Measured using SentenceBERT embeddings.

## Usage

1.  Use the scripts in `data_generation/` to have the fine-tuned models generate responses to the held-out test set.
2.  Use the notebooks and scripts in `metric_calculation/` to analyze the generated outputs and produce the results for Experiment 1. 