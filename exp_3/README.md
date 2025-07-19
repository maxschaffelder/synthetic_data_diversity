# Experiment 3: Impact of Synthetic Data Diversity on Self-Preference Bias

This directory contains the code and notebooks for Experiment 3, which investigates how the diversity of synthetic fine-tuning data affects self-preference bias in LLMs.

## Research Question

**RQ3**: Does a model fine-tuned on its own output exhibit higher self-preference bias than one trained on a synthetic dataset from diverse sources?

## Directory Structure

-   `data_generation/`: Scripts to generate summaries from the various models using articles from the CNN/DailyMail dataset.
-   `pairwise_rankings/`: Contains scripts and notebooks for the pairwise ranking of the generated summaries by the fine-tuned models.
-   `absolute_ratings/`: Includes scripts and notebooks for obtaining absolute quality ratings of the summaries from the fine-tuned models on a 5-point Likert scale.

## Usage

1.  Use the scripts in `data_generation/` to generate the summaries.
2.  Run the scripts in `pairwise_rankings/` and `absolute_ratings/` to have the fine-tuned models evaluate the summaries.
3.  Use the notebooks in each subdirectory to analyze the results and measure self-preference bias. 