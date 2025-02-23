# Detecting and Mitigating Toxic Comments in Online Platforms

## Overview
Reddit’s moderation system relies heavily on human intervention, which can be inefficient and inconsistent. This project proposes an **NLP-based framework** leveraging **Latent Dirichlet Allocation (LDA) for topic modeling** and **Toxic-BERT for toxicity detection** to analyze user engagement and toxicity in Reddit discussions.

## Team Members
- Marina Mitiaeva, mmitiaev@syr.edu
- Mervin McDougall, msmcdoug@syr.edu
- Lakshmi Peram, lperam@syr.edu
- Nick DeVita, njdevita@syr.edu

## Project Pipeline
1. **Data Collection**: Extracted and sampled Reddit dataset from Hugging Face.
2. **Exploratory Data Analysis (EDA)**: Visualized word distributions, toxicity patterns, and engagement metrics.
3. **Topic Modeling**: Implemented LDA to identify dominant themes in discussions.
4. **Toxicity Prediction**: Used a **pre-trained Toxic-BERT model** to score post titles.
5. **Analysis & Insights**: Examined relationships between toxicity, engagement, and topic diversity.

## Data
- **Source**: [Hugging Face - Reddit Questions with Best Answers](https://huggingface.co/datasets/nreimers/reddit_question_best_answers)
- **Size**: 50,000 sampled posts (original dataset: ~1.8M posts from 2010-2021)
- **Features**: Post titles, body text, scores, answer count, toxicity scores

## Key Findings
- The most **viral topics** (politics, technology, finance) had **low toxicity levels**.
- **Negligible correlation** between title length, toxicity, and engagement.
- **High engagement does not imply high toxicity**, challenging common moderation assumptions.

## Installation
To set up the environment, run:
```bash
git clone https://github.com/your_username/IST718-BigData-Project.git
cd IST718-BigData-Project
pip install -r requirements.txt
```

## Usage
1. **Run the Jupyter Notebook**:
   ```bash
   jupyter notebook notebooks/FINAL_BDA_project.ipynb
   ```
2. **Use standalone scripts**:
   - `src/preprocess.py`: Data cleaning and transformation
   - `src/model.py`: Topic modeling & toxicity prediction
   - `src/visualization.py`: Word clouds, topic distributions

## Dependencies
- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `pyspark`
- `transformers`
- `torch`
- `nltk`

## Results
Our analysis revealed several key insights:

- Viral Topics: The most engaging topics on Reddit were centered around politics, technology, and finance. Keywords such as "government," "vote," "data," and "tax" dominated discussions. Despite their high engagement, these topics exhibited surprisingly low toxicity levels, suggesting that high engagement does not necessarily lead to harmful discourse.

- Title Toxicity and Engagement: The correlation between title toxicity and engagement was negligible (-0.073), indicating that toxic content does not significantly drive user interaction. Similarly, the correlation between title length and engagement was weak (0.069), showing that longer titles do not necessarily lead to higher engagement.

- Toxicity Distribution: The average toxicity score was 0.0206, with most posts exhibiting minimal toxic content. Only a few discussions contained explicit language flagged as toxic, further indicating that Reddit's Q&A threads, at least in this dataset, do not predominantly contain harmful content.

- Topic Diversity: Our LDA analysis showed that nearly 50% of posts were dominated by only two major themes out of the 20 detected, indicating a concentration of discussions around specific areas of interest.

## Limitations & Future Work
- **Memory constraints**: Only a 50k sample used due to computational limits.
- **Model alignment**: Toxic-BERT may not be optimized for Reddit-specific discussions.
- **Future Improvements**:
  - Fine-tune a domain-specific toxicity model.
  - Explore **temporal trends** in toxicity levels over time.
  - Increase dataset size using more scalable infrastructure.