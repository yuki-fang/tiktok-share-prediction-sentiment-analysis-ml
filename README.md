# TikTok Share Prediction: Sentiment Analysis & Machine Learning

This project analyzes how emotional expressions in TikTok video comments affect virality, measured by share count. Using regression and decision tree models with a train/test split, I found that emotionally charged content—especially disapproval sentiment—significantly increases shares, while admiration sentiment shows a negative association. These insights suggest sentiment can drive active sharing and inform content design and recommendation strategies.

Research Question: How do emotional expressions in video comments affect virality, measured by share count, on TikTok?

---

## Project Motivation
Understanding how comment sentiment influences TikTok video virality helps creators and marketers design more engaging content and predict which videos are likely to be widely shared.

---

## Variables & Controls

### Dependent Variable
| Variable | Description |
|----------|-------------|
| `log_Share.Count` | Log-transformed number of shares per video (engagement metric) |

### Independent Variables: Comment-Based Features
| Variable | Description |
|----------|-------------|
| `log_mean_comment_disapproval` | Log-transformed mean disapproval sentiment in comments |
| `log_mean_comment_approval` | Log-transformed mean approval sentiment in comments |
| `log_mean_comment_disgust` | Log-transformed mean disgust sentiment in comments |
| `log_mean_comment_admiration` | Log-transformed mean admiration sentiment in comments |
| `log_mean_comment_anger` | Log-transformed mean anger sentiment in comments |

### Control Variables
| Variable | Rationale |
|----------|-----------|
| `Age` | Viewer engagement may differ by perceived age due to age-related biases or relatability |
| `Gender.deepface` | Gender may influence audience engagement and algorithmic visibility |
| `Race` | Race may affect viewer perception |
| `log_Follower.Count` | Larger followings typically lead to higher engagement due to greater reach |
| `log_Likes.sum.Count` | Reflects long-term popularity or account reputation |
| `log_Following.Count` | May reflect reciprocal engagement norms |
| `Verified.Status` | Verified accounts may receive algorithmic advantages and more user trust |
| `Image_quality` | Higher video quality may lead to better viewer experience |
| `Brightness` | Brighter videos affect mood and visibility |
| `Sharpness` | High visual clarity relates to better production value and viewer retention |


---

## Analysis Approach
1. Exploratory research, descriptive statistics, preliminary analysis
2. Multiple linear regression and regularized regression (Ridge, Lasso)  
3. Stepwise variable selection with cross-validation  
4. Decision tree ensemble and random forest models  
5. Train/test split for model evaluation  
6. Interpretation of sentiment effects on share count  

---

## Key Findings
- Disapproval sentiment strongly **boosts shares**  
- Admiration sentiment shows a **negative association** with shares  
- Emotionally charged comments drive active sharing, offering actionable insights for content strategy  

---

## Key Methods & Tools
| Category | Details |
|----------|---------|
| Methods  | Multiple Linear Regression, Ridge & Lasso Regression, Stepwise Variable Selection, Decision Tree, Random Forest |
| Tools    | RStudio (packages: *tidyverse*, *glmnet*, *rpart*, *randomForest*) |

---

## How to Run
1. Clone the repository:  
```bash
git clone https://github.com/YOUR_USERNAME/tiktok-share-prediction-sentiment-analysis-ml.git
