# Official Repository of ChronoBerg

We introduce CHRONOBERG, a temporally structured corpus of English book texts spanning 250 years, curated from Project Gutenberg and enriched with a variety of temporal annotations. We also introduce historically calibrated affective Valence-Arousal-Dominance (VAD)  lexicons to support temporally grounded interpretation. With the lexicons at hand, we demonstrate a need for modern LLM-based tools to better situate their detection of discriminatory language and contextualization of sentiment across various time-periods. In fact, we show how language models trained sequentially on CHRONOBERG struggle to encode diachronic shifts in meaning, emphasizing the need for temporally aware training and evaluation pipelines, and positioning CHRONOBERG as a scalable resource for the study of linguistic change and temporal generalization. $${\color{red}Disclaimer: }$$ This repository and dataset includes language and display of samples that could be offensive to readers. 

![ChronoBerg](https://github.com/paulsubarna/Chronoberg/blob/main/figures/chrono_flow.png)

To install the dependencies, run <pre/>pip install -r requirements.txt</pre> 

Catalog:
- [x] ChronoBerg: Dataset and Statistics
- [x] Lexical Analysis
- [x] Hate Speech Detection Models
- [x] Sequential training of LLMs on ChronoBerg

## Dataset
The dataset is available at Huggingface [ChronoBerg](https://huggingface.co/datasets/chb19/ChronoBerg/tree/main)
Dataset Catalog:
- [x] ChronoBerg raw [non-annotated](https://huggingface.co/datasets/chb19/ChronoBerg/tree/main/dataset)
- [x] ChronoBerg: sentence-level valence annotated (for each time interval: 50 year span) [annotated](https://huggingface.co/datasets/sdp56/ChronoBerg/tree/main/dataset)
- [x] [Valence Lexicons]([https://huggingface.co/datasets/sdp56/ChronoBerg/tree/main/lexicons](https://huggingface.co/datasets/sdp56/ChronoBerg/tree/main/Valence_lexicon)) 
- [x] [Dominance Lexicons]([https://huggingface.co/datasets/sdp56/ChronoBerg/tree/main/lexicons](https://huggingface.co/datasets/sdp56/ChronoBerg/tree/main/Dominance_lexicon))
- [x] [Arousal Lexicons]([https://huggingface.co/datasets/sdp56/ChronoBerg/tree/main/lexicons](https://huggingface.co/datasets/sdp56/ChronoBerg/tree/main/Arousal_lexicon))

## Lexical Analysis
We have provided notebooks describing how to work with the provided lexicons. 
- Notebook on analyzing words that have undergone semantic shifts
![Lexical](https://github.com/paulsubarna/Chronoberg/blob/main/figures/lexical_analysis.png)
- Notebook on determining affective connotations for sentences in ChronoBerg

## Create your own Lexicons
One could also create their own lexicons by training their own Word2Vec models to learn their vector embeddings for each word in Chronoberg. 
To work with the lexicons provided with ChronoBerg, we also made available the five pre-trained models on each 50-year time-interval.

**Pretrained Checkpoints** : 
Model-Type | 1750-99 | 1800-49 | 1850-99 | 1900-49 | 1950-99 |
--- | :---: | :---: | :---: |:---: |:---:
word2vec | [word2vec_1750](https://huggingface.co/datasets/chb19/ChronoBerg/tree/main/dataset) | [word2vec_1800](https://huggingface.co/datasets/chb19/ChronoBerg/tree/main/dataset) | [word2vec_1850](https://huggingface.co/datasets/chb19/ChronoBerg/tree/main/dataset) | [word2vec_1900](https://huggingface.co/datasets/chb19/ChronoBerg/tree/main/dataset) | [word2vec_1950](https://huggingface.co/datasets/chb19/ChronoBerg/tree/main/dataset) |


Train your own word2vec models:
<pre/>python train_word2vec_model.py --epochs 10 --window 10 --workers 6 --vector-size 300
</pre>

## Sequential training LLMS

#### 1. Data and Model Preparation

-   1. Extract the provided test set from the full dataset. 

-   2. Place the splits within "data" folder as shown in the structure above. 

-   3. Load and save model using "cl_methods/model/init_model.py"

#### 2. Training and Evaluating

To train, run: 

```
# Run training script with default parameters ("ewc.sh" or "lora.sh"):
sh scripts/ewc/ewc.sh
```

Key parameters in the training script:

-   `--model_name_or_path`: Path to the pretrained model
-   `--data_path`: Path to the training dataset
-   `--dataset_name`: Names of the datasets to train on
-   `--reg`: Regularization parameter (default: 0.5)
-   `--num_train_epochs`: Number of training epochs per task (default: 30)


#### Evaluate
To evaluate on the two test sets (valence_shifting and valence stable) and calculate the perplexities, run: 

-  For EWC and ST: 
```
python eval.py --model_dir ./outputs/$model_dir --test_data_dir ./data

```

-  For LoRA: 
```
python eval_lora.py --model_dir ./outputs/$model_dir --test_data_dir ./data

```

A comparison of all continual learning strategies can be found below. 

Method | Perplexity | Forward Gen. | Best Case | Worst Case 
--- | :---: | :---: | :---: |:---:
Sequential FT | 34\% $\uparrow$ | 33\% $\uparrow$ | 4.58 (1750--99) | 6.64 (1950--2000) 
EWC           | 12\% $\uparrow$ | 29\% $\uparrow$ | 4.65 (1800--49) | 6.77 (1950--2000) 
LoRA          | 15\% $\uparrow$ | 27\% $\uparrow$ | 4.48 (1850--99) | 6.19 (1950--2000) 


## Credits 
* Our implentation of cade is inspired from the official repository of [CADE]([https://github.com/mgermain/MADE/tree/master](https://github.com/vinid/cade)). 
