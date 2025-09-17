# Official Repository of ChronoBerg

We introduce \CHRONOBERG, a temporally structured corpus of English book texts spanning 250 years, curated from Project Gutenberg and enriched with a variety of temporal annotations. We also introduce historically calibrated affective Valence-Arousal-Dominance (VAD)  lexicons to support temporally grounded interpretation. With the lexicons at hand, we demonstrate a need for modern LLM-based tools to better situate their detection of discriminatory language and contextualization of sentiment across various time-periods. In fact, we show how language models trained sequentially on \CHRONOBERG struggle to encode diachronic shifts in meaning, emphasizing the need for temporally aware training and evaluation pipelines, and positioning \CHRONOBERG as a scalable resource for the study of linguistic change and temporal generalization. $${\color{red}Disclaimer: }$$ This repository and dataset includes language and display of samples that could be offensive to readers. 

![ChronoBerg](https://github.com/paulsubarna/Chronoberg/blob/main/figures/chrono_flow.png)

To install the dependencies, run <pre/>pip install -r requirements.txt</pre> 

Catalog:
- [x] ChronoBerg: Dataset and Statistics
- [x] Lexical Analysis
- [x] Hate Speech Detection Models
- [x] Sequential training of LLMs on ChronoBerg

## Dataset
The dataset is available at Huggingface [ChronoBerg](https://huggingface.co/datasets/sdp56/ChronoBerg/tree/main)
Dataset Catalog:
- [x] ChronoBerg raw non-annotated [non-annotated](https://huggingface.co/datasets/sdp56/ChronoBerg/tree/main/dataset)
- [x] ChronoBerg: sentence-level valence annotated (for each time interval: 50 year span) [annotated](https://huggingface.co/datasets/sdp56/ChronoBerg/tree/main/dataset)
- [x] [Valence Lexicons](https://huggingface.co/datasets/sdp56/ChronoBerg/tree/main/lexicons) 
- [x] [Dominance Lexicons](https://huggingface.co/datasets/sdp56/ChronoBerg/tree/main/lexicons)
- [x] [Arousal Lexicons](https://huggingface.co/datasets/sdp56/ChronoBerg/tree/main/lexicons)

## Lexical Analysis
![Lexical](https://github.com/paulsubarna/Chronoberg/blob/main/figures/lexical_analysis.png)

## Sequential training LLMS

Method | Perplexity | Forward Gen. | Best Case | Worst Case 
--- | :---: | :---: | :---: |:---:
Sequential FT | 34\% $\uparrow$ | 33\% $\uparrow$ | 4.57 (1750--99) | 6.77 (1950--2000) 
EWC           | 12\% $\uparrow$ | 29\% $\uparrow$ | 4.78 (1800--49) | 5.77 (1950--2000) 
LoRA          | 15\% $\uparrow$ | 22\% $\uparrow$ | 4.81 (1850--99) | 5.89 (1950--2000) 
