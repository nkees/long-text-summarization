# long-text-summarization
Code to my Talk "Summarizing Large Texts – a Deep Dive into NLP with Bidirectional Encoders", Predictive Analytics World (PAW) Conference, November 17, 2020

Slides to the Talk: tba.
Authors: Nataliia Kees (qdive GmbH), Clemens Kraus (qdive GmbH) @ck-real

https://predictiveanalyticsworld.de/programm/


## Install
Run `requirements.txt` in your virtual environment. This is perfectly compatible with Python 3.6.

## Usage
1 . Make sure you have collected all the narratives to be summarized in `bertabs/data`. The summaries will be created in the same folder is separate file, with suffix `_summary`.
When using the heuristics, you can change the document directory to some other one, in our case we called it "experiments".
2 . Run the following command to produce the summaries
```
python run_summarization.py --documents_dir data --no_cuda true --batch_size 4 --min_length 50 --max_length 200 --beam_size 5 --alpha 0.95 --block_trigram true
```
With heuristics:
```
python heuristics.py --documents_dir experiments --no_cuda true --batch_size 4 --min_length 50 --max_length 200 --beam_size 5 --alpha 0.95 --block_trigram true --heuristic 1
```

3. Evaluation
Replace 1 with the number of heuristic you want to evaluate.
```
python evaluate_summaries.py --heuristic 1
```

When trying to produce summaries with a CPU only, keep in mind that this will take some time (about a 100 summaries/hour on a normal computer).

## Credits
The system was created from the PreSumm model developed by Liu & Lapata 2019: https://www.paperswithcode.com/paper/text-summarization-with-pretrained-encoders.
It uses the encoders, pre-trained on BERT, trains them to do the extractive summarization task, and on top of this trains them to do the abstractive summarization task, so that the abstractive model can benefit from the extractive one and become even more performant.
The model was trained on CNN news data. 
The performance is state-of-the-art. 

The existing app has been built upon the abovementioned structures, using them as a tool for processing the data for tokenization and to produce summaries.

More credits:
- Transformer library with its text summarization example: https://github.com/huggingface/transformers/tree/master/examples/seq2seq/bertabs
- MMR implementation: https://github.com/swisscom/ai-research-keyphrase-extraction
- Sentence Encoders: https://github.com/UKPLab/sentence-transformers 