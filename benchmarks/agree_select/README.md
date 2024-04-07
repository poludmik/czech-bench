# AGREE dataset (verb selection task)

The dataset was obtained from it's official [site](https://nlp.fi.muni.cz/~xbaisa/agree/).

The data were transformed to accommodate a missing word selection task. 
Sentences containing more than one marked verb were discarded. 
In the remaining sentences, the marked verb was completely replaced with the "____" token. 
All five possible verb variants formed the list of available choices. 
Index of the correct choice was stored as the label.

- Language: CS (Original)
- Task: Language proficiency: subject-verb agreement
- Samples: 673 (Test set)
- Few-shot examples: 5 (From validation set)

## References

[1] Baisa, [Byte Level Language Models](https://is.muni.cz/th/139654/fi_d/), 2016
