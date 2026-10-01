# Retrieval evaluation

23 answerable + 5 off-topic questions, model `sentence-transformers/all-MiniLM-L6-v2`, MIN_SCORE=0.3

| strategy | size | overlap | chunks | Hit@4 | MRR | Off-topic refusal |
|---|---|---|---|---|---|---|
| fixed | 200 | 40 | 41 | 100% | 0.949 | 80% |
| fixed | 500 | 100 | 19 | 96% | 0.870 | 80% |
| fixed | 1000 | 200 | 9 | 100% | 0.859 | 80% |
| sentence | 250 | 50 | 33 | 100% | 0.971 | 80% |
| sentence | 500 | 100 | 19 | 100% | 0.880 | 80% |
| sentence | 1000 | 200 | 9 | 100% | 0.859 | 80% |
