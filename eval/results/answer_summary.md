# 回答生成 hallucination評価サマリー

- 問題数: 15（answerable 12 / trap 3）
- groundedness率: 1.0（supported 69 / unsupported 0 / claims合計 69）
- citation precision: 1.0（valid 69 / checked合計 69）
- trap abstention accuracy: 0.6667（3問中の正答率）
- answerable問での不必要な回答拒否: 3件

## 質問別内訳

| id | answerable | claims | supported | unsupported | citations(valid/checked) | abstained | abstain_correct |
|---|---|---|---|---|---|---|---|
| q001 | True | 1 | 1 | 0 | 1/1 | False | True |
| q003 | True | 0 | 0 | 0 | 0/0 | True | False |
| q006 | True | 8 | 8 | 0 | 8/8 | False | True |
| q007 | True | 10 | 10 | 0 | 10/10 | False | True |
| q009 | True | 5 | 5 | 0 | 5/5 | False | True |
| q013 | True | 5 | 5 | 0 | 5/5 | False | True |
| q017 | True | 0 | 0 | 0 | 0/0 | True | False |
| q021 | True | 6 | 6 | 0 | 6/6 | False | True |
| q026 | True | 10 | 10 | 0 | 10/10 | False | True |
| q028 | True | 6 | 6 | 0 | 6/6 | False | True |
| q033 | True | 0 | 0 | 0 | 0/0 | True | False |
| q040 | True | 9 | 9 | 0 | 9/9 | False | True |
| t01 | False | 0 | 0 | 0 | 0/0 | True | True |
| t02 | False | 5 | 5 | 0 | 5/5 | False | False |
| t03 | False | 4 | 4 | 0 | 4/4 | True | True |
